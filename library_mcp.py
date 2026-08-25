#!/usr/bin/env python3
"""
library_mcp.py -- MCP server exposing Junie's library as scoped search tools.

Runs on the VM (Linux Mint), next to the Chroma index copied over from the
host build. OpenClaw launches it over stdio; Junie sees three read-only tools:

  search_library(query, k, kind, source_id)  -> cited snippets
  get_passage(chunk_id, before, after)       -> expand a hit with neighbours
  list_sources()                             -> what's actually in the library

Nothing here can write, execute, or reach the network at all once the embedding
model is cached locally, so it slots in without reopening `exec` or the browser.

Deps (in the VM):  pip install "mcp[cli]" chromadb sentence-transformers
Embeddings run in-process on CPU (no Ollama daemon). Pre-download the model
once while online:  hf download nomic-ai/nomic-embed-text-v1.5

STARTUP CONTRACT
----------------
This server refuses to start rather than serve a library that isn't really
there. It exits non-zero if any of these fail:

  * LIBRARY_CHROMA_PATH is not an existing directory
  * the collection does not already exist  (get_collection, never create)
  * the collection is empty
  * the query embedder's dimension != the indexed vectors' dimension
  * index_meta.json disagrees about model/backend  (warning unless strict)

The reason this matters more than usual: search_library tells Junie to say
"not in the library" on an empty result and NOT to answer from memory. That is
correct when the topic genuinely isn't covered -- and catastrophic when the
disk simply isn't mounted, because she will faithfully deny all 116 sources.
A silent empty index turns her honesty rule into a lie. So: fail loudly here.

Env vars:
  LIBRARY_CHROMA_PATH   path to the copied chroma/ directory
  LIBRARY_COLLECTION    collection name (default: library)
  LIBRARY_EMBED_*       backend/model/device -- see embedder.py
  LIBRARY_ALLOW_EMPTY=1 downgrade the empty-index check to a warning (testing)
  LIBRARY_STRICT_META=0 downgrade metadata mismatches to warnings
  LIBRARY_MIN_SCORE     relevance floor for search hits (default 0.6)
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path
from typing import Any

import chromadb
from mcp.server.fastmcp import FastMCP

from embedder import DEFAULT_MODEL, OLLAMA_MODEL, get_embedder

CHROMA_PATH   = os.environ.get("LIBRARY_CHROMA_PATH", "/mnt/library/chroma")
COLLECTION    = os.environ.get("LIBRARY_COLLECTION", "library")
META_FILE     = Path(CHROMA_PATH) / "index_meta.json"
MAX_K         = 12
SNIPPET_CHARS = 1200
SCAN_PAGE     = 10_000          # fallback source-scan page size

# Relevance floor. Vector search is nearest-neighbour, not matching: it always
# returns k results, however far away they are. Without a floor, asking about a
# topic the library does not cover returns the least-irrelevant passage in it,
# and the "no results means say so" contract below can never fire on an
# unfiltered query -- turning the honesty rule into a source of confident
# nonsense. Measured on this index: genuine hits score 0.72-0.86, while an
# off-topic control ("baroque harpsichord tuning") pulled Fourier transforms at
# 0.575. 0.6 sits in that gap. Raise it if unrelated passages still get through.
MIN_SCORE     = float(os.environ.get("LIBRARY_MIN_SCORE", "0.6"))

ALLOW_EMPTY   = os.environ.get("LIBRARY_ALLOW_EMPTY", "0") not in ("0", "", "false", "False")
STRICT_META   = os.environ.get("LIBRARY_STRICT_META", "1") not in ("0", "", "false", "False")


# --------------------------------------------------------------------------
# startup: prove the library is real before exposing a single tool
# --------------------------------------------------------------------------

def _log(msg: str) -> None:
    """stdout is the MCP transport -- diagnostics must go to stderr."""
    print(f"library_mcp: {msg}", file=sys.stderr)


def _die(msg: str, hint: str = "") -> None:
    _log(f"FATAL: {msg}")
    if hint:
        _log(f"       {hint}")
    raise SystemExit(2)


def _open_collection():
    """get_collection, NEVER get_or_create.

    PersistentClient happily creates the directory, and get_or_create then makes
    an empty `library` inside it. An unmounted Cozy-Library disk or a path off
    by one folder would produce a server that starts clean and serves nothing.
    Read-only code must not be able to author state.
    """
    if not Path(CHROMA_PATH).is_dir():
        _die(f"LIBRARY_CHROMA_PATH is not a directory: {CHROMA_PATH}",
             "Is the Cozy-Library disk mounted? Is the path off by one folder?")

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    try:
        return client.get_collection(COLLECTION)
    except Exception as e:
        try:
            names = [c.name for c in client.list_collections()]
        except Exception:
            names = []
        _die(f"collection {COLLECTION!r} not found at {CHROMA_PATH} ({e})",
             f"collections present: {names or '(none -- wrong directory)'}")


def _indexed_dim(col) -> int | None:
    """Dimension of the vectors actually stored, read straight from the index."""
    try:
        got = col.get(limit=1, include=["embeddings"])
        embs = got.get("embeddings")
        if embs is not None and len(embs):
            return len(embs[0])
    except Exception as e:
        _log(f"warning: could not read a stored embedding ({e})")
    return None


def _load_meta() -> dict:
    if not META_FILE.exists():
        _log(f"warning: no index_meta.json beside the index ({META_FILE}).")
        _log("         Model/backend agreement cannot be verified; dimension is still checked.")
        return {}
    try:
        return json.loads(META_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        _log(f"warning: index_meta.json unreadable ({e}); skipping metadata checks.")
        return {}


def _resolve_model_id(backend: str) -> str:
    """Same resolution order get_embedder uses, so we can report what we loaded."""
    return os.environ.get("LIBRARY_EMBED_MODEL") or (
        OLLAMA_MODEL if backend == "ollama" else DEFAULT_MODEL)


_col = _open_collection()

_n_chunks = _col.count()
if _n_chunks == 0:
    if ALLOW_EMPTY:
        _log("warning: index is EMPTY (LIBRARY_ALLOW_EMPTY set, continuing anyway)")
    else:
        _die(f"collection {COLLECTION!r} exists but contains 0 chunks",
             "Was the index copied, or only the empty directory? "
             "Set LIBRARY_ALLOW_EMPTY=1 to start anyway.")

_meta = _load_meta()

# Loaded once at startup; the model stays warm for the life of the server.
_backend = (os.environ.get("LIBRARY_EMBED_BACKEND", "st")).lower()
_embedder = get_embedder(device=os.environ.get("LIBRARY_EMBED_DEVICE", "cpu"))
_model_id = _resolve_model_id(_backend)

# Dimension is the one check that needs no sidecar file: ask the index directly.
_probe_dim = len(_embedder.embed_query("dimension probe"))
_index_dim = _indexed_dim(_col)
if _index_dim is not None and _index_dim != _probe_dim:
    _die(f"embedding dimension mismatch: index holds {_index_dim}-d vectors, "
         f"this embedder produces {_probe_dim}-d",
         "Different model, or Matryoshka truncation applied on only one side. "
         "Retrieval would return plausible-looking nonsense, so refusing to serve.")

# Model/backend agreement needs index_meta.json, written by chunk_and_index.py.
if _meta:
    problems = []
    if _meta.get("embed_model") and _meta["embed_model"] != _model_id:
        problems.append(f"model: indexed with {_meta['embed_model']!r}, querying with {_model_id!r}")
    if _meta.get("backend") and _meta["backend"] != _backend:
        problems.append(
            f"backend: indexed with {_meta['backend']!r}, querying with {_backend!r} "
            "(Ollama serves a quantized GGUF; sentence-transformers is full precision)")
    idx_chroma = str(_meta.get("chromadb", ""))
    if idx_chroma and idx_chroma.split(".")[0] != chromadb.__version__.split(".")[0]:
        problems.append(f"chromadb major: index built with {idx_chroma}, "
                        f"reading with {chromadb.__version__}")
    if problems:
        for p in problems:
            _log(("FATAL: " if STRICT_META else "warning: ") + p)
        if STRICT_META:
            _die("index/query configuration disagree",
                 "Re-index, or set LIBRARY_STRICT_META=0 if you are certain.")

_log(f"ready: {_n_chunks} chunks, {_probe_dim}-d, model={_model_id}, "
     f"backend={_backend}, min_score={MIN_SCORE}, chroma={CHROMA_PATH}")

mcp = FastMCP("library")

_sources_cache: list[dict] | None = None


def _embed_query(q: str) -> list[float]:
    """Prefixing is handled inside the embedder (search_query: for lookups)."""
    return _embedder.embed_query(q)


def _citation(m: dict[str, Any]) -> str:
    """Render the metadata the indexer stamped into a human citation."""
    bits = [m.get("title") or m.get("source_id") or "unknown source"]
    if m.get("author"):
        bits[0] = f"{m['author']}, {bits[0]}"
    if m.get("part"):                    # which lecture within a series
        bits.append(m["part"])
    if m.get("section"):
        bits.append(m["section"])
    if m.get("page_start") is not None:
        p = f"p. {m['page_start']}"
        if m.get("page_end") not in (None, m.get("page_start")):
            p = f"pp. {m['page_start']}-{m['page_end']}"
        bits.append(p)
    elif m.get("time_start"):
        t = m["time_start"]
        if m.get("time_end") and m["time_end"] != t:
            t = f"{t}-{m['time_end']}"
        bits.append(t)
    return " — ".join(bits)


def _scan_sources() -> list[dict]:
    """Fallback when index_meta.json has no source table: page through the
    collection once, memory-bounded, and cache the result for this process."""
    seen: dict[str, dict] = {}
    offset = 0
    while True:
        got = _col.get(include=["metadatas"], limit=SCAN_PAGE, offset=offset)
        metas = got.get("metadatas") or []
        if not metas:
            break
        for m in metas:
            sid = (m or {}).get("source_id")
            if not sid:
                continue
            e = seen.setdefault(sid, {"source_id": sid, "title": m.get("title", sid),
                                      "author": m.get("author", ""),
                                      "kind": m.get("kind", ""), "n": 0})
            e["n"] += 1
        offset += len(metas)
        if len(metas) < SCAN_PAGE:
            break
    return list(seen.values())


def _sources() -> list[dict]:
    """Source table: from index_meta.json if present (free), else one cached scan."""
    global _sources_cache
    if _sources_cache is None:
        if _meta.get("sources"):
            _sources_cache = list(_meta["sources"])
        else:
            _log("no source table in index_meta.json; scanning the collection once")
            _sources_cache = _scan_sources()
    return _sources_cache


@mcp.tool()
def search_library(query: str, k: int = 5, kind: str = "",
                   source_id: str = "") -> str:
    """Search Teofil's textbook/paper/lecture library for passages relevant to a
    question, and return them with exact citations (title, section, page or
    timestamp). Use this before answering any substantive technical question,
    and cite the returned sources rather than answering from memory.

    Args:
        query: A natural-language question or topic, e.g. "epipolar constraint
            derivation" or "why does XPBD converge". Full questions work better
            than single keywords.
        k: How many passages to return (1-12, default 5).
        kind: Optional filter — "book", "paper", or "lecture".
        source_id: Optional filter to a single source, e.g. "hartley-zisserman".
            Use list_sources() to see valid ids.

    Passages below a relevance floor are withheld, so "nothing relevant" is a
    real answer and not a failure: trust it and say the library does not cover
    the topic, rather than reaching for memory. Fewer results than k means the
    weaker ones were withheld, which is a hint that coverage is thin.
    """
    k = max(1, min(int(k), MAX_K))

    conds = []
    if kind:
        conds.append({"kind": kind})
    if source_id:
        conds.append({"source_id": source_id})
    where = conds[0] if len(conds) == 1 else ({"$and": conds} if conds else None)

    try:
        res = _col.query(query_embeddings=[_embed_query(query)],
                         n_results=k, where=where)
    except Exception as e:
        return f"Library search failed: {e}"

    ids   = res.get("ids", [[]])[0]
    docs  = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]

    filt = ""
    if kind or source_id:
        filt = f" (filtered to {kind or ''}{' ' if kind and source_id else ''}{source_id or ''})"

    if not ids:
        # Nothing came back at all: only possible when a kind/source_id filter
        # excluded everything, since the index is non-empty (checked at startup).
        return (f"No passages matched the filter{filt}. Retry without the filter, "
                "or call list_sources() for valid ids -- do not answer from memory.")

    # Chroma returns nearest-first, so anything under the floor is a suffix.
    hits = [(c, d_, m, 1 - dist) for c, d_, m, dist in zip(ids, docs, metas, dists)
            if 1 - dist >= MIN_SCORE]

    if not hits:
        best = 1 - dists[0]
        return (f"Nothing in the library is relevant to \"{query}\"{filt}. "
                f"The closest passage scored {best:.3f}, below the {MIN_SCORE} "
                "relevance floor, so it was withheld rather than shown as if it "
                "supported an answer. Say plainly that the library does not cover "
                "this, rather than answering from memory. If you believe it should "
                "be covered, retry with different wording or call list_sources().")

    dropped = len(ids) - len(hits)
    head = f'{len(hits)} passage(s) for "{query}"'
    if dropped:
        # Thin coverage is a signal worth passing on, not noise to hide.
        head += (f" ({dropped} weaker match(es) withheld below the "
                 f"{MIN_SCORE} floor -- coverage here may be thin)")
    out = [head + ":\n"]
    for cid, doc, m, score in hits:
        text = doc if len(doc) <= SNIPPET_CHARS else doc[:SNIPPET_CHARS] + " […]"
        out.append(f"--- [{score:.3f}] {_citation(m)}\n"
                   f"    chunk_id: {cid}\n{text}\n")
    out.append("Cite the source line above for anything you use. "
               "Call get_passage(chunk_id) to read more around a hit.")
    return "\n".join(out)


@mcp.tool()
def get_passage(chunk_id: str, before: int = 1, after: int = 1) -> str:
    """Expand a search hit by fetching the passages immediately around it in the
    original text. Use when a snippet is cut off mid-derivation or you need more
    surrounding context before explaining something.

    Args:
        chunk_id: The chunk_id from a search_library result, e.g. "szeliski-cv::00142".
        before: How many preceding passages to include (0-5).
        after: How many following passages to include (0-5).
    """
    try:
        src, idx_s = chunk_id.rsplit("::", 1)
        idx = int(idx_s)
    except ValueError:
        return f"Malformed chunk_id: {chunk_id!r} (expected 'source::00042')."

    before, after = max(0, min(before, 5)), max(0, min(after, 5))
    wanted = [f"{src}::{i:05d}" for i in range(max(0, idx - before), idx + after + 1)]

    got = _col.get(ids=wanted)
    if not got.get("ids"):
        return f"No passages found for {chunk_id}."

    order = {cid: n for n, cid in enumerate(wanted)}
    rows = sorted(zip(got["ids"], got["documents"], got["metadatas"]),
                  key=lambda r: order.get(r[0], 0))
    out = []
    for cid, doc, m in rows:
        mark = "  <-- requested" if cid == chunk_id else ""
        out.append(f"--- {_citation(m)}{mark}\n    chunk_id: {cid}\n{doc}\n")
    return "\n".join(out)


@mcp.tool()
def list_sources() -> str:
    """List every source in the library with its source_id, so you can tell the
    user what you do and don't have, and filter searches to one book. Use this
    when asked what's available, or before claiming a topic isn't covered."""
    try:
        srcs = _sources()
    except Exception as e:
        return f"Could not read the library: {e}"

    if not srcs:
        return ("The library index has no readable source metadata. Tell the user "
                "the library looks broken rather than answering as if it were empty.")

    lines = [f"{len(srcs)} source(s) indexed ({_n_chunks} passages total):"]
    for e in sorted(srcs, key=lambda x: (x.get("title") or "").lower()):
        who = f" — {e['author']}" if e.get("author") else ""
        lines.append(f"  {e['source_id']}: {e.get('title', e['source_id'])}{who} "
                     f"[{e.get('kind', '')}, {e.get('n', '?')} passages]")
    return "\n".join(lines)


if __name__ == "__main__":
    # All preflight checks already ran at import; reaching here means the
    # library is mounted, non-empty, and dimensionally consistent.
    mcp.run()
