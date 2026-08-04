#!/usr/bin/env python3
"""
chunk_and_index.py -- Semantic chunk + embed + store for Junie's library.

Reads Marker-produced Markdown (with --paginate_output page markers) and
Whisper transcripts (with [mm:ss] timestamp markers), splits them into
SECTION-AWARE chunks that carry {title, author, source_id, part, section,
page_start/page_end  OR  time_start/time_end} metadata, embeds each chunk
locally with nomic-embed-text v1.5 through embedder.py (sentence-transformers
by default, --backend ollama for the legacy route; the search_document: prefix
is applied there), and writes everything into a persistent ChromaDB collection.

Runs in the JunieAirport env on the Windows host.

SERIES SUPPORT: a manifest `filename` may be a glob (e.g. "Lecture 13-*.mp4"),
so one row covers a whole lecture course. Each matched file becomes a distinct
`part` under the shared source_id, keeping per-file timestamps meaningful and
chunk ids unique.

  python chunk_and_index.py --md md --transcripts transcripts \
                            --manifest manifest.csv --chroma chroma
"""
from __future__ import annotations
import argparse, csv, datetime, fnmatch, json, os, re, sys
from pathlib import Path

import chromadb
from tqdm import tqdm

from embedder import DEFAULT_MODEL, OLLAMA_MODEL, get_embedder

# ---- tunables -------------------------------------------------------------
TARGET_WORDS  = 380      # ~500 tokens/chunk (nomic context is 8192, lots of headroom)
OVERLAP_WORDS = 60       # ~80-token overlap carried between chunks in the same section
COLLECTION    = "library"
ADD_BATCH     = 256

# Marker page marker:  {12}------------------------------------------------
PAGE_MARKER = re.compile(r'^\s*\{?(\d+)\}?-{10,}\s*$')
# Whisper timestamp line:  [00:12:30] text...  or  [12:30] text...
TIME_MARKER = re.compile(r'^\s*\[(\d{1,2}:\d{2}(?::\d{2})?)\]')
HEADING     = re.compile(r'^(#{1,4})\s+(.*)$')
# ---------------------------------------------------------------------------


def slug(s: str) -> str:
    return re.sub(r'-+', '-', re.sub(r'[^a-z0-9]+', '-', s.lower())).strip('-')


def load_manifest(path: Path) -> tuple[dict, list]:
    """Returns (exact_map, glob_rules).

    exact_map:  stem.lower() -> row          (single-file sources)
    glob_rules: [(pattern_stem.lower(), row)] (series; longest pattern wins)
    """
    exact, globs = {}, []
    if not path.exists():
        print(f"  (no manifest at {path}; falling back to filenames for titles)")
        return exact, globs

    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            fn = (row.get("filename") or "").strip()
            sid = (row.get("source_id") or "").strip()
            if not fn or not sid:
                continue                       # skip blank/incomplete rows
            key = Path(fn).stem.lower()        # match on stem: .mp4 -> .md is fine
            if any(ch in key for ch in "*?["):
                globs.append((key, row))
            else:
                exact[key] = row
    globs.sort(key=lambda kv: len(kv[0]), reverse=True)   # most specific first
    return exact, globs


def lookup(stem: str, exact: dict, globs: list) -> tuple[dict, bool]:
    """Resolve a file to its manifest row. Returns (row, is_series)."""
    k = stem.lower()
    if k in exact:
        return exact[k], False
    for pattern, row in globs:
        if fnmatch.fnmatch(k, pattern):
            return row, True
    return {}, False


def parse_segments(text: str) -> list[dict]:
    """Tag every paragraph with its section path and the page/time it sits under."""
    segments, heading, para = [], [], []
    page, time_s = None, None

    def section() -> str:
        return " > ".join(t for _, t in heading)

    def push():
        nonlocal para
        joined = " ".join(l.strip() for l in para).strip()
        para = []
        if joined:
            segments.append({"section": section(), "page": page,
                             "time": time_s, "text": joined})

    for raw in text.splitlines():
        line = raw.rstrip()

        mp = PAGE_MARKER.match(line)
        if mp:
            push(); page = int(mp.group(1)); continue

        mt = TIME_MARKER.match(line)
        if mt:
            push(); time_s = mt.group(1)
            rest = line[mt.end():].strip()
            if rest:
                para.append(rest)
            continue

        mh = HEADING.match(line)
        if mh:
            push()
            level, title = len(mh.group(1)), mh.group(2).strip()
            while heading and heading[-1][0] >= level:
                heading.pop()
            heading.append((level, title))
            continue

        if not line.strip():
            push()
        else:
            para.append(line)
    push()
    return segments


def pack(segments: list[dict]) -> list[dict]:
    """Group contiguous same-section segments into ~TARGET_WORDS chunks with a
    small word-level overlap. Never crosses a section boundary."""
    chunks = []
    cur_sec = None
    words, pages, times = [], [], []

    def flush():
        nonlocal words, pages, times
        if words:
            chunks.append({
                "section": cur_sec or "",
                "text": " ".join(words),
                "page_start": min(pages) if pages else None,
                "page_end":   max(pages) if pages else None,
                "time_start": times[0]   if times else None,
                "time_end":   times[-1]  if times else None,
            })
        words, pages, times = [], [], []

    for seg in segments:
        if seg["section"] != cur_sec:
            flush(); cur_sec = seg["section"]
        sw = seg["text"].split()
        if words and len(words) + len(sw) > TARGET_WORDS:
            tail = words[-OVERLAP_WORDS:] if OVERLAP_WORDS else []
            flush()
            words = list(tail)
        words.extend(sw)
        if seg["page"] is not None: pages.append(seg["page"])
        if seg["time"] is not None: times.append(seg["time"])
    flush()
    return chunks


def process_file(path: Path, default_kind: str, exact: dict, globs: list, col, embedder,
                 sources: dict | None = None) -> int:
    meta, is_series = lookup(path.stem, exact, globs)
    title  = meta.get("title") or path.stem
    author = meta.get("author", "")
    src_id = meta.get("source_id") or slug(path.stem)
    kind   = meta.get("type") or default_kind

    # A series row covers many files: each file is a `part`, which keeps
    # timestamps scoped to their own lecture AND keeps chunk ids unique.
    part = path.stem.strip() if is_series else ""
    doc_id = f"{src_id}::{slug(part)}" if part else src_id

    chunks = pack(parse_segments(path.read_text(encoding="utf-8", errors="ignore")))

    ids, docs, metas = [], [], []
    for k, c in enumerate(chunks):
        if len(c["text"].split()) < 5:
            continue
        ids.append(f"{doc_id}::{k:05d}")
        docs.append(c["text"])
        m = {"source_id": src_id, "title": title, "author": author,
             "kind": kind, "section": c["section"]}
        if part:
            m["part"] = part
        if c["page_start"] is not None:            # Chroma rejects None values
            m["page_start"], m["page_end"] = c["page_start"], c["page_end"]
        if c["time_start"] is not None:
            m["time_start"], m["time_end"] = c["time_start"], c["time_end"]
        metas.append(m)

    for s in range(0, len(ids), ADD_BATCH):
        sl = slice(s, s + ADD_BATCH)
        col.add(ids=ids[sl], documents=docs[sl],
                embeddings=embedder.embed_documents(docs[sl]), metadatas=metas[sl])

    if sources is not None and ids:
        e = sources.setdefault(src_id, {"source_id": src_id, "title": title,
                                        "author": author, "kind": kind, "n": 0})
        e["n"] += len(ids)

    return len(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md",          type=Path, default=Path("md"))
    ap.add_argument("--transcripts", type=Path, default=Path("transcripts"))
    ap.add_argument("--manifest",    type=Path, default=Path("manifest.csv"))
    ap.add_argument("--chroma",      type=Path, default=Path("chroma"))
    ap.add_argument("--dry-run",     action="store_true",
                    help="report how each file resolves against the manifest; no embedding")
    ap.add_argument("--backend",     default="st", choices=["st", "ollama"],
                    help="st = sentence-transformers in-process (default); ollama = HTTP daemon")
    ap.add_argument("--device",      default="auto", choices=["auto", "cuda", "cpu"],
                    help="st backend only; cpu keeps the GPU entirely free")
    ap.add_argument("--batch-size",  type=int, default=32)
    args = ap.parse_args()

    exact, globs = load_manifest(args.manifest)

    def _not_scratch(p: Path, root: Path) -> bool:
        # skip _underscore dirs (e.g. a crashed extract run's temp folders)
        return not any(part.startswith("_") for part in p.relative_to(root).parts)

    files = []
    if args.md.exists():
        files += [(p, "book")    for p in sorted(args.md.rglob("*.md")) if _not_scratch(p, args.md)]
    if args.transcripts.exists():
        files += [(p, "lecture") for p in sorted(args.transcripts.rglob("*.md")) if _not_scratch(p, args.transcripts)]
    if not files:
        sys.exit("No .md found in md/ or transcripts/. Run extract.py / transcribe.py first.")

    if args.dry_run:
        unmatched = 0
        for path, kind in files:
            meta, is_series = lookup(path.stem, exact, globs)
            if meta:
                tag = f"{meta['source_id']}" + (f"  part={path.stem}" if is_series else "")
                print(f"  OK   {path.name}  ->  {tag}")
            else:
                unmatched += 1
                print(f"  ??   {path.name}  ->  (no manifest row; will cite by filename)")
        print(f"\n{len(files)} file(s), {unmatched} unmatched.")
        return

    embedder = get_embedder(args.backend, device=args.device, batch_size=args.batch_size)
    dev = getattr(embedder, "device", "n/a")
    print(f"embedding backend: {args.backend} (device={dev})")

    col = chromadb.PersistentClient(path=str(args.chroma)).get_or_create_collection(
        COLLECTION, metadata={"hnsw:space": "cosine"})

    sources: dict = {}
    total = 0
    for path, kind in tqdm(files, desc="indexing", unit="doc"):
        try:
            total += process_file(path, kind, exact, globs, col, embedder, sources)
        except Exception as e:
            print(f"  !! {path.name}: {e}", file=sys.stderr)

    # ---- provenance sidecar -------------------------------------------------
    # library_mcp.py reads this at startup and refuses to serve on a mismatch.
    # It also gives list_sources() a free source table instead of a full scan.
    model_id = os.environ.get("LIBRARY_EMBED_MODEL") or (
        OLLAMA_MODEL if args.backend == "ollama" else DEFAULT_MODEL)
    meta = {
        "embed_model": model_id,
        "backend": args.backend,
        "dim": len(embedder.embed_query("dimension probe")),
        "prefix_document": "search_document: ",
        "prefix_query": "search_query: ",
        "normalized": True,
        "distance": "cosine",
        "collection": COLLECTION,
        "chromadb": chromadb.__version__,
        "target_words": TARGET_WORDS,
        "overlap_words": OVERLAP_WORDS,
        "n_chunks": col.count(),
        "n_files": len(files),
        "built_at": datetime.datetime.now(datetime.timezone.utc)
                    .isoformat(timespec="seconds"),
        "sources": sorted(sources.values(), key=lambda e: e["source_id"]),
    }
    meta_path = Path(args.chroma) / "index_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                         encoding="utf-8")

    print(f"\nDone. {total} chunks from {len(files)} files -> {args.chroma}/")
    print(f"Wrote {meta_path}  (model={model_id}, dim={meta['dim']}, "
          f"chromadb={meta['chromadb']}) — copy it to the VM with the index.")


if __name__ == "__main__":
    main()
