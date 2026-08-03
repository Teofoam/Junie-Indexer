#!/usr/bin/env python3
"""
query_test.py -- Sanity-check retrieval + citation quality before the MCP step.

Embeds your query (search_query: prefix), pulls the top hits from Chroma, and
prints each one the way Junie will cite it. If these look right, the index is
ready to wire to the MCP server.

  python query_test.py "what is the epipolar constraint" -k 5
"""
import argparse
from pathlib import Path

import chromadb

from embedder import get_embedder

COLLECTION = "library"


def cite(m: dict) -> str:
    if "page_start" in m:
        loc = f"p. {m['page_start']}" + (
            f"-{m['page_end']}" if m.get("page_end") != m.get("page_start") else "")
    elif "time_start" in m:
        loc = m["time_start"] + (f"-{m['time_end']}" if m.get("time_end") else "")
    else:
        loc = ""
    part = f" -- {m['part']}" if m.get("part") else ""
    sec = f" -- {m['section']}" if m.get("section") else ""
    tail = f" ({loc})" if loc else ""
    return f"{m.get('title', '?')}{part}{sec}{tail}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--chroma", type=Path, default=Path("chroma"))
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--backend", default="st", choices=["st", "ollama"])
    ap.add_argument("--device",  default="auto", choices=["auto", "cuda", "cpu"])
    a = ap.parse_args()

    col = chromadb.PersistentClient(path=str(a.chroma)).get_or_create_collection(
        COLLECTION, metadata={"hnsw:space": "cosine"})
    embedder = get_embedder(a.backend, device=a.device)
    res = col.query(query_embeddings=[embedder.embed_query(a.query)], n_results=a.k)

    for doc, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        print(f"\n[score {1 - dist:.3f}]  {cite(m)}")
        print("   " + " ".join(doc[:300].split()) + " ...")


if __name__ == "__main__":
    main()
