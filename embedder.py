#!/usr/bin/env python3
"""
embedder.py -- One embedding interface, two backends.

    st      sentence-transformers, in-process, no daemon   (default)
    ollama  HTTP to a local Ollama server                  (legacy route)

Both use nomic-embed-text v1.5 and both apply the task prefixes the model
requires: `search_document:` when indexing, `search_query:` when searching.
Skipping the prefixes silently degrades recall, so they are applied here rather
than left to each caller.

CRITICAL: index and query must use the SAME backend and model. Ollama serves a
quantized GGUF build; sentence-transformers runs full precision. The vectors are
close but not identical, and mixing them quietly costs you retrieval quality
with no error message. Pick one and use it on both host and VM.
"""
from __future__ import annotations
import os

DEFAULT_MODEL = "nomic-ai/nomic-embed-text-v1.5"   # HF id
OLLAMA_MODEL  = "nomic-embed-text"                 # Ollama tag
DOC_PREFIX, QUERY_PREFIX = "search_document: ", "search_query: "


class STEmbedder:
    """sentence-transformers, loaded in-process. No server, no daemon.

    device="auto" picks CUDA when present. Use device="cpu" to keep the GPU
    completely free (fine for the VM and for single queries; slower for bulk).
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "auto",
                 batch_size: int = 32):
        from sentence_transformers import SentenceTransformer
        import torch

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device, self.batch_size = device, batch_size

        try:                      # newer transformers/ST don't need this flag
            self.model = SentenceTransformer(model_name, device=device)
        except (ValueError, TypeError):
            self.model = SentenceTransformer(model_name, device=device,
                                             trust_remote_code=True)

    def _encode(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(
            texts, batch_size=self.batch_size, normalize_embeddings=True,
            show_progress_bar=False, convert_to_numpy=True,
        ).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode([DOC_PREFIX + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._encode([QUERY_PREFIX + text])[0]

    def unload(self) -> None:
        """Drop the model and free VRAM (useful before launching Marker)."""
        import torch
        del self.model
        if self.device == "cuda":
            torch.cuda.empty_cache()


class OllamaEmbedder:
    """HTTP to a local Ollama server. Requires `ollama pull nomic-embed-text`."""

    def __init__(self, model_name: str = OLLAMA_MODEL,
                 url: str = "http://localhost:11434", **_):
        import requests
        self._requests, self.model_name, self.url = requests, model_name, url.rstrip("/")

    def _one(self, prompt: str) -> list[float]:
        r = self._requests.post(f"{self.url}/api/embeddings",
                                json={"model": self.model_name, "prompt": prompt},
                                timeout=180)
        r.raise_for_status()
        return r.json()["embedding"]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._one(DOC_PREFIX + t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._one(QUERY_PREFIX + text)

    def unload(self) -> None:
        pass


def get_embedder(backend: str = "", model: str = "", device: str = "auto",
                 batch_size: int = 32):
    """Build an embedder. Env vars let the MCP server be configured without code.

        LIBRARY_EMBED_BACKEND = st | ollama      (default: st)
        LIBRARY_EMBED_MODEL   = model id/tag
        LIBRARY_EMBED_DEVICE  = auto | cpu | cuda
    """
    backend = (backend or os.environ.get("LIBRARY_EMBED_BACKEND", "st")).lower()
    device = device or os.environ.get("LIBRARY_EMBED_DEVICE", "auto")

    if backend in ("st", "sentence-transformers", "hf"):
        return STEmbedder(model or os.environ.get("LIBRARY_EMBED_MODEL", DEFAULT_MODEL),
                          device=device, batch_size=batch_size)
    if backend == "ollama":
        return OllamaEmbedder(model or os.environ.get("LIBRARY_EMBED_MODEL", OLLAMA_MODEL),
                              url=os.environ.get("OLLAMA_URL", "http://localhost:11434"))
    raise ValueError(f"Unknown embedding backend: {backend!r} (use 'st' or 'ollama')")


if __name__ == "__main__":                      # smoke test
    import sys
    e = get_embedder(sys.argv[1] if len(sys.argv) > 1 else "")
    v = e.embed_query("epipolar constraint")
    print(f"{type(e).__name__}: dim={len(v)}  first={v[:4]}")
