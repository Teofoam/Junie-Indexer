# Junie's library — host-side ingestion pipeline

Turns your PDFs / EPUBs / lecture videos into a citation-ready, locally-searchable
knowledge base that Junie queries over one scoped MCP tool.

| | |
|---|---|
| **Code lives in** | `F:\Dev\Identities\Aluminum\Projects\JunieLib` (PyCharm project) |
| **Runs in** | the `JunieAirport` mamba env — Python 3.13, `torch 2.13.0+cu130`, RTX 5060 Laptop (8 GB) |
| **Ships to** | Junie's **Cozy-Library** disk on the Linux Mint VM |

The host does all the GPU work — Marker conversion, Whisper transcription, bulk
embedding — and hands the VM a finished index. The VM never touches the GPU.

## What each stage does

```
raw/ (PDF, EPUB) ──extract.py──▶ md/            (Marker: structured MD + page markers)
video/ (mp4…)   ──transcribe.py─▶ transcripts/  (Whisper: MD + [mm:ss] markers)
                                       │
                       chunk_and_index.py  (section-aware chunks + {title,section,page/time})
                                       │
                                       ▼
                             chroma/ + index_meta.json
```

Citations come out as **"Hartley & Zisserman — 11.1 Epipolar geometry (p. 239–241)"**
for books and **"Stachniss, Photogrammetry I (12:30–14:05)"** for lectures.

## Building the env

`JunieAirport` already exists. This is here so it can be rebuilt correctly.

```powershell
mamba create -n JunieAirport python=3.13
mamba activate JunieAirport

# torch FIRST, from the CUDA 13 index
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
python -c "import torch; print(torch.version.cuda, torch.cuda.get_device_capability(0))"
#  -> 13.0 (12, 0)

pip install marker-pdf chromadb sentence-transformers tqdm ^
            --extra-index-url https://download.pytorch.org/whl/cu130
python -c "import torch; print(torch.version.cuda)"   # still 13.0?

pip install faster-whisper
```

**Order matters.** `marker-pdf` depends on torch. Install it first and pip resolves
torch from PyPI as a transitive dependency; the later cu130 command then reports
"Requirement already satisfied" and silently does nothing. Torch first, everything
else after.

**Why the RTX 5060 forces CUDA 13.** Blackwell is compute capability 12.0 (`sm_120`).
Wheels built against CUDA 12.6 or older contain no kernels for it — they install
cleanly and fail at the first `.cuda()` call. CUDA 12.8 is the floor; cu130 is the
current top of PyTorch's build menu.

**Let mamba own only the interpreter.** `marker-pdf` isn't on conda-forge and its
graph drags in numpy, transformers, and onnxruntime. Splitting ownership of those
between mamba and pip produces ABI mismatches that surface as unreadable import
errors. Install everything above it with pip.

**Redirect the HuggingFace cache off C:** before pulling models, or several GB of
weights land on the wrong drive:

```powershell
[Environment]::SetEnvironmentVariable("HF_HOME", "F:\Dev\Cache\huggingface", "User")
```

New shell afterwards. Enabling Windows Developer Mode also restores symlink-based
deduplication in that cache, which meaningfully cuts its size.

## Embedding backend — pick one, use it everywhere

Both routes run the same model (nomic-embed-text v1.5) with the same
`search_document:` / `search_query:` prefixes. They differ only in how it's hosted.

### Route A — sentence-transformers (default, recommended)

Loads the model **in-process**. No daemon, no background service, and the model is
released the moment the script exits. `torch` is already in `JunieAirport`, so
`sentence-transformers` is the only addition.

```powershell
python chunk_and_index.py --backend st              # GPU if free, else CPU
python chunk_and_index.py --backend st --device cpu # leave the GPU untouched
```

Footprint is small: v1.5 is ~137M parameters, about **0.3 GB** in fp16 — next to
Marker's ~5 GB and Whisper large-v3's ~4.7 GB it is noise. The reason to prefer
this route is not memory, it's that nothing stays resident afterwards, and that
batching happens in-process instead of one HTTP round-trip per chunk.

### Route B — Ollama (legacy)

Kept working for anyone who already runs Ollama. It stands up a persistent
background server and keeps the model loaded for `keep_alive` (5 min default)
after each call.

```powershell
ollama pull nomic-embed-text
python chunk_and_index.py --backend ollama
```

### Do not mix them

Ollama serves a quantized GGUF build; sentence-transformers runs full precision.
The vectors are close but **not identical**, so indexing with one and querying with
the other degrades retrieval with no error message. `library_mcp.py` now refuses to
start on this mismatch rather than serving degraded results — but only if
`index_meta.json` travels with the index.

## Folder layout

```
JunieLib/
├── raw/            # your PDFs and EPUBs
├── video/          # lecture video/audio files
├── manifest.csv    # maps each file -> title / author / type (edit this!)
├── extract.py  transcribe.py  chunk_and_index.py  query_test.py
├── embedder.py     # shared embedding backend (st | ollama)
├── library_mcp.py  # VM-side MCP server (ships to Cozy-Library, not run here)
├── md/  transcripts/
└── chroma/
    └── index_meta.json    # written by chunk_and_index.py; must travel with chroma/
```

## Run order

1. **Fill `manifest.csv`.** One row per source: `source_id, filename, title, author, type`
   (`type` ∈ `book | paper | lecture`). `filename` may be a glob (`Lecture 13-*.mp4`)
   so one row covers a whole lecture series; each matched file becomes a `part`,
   which keeps timestamps scoped and chunk ids unique. Anything unlisted still
   indexes — it just cites by filename instead of a clean title.

2. **Extract** (run with the VM powered off to free its 12 GB):
   ```powershell
   python extract.py --raw raw --out md
   python extract.py --raw raw --out md --force-ocr   # for scanned books
   ```
   **Then verify the page markers before indexing anything.** Open a file in `md/`
   and confirm they look like `{N}----…`. This is `marker-pdf 2.0.0`, a major
   version ahead of what `PAGE_MARKER` was written against — treat the regex *and*
   the `--paginate_output` flag as unverified until you've seen real output. If the
   format changed, every page citation in the corpus is silently wrong, which looks
   exactly like success.

3. **Transcribe** (separately — shares the GPU with Marker):
   ```powershell
   python transcribe.py --video video --out transcripts
   ```
   Keep `compute_type="float16"`. `int8` fails on Blackwell with
   `CUBLAS_STATUS_NOT_SUPPORTED` — a CTranslate2/`sm_120` incompatibility, not a
   broken install.

4. **Chunk + embed + index:**
   ```powershell
   # check the manifest resolves first -- seconds, vs. an hour of wasted embedding
   python chunk_and_index.py --dry-run

   python chunk_and_index.py --md md --transcripts transcripts ^
                             --manifest manifest.csv --chroma chroma ^
                             --backend st
   ```
   Run **after** Marker and Whisper have finished — separate invocations, so the
   GPU is free by the time you get here. On the first small run, confirm the
   distance metric actually took: `metadata={"hnsw:space": "cosine"}` may be
   deprecated in chromadb 1.5.9 in favour of a `configuration=` argument. A
   deprecation warning is harmless; a silently-ignored setting gives you L2 and
   wrong-looking scores.

   This writes `chroma/index_meta.json` — model, dimension, prefixes, chromadb
   version, and the source table. It is not optional; the VM server reads it.

5. **Verify citations before moving on:**
   ```powershell
   python query_test.py "what is the epipolar constraint" -k 5
   ```
   You want hits whose header line reads like a real book + section + page. If a
   book shows section but no page, its Marker output had no page markers (common
   for EPUBs — reflowable, so section-level is the honest granularity there).

## Copying to the VM

Move `md/`, `transcripts/`, and **`chroma/` including `index_meta.json`** to the
Cozy-Library disk, along with `library_mcp.py` and `embedder.py`.

```bash
pip install "mcp[cli]" chromadb==1.5.9 sentence-transformers
hf download nomic-ai/nomic-embed-text-v1.5
```

Pin chromadb to the same version the index was built with — a major-version skew
is one of the things the server checks and refuses to start on. Note the CLI is
`hf`, not `huggingface-cli`, on huggingface_hub 1.x.

`library_mcp.py` embeds queries **on CPU in-process** — one short query per search,
so no GPU and no daemon are needed there. After the model is cached, the whole
retrieval path is offline.

### The startup contract

The server refuses to start rather than serve a library that isn't really there.
It exits non-zero if the Chroma path isn't a directory, the collection doesn't
already exist, the collection is empty, the query embedder's dimension doesn't
match the stored vectors, or `index_meta.json` disagrees about model or backend.

This is deliberately harsh because of how the failure would otherwise present:
`search_library` instructs Junie to say "not in the library" on an empty result and
*not* to answer from memory. That's correct when a topic genuinely isn't covered —
and catastrophic when the disk simply isn't mounted, because she'll faithfully deny
all 116 sources while sounding perfectly well-behaved. An unmounted disk must be an
error, not an empty shelf.

| Env var | Purpose |
|---|---|
| `LIBRARY_CHROMA_PATH` | path to the copied `chroma/` directory (default `/mnt/library/chroma`) |
| `LIBRARY_COLLECTION` | collection name (default `library`) |
| `LIBRARY_EMBED_BACKEND` | `st` (default) or `ollama` — must match how the index was built |
| `LIBRARY_EMBED_DEVICE` | `cpu` on the VM |
| `LIBRARY_ALLOW_EMPTY=1` | start on an empty index anyway (testing only) |
| `LIBRARY_STRICT_META=0` | downgrade metadata mismatches to warnings |

Diagnostics go to **stderr** — stdout is the MCP JSON-RPC transport, so anything
printed there corrupts the framing.

## Tuning knobs (top of `chunk_and_index.py`)

- `TARGET_WORDS` / `OVERLAP_WORDS` — chunk size and overlap.
- `PAGE_MARKER` — the regex that reads Marker's page boundaries; verify once, per
  Marker version.
- nomic's `search_document:` / `search_query:` prefixes live in `embedder.py` —
  don't remove them, recall drops noticeably without them.
- `--backend` / `--device` / `--batch-size` — raise the batch size on GPU for
  throughput; lower it if you hit an OOM.

## Known pins and gotchas

- **Pillow is held at `<11`** by marker-pdf. Anything later needing Pillow ≥11 will
  conflict; that pin is marker's, not yours to relax.
- **Matryoshka truncation.** v1.5 supports truncating 768-d output to 512/256/128
  with graceful degradation. Decide *before* creating the collection — dimension is
  fixed at creation, and a mismatch between index and query is exactly what the
  server's dimension check exists to catch.
- **8 GB VRAM.** Marker (~5 GB) and Whisper large-v3 (~4.7 GB) don't co-reside.
  Sequence them; that's why they're separate scripts.
