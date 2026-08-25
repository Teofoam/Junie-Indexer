# Junie's library — host-side ingestion pipeline

Turns your PDFs / EPUBs / lecture videos into a citation-ready, locally-searchable
knowledge base that Junie queries over one scoped MCP tool.

| | |
|---|---|
| **Code lives in** | the `JunieLib` project directory (PyCharm project) |
| **Runs in** | the `JunieAirport` mamba env — Python 3.13, `torch 2.13.0+cu130`, RTX 5060 Laptop (8 GB) |
| **Ships to** | Junie's **Cozy-Library** disk on the Linux Mint VM |

The host does all the GPU work — Marker conversion, Whisper transcription, bulk
embedding — and hands the VM a finished index. The VM never touches the GPU.

## What each stage does

```
raw/ (PDF, EPUB) ──extract_chunked.py──▶ md/    (Marker: structured MD + page markers)
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

pip install "marker-pdf<2" chromadb sentence-transformers tqdm ^
            --extra-index-url https://download.pytorch.org/whl/cu130
python -c "import torch; print(torch.version.cuda)"   # still 13.0?

pip install faster-whisper
```

**Order matters.** `marker-pdf` depends on torch. Install it first and pip resolves
torch from PyPI as a transitive dependency; the later cu130 command then reports
"Requirement already satisfied" and silently does nothing. Torch first, everything
else after.

**Why marker-pdf is pinned below 2.** marker 2.0 stopped running its models
in-process: it spawns an OpenAI-compatible inference server and, on any machine
with an NVIDIA GPU, that means the `vllm/vllm-openai` **Docker image** — no
Docker, no conversion (`SpawnError: docker binary not found`). The non-Docker
fallback (`SURYA_INFERENCE_BACKEND=llamacpp`) needs a separately installed
`llama-server` binary, and vllm's memory tuning starts at 16 GB cards anyway.
marker 1.x (currently resolves to 1.10.2) is the in-process torch pipeline this
project was built on and runs the 5060 directly. The downgrade was verified
safe: `pip check` clean, embedder output bit-identical before/after. Revisit 2.x
only with a `llama-server` CUDA build installed, or on hardware where vllm makes
sense.

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
[Environment]::SetEnvironmentVariable("HF_HOME", "F:\Dev\Caches\huggingface", "User")
```

New shell afterwards. Enabling Windows Developer Mode also restores symlink-based
deduplication in that cache, which meaningfully cuts its size.

### Smoke-test the embedder before anything else

```powershell
python embedder.py st        # expect: STEmbedder: dim=768
```

Downloads ~550 MB on first run. This settles three things you'd otherwise be
assuming: that `trust_remote_code` does or doesn't work on whatever transformers
the marker pin resolved (4.57.x under marker 1.x — verified to produce vectors
bit-identical to transformers 5.14's; `embedder.py` handles either branch), that
the model lands in the relocated `HF_HOME`, and **what dimension you're about to
bake into the collection permanently.** Re-run it after any pip operation that
touches transformers or huggingface-hub — same first four vector components =
the index is still valid.

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
├── extract_chunked.py    # RAM-safe batch extraction (use this for books)
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
   python extract_chunked.py --raw raw --out md
   python extract_chunked.py --raw raw --out md --force-ocr   # for scanned books
   python extract_chunked.py --raw raw --out md --log run.log # overnight runs
   ```
   Use `extract_chunked.py`, not `extract.py`, for anything book-sized. Marker
   renders **every page of a document to a 192-DPI bitmap in RAM before
   converting** (~13 MB/page): a 1000-page textbook commits >13 GB, dies with
   `MemoryError`, and the poisoned worker then fails every later file on
   tiny allocations — and `--workers 1` guards *neither* ceiling. It caps how
   many documents convert at once, but nothing guards system RAM, and it does
   not touch inference batch sizes at all (see the VRAM gotcha below).
   The chunked wrapper splits each PDF into 150-page chunks (`--chunk-pages`),
   converts each in a **fresh marker subprocess** (~6 GB peak, fully returned to
   the OS on exit), and stitches the chunks back with global page numbering —
   `{N}` markers and `_page_N_…` image names are offset to true PDF page
   indices, so citations stay correct. It also caps surya's batch sizes on
   cards under 14 GB, which marker itself does not. Runs smallest-book-first.
   `extract.py` remains for single papers, where whole-file conversion is fine.

   **Use `--log` on any run you won't be watching.** It tees this script's
   output, marker's, and any traceback into one file, and prints per-chunk
   durations with a running ETA — without it a long chunk is indistinguishable
   from a hung one until you go read GPU counters off the live process.
   The file is appended, not truncated, so a resumed run keeps the log of the
   run that died.

   **Resume is per-chunk, not per-book.** Finished chunks get a `.chunk_done`
   sentinel, so a crash on chunk 7 of a 1000-page textbook re-does only chunk 7
   — which matters, because big books are exactly why chunking exists. A chunk
   directory without its sentinel is treated as a truncated write and redone.
   Scratch lives at `./_chunk_scratch/`, deliberately outside `md/`
   (`chunk_and_index.py` rglobs `md/`, and stray fragments would index as extra
   documents with chunk-local page numbers). `JUNIELIB_SCRATCH` names a *parent*
   to put it in — the script only ever deletes its own `_chunk_scratch`
   subdirectory, never the path you hand it. Scratch is kept when a book fails,
   so re-running resumes; `--keep-scratch` keeps it after success too.

   **Then verify the page markers before indexing anything.** Verified on
   marker 1.10.2: output pages look like `{N}` + 48 dashes, which
   `PAGE_MARKER` in `chunk_and_index.py` matches, and image files are named
   `_page_N_<Type>_<i>.jpeg`. Re-verify once after any marker version change —
   if the format drifts, every page citation in the corpus is silently wrong,
   which looks exactly like success.

   Check the marker's **direction** too, which is a separate question from
   whether the regex matches: `parse_segments` assumes `{N}----` *precedes* the
   content of page N. If marker actually emits it as a page *terminator*, every
   citation lands one page late. Open a converted `.md`, find `{5}----`, and
   confirm the text after it is what's on page 5 of the PDF — not page 6.
   A cheap way to see boundaries clearly:

   ```powershell
   python extract_chunked.py --raw raw --out md --limit 1 --chunk-pages 10 --keep-scratch
   ```

3. **Transcribe** (separately — shares the GPU with Marker):
   ```powershell
   python transcribe.py --video video --out transcripts ^
       --prompt "Photogrammetry lecture: epipolar geometry, homography, bundle adjustment, Stachniss."
   ```
   Keep `--compute-type float16`. `int8` fails on Blackwell with
   `CUBLAS_STATUS_NOT_SUPPORTED` — a CTranslate2/`sm_120` incompatibility, not a
   broken install.

   See **Transcription quality** below before doing a full course.

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

## Transcription quality

For a citation index the relevant failure mode is not word error rate in
general — it's **domain jargon**. Ordinary prose transcribes fine while the exact
terms Junie searches on get mangled: "epipolar" → "epi-polar", XPBD →
"X-P-B-D", author surnames anywhere at all. A wrong article costs nothing. A
wrong technical term costs the retrieval, and it does so silently, because the
transcript still reads plausibly.

### `--prompt` is the biggest lever — bigger than model size

`initial_prompt` seeds the decoder with vocabulary, biasing it toward terms you
supply. One line per lecture series does more for citation quality than any
model upgrade:

```powershell
python transcribe.py --video video\photogrammetry --out transcripts ^
    --prompt "Lecture on epipolar geometry, homography, bundle adjustment, RANSAC, Stachniss."
```

*Future work:* a `prompt` column in `manifest.csv` so each series carries its own
vocabulary automatically. `transcribe.py` doesn't read the manifest today.

### Choosing a model

| Model | VRAM | Notes |
|---|---|---|
| `large-v3` | ~4.7 GB | Default. Best accuracy. |
| `large-v3-turbo` | notably less | A distillation of large-v3 with the decoder cut from 32 layers to 4. Several times faster, slightly less accurate — the gap is small on clean English, wider in some other languages. |
| `medium` | ~2 GB | Fallback if VRAM is tight. |

Turbo is a **speed** option, not an accuracy one — reaching for it when you want
better transcripts moves you the wrong way. It earns its place if you have many
hours of video and large-v3 proves to be the pipeline bottleneck. Decide that
from the `Nx realtime` figure the script now prints after each file, not in
advance. Models are downloads rather than packages, so switching is just
`--model`; nothing to reinstall.

### Repetition loops

`condition_on_previous_text` is Whisper's default and is **off** here. It
occasionally sends long lectures into a repetition loop, and a looped segment
poisons its timestamps along with its text — which in this pipeline means a
citation pointing at the wrong minute. Pass `--condition-on-previous` to restore
the upstream default if you'd rather have the coherence.

`--beam-size` defaults to 5, faster-whisper's own default; raising it buys little.

## Copying to the VM

Move `md/`, `transcripts/`, and **`chroma/` including `index_meta.json`** to the
Cozy-Library disk, along with `library_mcp.py` and `embedder.py`.

```bash
pip install "mcp[cli]<2" chromadb==1.5.9 sentence-transformers
hf download nomic-ai/nomic-embed-text-v1.5
```

Pin chromadb to the same version the index was built with — a major-version skew
is one of the things the server checks and refuses to start on. Note the CLI is
`hf`, not `huggingface-cli`, on huggingface_hub 1.x.

**`mcp` must be `<2`.** A bare `pip install "mcp[cli]"` now resolves to 2.x, where
`FastMCP` has been replaced by `MCPServer` under a restructured API, and
`library_mcp.py` dies on its import line before any of the startup checks run.

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
| `LIBRARY_MIN_SCORE` | relevance floor for search hits (default `0.6`) |
| `LIBRARY_ALLOW_EMPTY=1` | start on an empty index anyway (testing only) |
| `LIBRARY_STRICT_META=0` | downgrade metadata mismatches to warnings |

Diagnostics go to **stderr** — stdout is the MCP JSON-RPC transport, so anything
printed there corrupts the framing.

### The relevance floor

The startup contract above stops the server serving an empty shelf. `MIN_SCORE`
stops it serving a *full* one badly.

Vector search is nearest-neighbour, not matching: it returns `k` rows however far
away they are, so an unfiltered query can never come back empty and the "say it
isn't in the library" instruction could never fire. Measured on the 23-book index,
"baroque harpsichord tuning temperament" returned a passage on Fourier transforms
at **0.575**, while genuine hits score **0.72–0.86**. Junie would have cited it.

`0.6` sits in that gap. Hits below it are withheld, and three outcomes that used to
look identical are now distinct:

| Result | What Junie is told |
|---|---|
| hits above the floor | normal output, unchanged |
| some below | strong ones returned, plus how many were withheld — a thin-coverage signal |
| none above | names the closest score and that it was withheld; say the library doesn't cover it |
| filter matched nothing | separate message pointing at `list_sources()` |

That last row matters on its own: a mistyped `source_id` used to produce "not in
the library", which would have Junie deny a book that is sitting in the index.

Re-tune with `LIBRARY_MIN_SCORE` if your corpus scores differently — the number is
calibrated against *this* index, not a universal constant.

## Serving the library on the host

The VM is the intended home, but the server runs against the freshly built
`chroma/` directory in place, which is useful for testing retrieval before copying
anything. `.mcp.json` in the repo root wires it up over stdio.

The committed config is deliberately path-free so it starts on any machine:

| Key | Value |
|---|---|
| `command` | `${JUNIELIB_PYTHON:-python}` |
| `args` | `library_mcp.py` (relative to the project root) |
| `LIBRARY_CHROMA_PATH` | `${LIBRARY_CHROMA_PATH:-chroma}` |

Set `JUNIELIB_PYTHON` to an interpreter that has `mcp<2`, `chromadb` and
`sentence-transformers` installed. A bare `python` from `PATH` is the fallback, but
on Windows that often resolves to the Microsoft Store stub, which will fail.

Machine-specific values belong in `.claude/settings.local.json`, which is
gitignored:

```json
{
  "env": {
    "JUNIELIB_PYTHON": "C:\\path\\to\\env\\python.exe",
    "LIBRARY_CHROMA_PATH": "C:\\path\\to\\JunieLib\\chroma"
  }
}
```

`LIBRARY_EMBED_DEVICE` is pinned to `cpu` in the committed config. The embedder
resolves `auto` to CUDA when a card is present, and on the machine this was built
on that bugchecks the box within minutes — see `CRASH_INVESTIGATION.md`. Queries
are one short embedding each, so CPU costs nothing noticeable here regardless.

## Tuning knobs (top of `chunk_and_index.py`)

- `TARGET_WORDS` / `OVERLAP_WORDS` — chunk size and overlap.
- `PAGE_MARKER` — the regex that reads Marker's page boundaries; verify once, per
  Marker version.
- nomic's `search_document:` / `search_query:` prefixes live in `embedder.py` —
  don't remove them, recall drops noticeably without them.
- `--backend` / `--device` / `--batch-size` — raise the batch size on GPU for
  throughput; lower it if you hit an OOM.

## Known pins and gotchas

- **marker-pdf is pinned `<2`** (see *Building the env* for why). The pin drags
  the whole neighborhood: transformers 4.57.x, huggingface-hub 0.36.x,
  `openai` 1.x, `anthropic` 0.46, `google-genai` 1.x, pypdfium2 4.30. The
  embedder is verified unaffected, but any *other* script sharing `JunieAirport`
  that uses those SDKs directly gets the older majors.
- **`mcp` is pinned `<2`.** `library_mcp.py` imports `FastMCP` from
  `mcp.server.fastmcp`; in 2.x that is gone, replaced by `MCPServer` under a
  restructured API, so a bare `pip install "mcp[cli]"` produces a server that
  fails on its import line before a single startup check runs. Porting to 2.x is
  a deliberate rewrite, not a version bump.
- **System RAM, not just VRAM, is a ceiling.** Marker's up-front page rendering
  is why `extract_chunked.py` exists; peak commit scales with `--chunk-pages`
  (~13 MB/page at 192 DPI plus the models), not with book size. **Measured: 150
  pages ≈ 6 GB working set** on a 1137-page textbook — budget from that, not
  from the page arithmetic alone.
- **Watch the commit charge, not just "available" RAM.** With a 4 GB pagefile
  this box runs ~32/35 GB committed during a chunk — 91%, while Task Manager
  still reports 12 GB "available". Commit exhaustion is what produced the
  original `MemoryError` (and took PyCharm's JVM down with it), so if you run
  anything heavy alongside a conversion, raise the pagefile rather than trusting
  the available figure.
- **`Failed to start MPS server` warning on Windows is noise** — NVIDIA MPS is
  Linux-only; marker tries it and falls back cleanly.
- **Pillow is held at `<11`** by marker-pdf. Anything later needing Pillow ≥11 will
  conflict; that pin is marker's, not yours to relax.
- **Matryoshka truncation.** v1.5 supports truncating 768-d output to 512/256/128
  with graceful degradation. Decide *before* creating the collection — dimension is
  fixed at creation, and a mismatch between index and query is exactly what the
  server's dimension check exists to catch.
- **8 GB VRAM.** Marker (~5 GB) and Whisper large-v3 (~4.7 GB) don't co-reside.
  Sequence them; that's why they're separate scripts.
- **Marker skips its own VRAM caps on cards under 14 GB.**
  `marker/utils/batch.py` computes `workers = max(1, vram // 7)` and returns
  *no* batch-size overrides when `workers == 1`, so its conservative numbers
  apply only when the card can hold two 7 GB workers. This 8151 MiB card reads
  as `int(8151/1024)` = **7 GB**, lands on the `workers == 1` branch, and gets
  surya's raw CUDA defaults instead: recognition **256**, ocr_error 64,
  detection 36, layout/table_rec 32. `--workers 1` does not help —
  `convert.py` fixes the batch sizes *before* applying the workers override.
  `extract_chunked.py` therefore sets `RECOGNITION_BATCH_SIZE=64`,
  `DETECTOR_BATCH_SIZE=8`, `LAYOUT_BATCH_SIZE=12`, `TABLE_REC_BATCH_SIZE=12`,
  `OCR_ERROR_BATCH_SIZE=12` (marker's own safe values) below 14 GB, via
  `setdefault` so an explicit export still wins, and logs which branch it took.
  **What this does not do is make conversion faster** — see below.
- **The caps did not change throughput, and the cause of slow conversion is
  unsettled.** Measured, one 150-page chunk of dense textbook: **57:30 with the
  caps** against a **59:09 mean without them** — inside the normal per-chunk
  spread (37:25–77:57), so no effect either way. The VRAM spill is real and
  measured (7.46 GB dedicated plus **1.82 GB paged into shared host memory**,
  GPU reading 99% utilization at 41 W while its enforced limit is **115 W** —
  `nvidia-smi -q -d POWER` shows Current 115 W, Default 50 W, so the card is
  drawing a third of what it may and is not power-throttled at all — clocks
  meanwhile at 2790/3090 MHz),
  but it does not explain the speed, and neither do model weights: all five
  models total **~3.3 GB on disk** against 7.46 GB resident, most of the rest
  being PyTorch's caching allocator holding blocks it never returns.
  What actually predicts runtime is **content density**: dense textbook pages
  run **0.042 pp/s** while light pages (*Ray Tracing in One Weekend*) run
  **0.93 pp/s** — a 20× spread at identical settings. A plausible unverified
  explanation is that surya 0.17's recognition model is an autoregressive
  decoder and decode is memory-bandwidth-bound, which produces exactly this
  high-utilization/low-wattage signature. If that is right, cutting
  recognition batch from 256 to 64 may *cost* throughput, since large batches
  amortize bandwidth. **Untested.** The clean experiment is one fixed
  150-page chunk run with the caps on and off, one variable, same book.
- **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is a no-op on Windows.**
  The obvious thing to reach for when VRAM spills, and it does nothing here:
  `c10/cuda/CUDAAllocatorConfig.h` hard-returns `false` unless
  `PYTORCH_C10_DRIVER_API_SUPPORTED` is defined, which gates the Linux-only CUDA
  driver-API path and is absent from this build's headers. You get one
  `TORCH_WARN_ONCE` and no behavior change. Cap the batch sizes instead.
- **PDF page ≠ printed page. Handled, but only for 15 of 23 books.** Marker
  numbers PDF pages from 0, while a book's printed numbering starts after its
  front matter. `manifest.csv` carries a per-book `page_offset`;
  `process_file` subtracts it so `page_start`/`page_end` — what
  `library_mcp.py` renders as `p. N` — is the **printed** page, and keeps the
  physical index in `pdf_page_start`/`pdf_page_end` so you can still open the
  file at the right place.

  **The offset is NOT constant for every book**, contrary to what this section
  used to claim. Verified counterexamples: `patterson-hennessy` interleaves
  online-only `.e` pages and runs at offset 100 by p397 but 136 by p511;
  `hwu-mppp` drifts 22 → 18 → 17; `hennessy-patterson`'s appendices are
  numbered `A-38`, `C-66`, `F-45`, which no integer can express. Those books
  and five others sit at `0`, meaning their citations are PDF pages — wrong,
  but wrong in a single obvious direction rather than plausibly-wrong.

  Offsets were derived by sampling ten body pages per PDF and reading the folio
  out of the running head/foot, accepting a value only when ≥6 agreed. **Do not
  use embedded PDF page labels for this** — `get_page_label()` produced
  confident answers that disagreed with the actual printed folios on 4 of the
  8 books spot-checked. Re-derive with the same majority-vote method if you add
  books; a wrong offset is worse than none, because it looks right.
- **`26. Numerical Mathematics and Computing` is a scan** — zero characters
  extract across ten sample pages, so it has no text layer. It needs its own
  `--force-ocr` pass, and no `page_offset` is recoverable from it without OCR.
