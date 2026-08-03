# Junie's library — host-side ingestion pipeline

Runs on your **Windows host** in the `lmme_dl` env (the one with the RTX 5060 +
`torch 2.8.0+cu129`). It turns your PDFs / EPUBs / lecture videos into a
citation-ready, locally-searchable knowledge base that Junie will later query
over an MCP tool. The VM stays out of this entirely — it never touches the GPU.

## What each stage does

```
raw/ (PDF, EPUB) ──extract.py──▶ md/           (Marker: structured MD + page markers)
video/ (mp4…)   ──transcribe.py─▶ transcripts/  (Whisper: MD + [mm:ss] markers)
                                       │
                       chunk_and_index.py  (section-aware chunks + {title,section,page/time})
                                       │
                                       ▼
                                    chroma/      (local vector DB, nomic-embed-text)
```

Citations come out as **"Hartley & Zisserman — 11.1 Epipolar geometry (p. 239–241)"**
for books and **"Stachniss, Photogrammetry I (12:30–14:05)"** for lectures.

## Embedding backend — pick one, use it everywhere

Both routes run the same model (nomic-embed-text v1.5) with the same
`search_document:` / `search_query:` prefixes. They differ only in how it's hosted.

### Route A — sentence-transformers (default, recommended)

Loads the model **in-process** inside `lmme_dl`. No daemon, no background
service, and the model is released the moment the script exits. You already have
`torch 2.8.0+cu129`, so there is nothing else to install.

```powershell
pip install marker-pdf chromadb tqdm faster-whisper sentence-transformers
```

```powershell
python chunk_and_index.py --backend st              # GPU if free, else CPU
python chunk_and_index.py --backend st --device cpu # leave the GPU untouched
```

Footprint is small: nomic-embed-text v1.5 is ~137M parameters, about **0.3 GB**
in fp16 — next to Marker's ~5 GB and Whisper large-v3's ~4.7 GB it is noise. The
reason to prefer this route is not memory, it's that nothing stays resident
afterwards.

Add `--device cpu` if you want to embed while something else holds the GPU. It's
slower for the full corpus but completely avoids contention.

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
The vectors are close but **not identical**, so indexing with one and querying
with the other degrades retrieval with no error message. Whichever you pick on
the host, use the same on the VM.

## Folder layout

Put everything under one working dir, e.g. `F:\Dev\Shared\VMs\Junie\library-build\`:

```
library-build/
├── raw/            # your PDFs and EPUBs
├── video/          # lecture video/audio files
├── manifest.csv    # maps each file -> title / author / type (edit this!)
├── extract.py  transcribe.py  chunk_and_index.py  query_test.py
├── embedder.py                    # shared embedding backend (st | ollama)
├── md/  transcripts/  chroma/     # created by the scripts
```

## Run order

1. **Fill `manifest.csv`.** One row per source: `source_id, filename, title, author, type`
   (`type` ∈ `book | paper | lecture`). `filename` must match the file's name;
   your numbered list of 116 sources basically *is* this table already. Anything
   not listed still indexes — it just cites by filename instead of a clean title.

2. **Extract** (run with the VM powered off to free its 12 GB):
   ```powershell
   python extract.py --raw raw --out md
   python extract.py --raw raw --out md --force-ocr   # for scanned books
   ```
   Then open one file in `md/` and confirm the page markers look like `{N}----…`.
   If they don't, adjust `PAGE_MARKER` at the top of `chunk_and_index.py`.

3. **Transcribe** (separately — shares the GPU with Marker):
   ```powershell
   python transcribe.py --video video --out transcripts
   ```

4. **Chunk + embed + index:**
   ```powershell
   # check the manifest resolves first -- seconds, vs. an hour of wasted embedding
   python chunk_and_index.py --dry-run

   python chunk_and_index.py --md md --transcripts transcripts ^
                             --manifest manifest.csv --chroma chroma ^
                             --backend st
   ```
   Run this **after** Marker and Whisper have finished — they are separate
   invocations, so the GPU is already free by the time you get here.

5. **Verify citations before moving on:**
   ```powershell
   python query_test.py "what is the epipolar constraint" -k 5
   ```
   You want hits whose header line reads like a real book + section + page. If a
   book shows section but no page, its Marker output had no page markers (common
   for EPUBs — reflowable, so section-level is the honest granularity there).

## Then

Copy `md/`, `transcripts/`, and `chroma/` onto Junie's **Cozy-Library** disk.
Next piece (VM side): a small MCP `search_library` server that lets Junie call
this index as one scoped tool and cite the metadata — no need to reopen `exec`.

## Tuning knobs (top of `chunk_and_index.py`)

- `TARGET_WORDS` / `OVERLAP_WORDS` — chunk size and overlap.
- `PAGE_MARKER` — the regex that reads Marker's page boundaries; verify once.
- nomic's `search_document:` / `search_query:` prefixes are applied inside
  `embedder.py` — don't remove them, recall drops noticeably without them.
- `--backend` / `--device` / `--batch-size` — raise the batch size on GPU for
  throughput; lower it if you hit an OOM.

## Copying to the VM

Move `md/`, `transcripts/`, and `chroma/` to the Cozy-Library disk. On the VM,
`library_mcp.py` embeds queries **on CPU in-process** — one short query per
search, so no GPU and no daemon are needed there at all. Pre-download the model
once while the VM has network:

```bash
pip install sentence-transformers
huggingface-cli download nomic-ai/nomic-embed-text-v1.5
```

After that the whole retrieval path is offline.
