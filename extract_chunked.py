#!/usr/bin/env python3
"""
extract_chunked.py -- RAM-safe, resumable batch PDF conversion with Marker 1.x.

Marker renders every page of a document into RAM before converting
(~13MB/page at 192 DPI), so 1000-page textbooks exhaust a 32GB machine.
This wrapper splits each PDF into fixed-size page chunks, converts each
chunk in a fresh `marker` subprocess (memory returns to the OS between
chunks), then stitches the chunk Markdown back together with global page
numbering -- `{N}----` markers and `_page_N_...` image names are offset to
the true page index, so citations match the original PDF.

RESUMABLE AT CHUNK GRANULARITY. Each converted chunk gets a .chunk_done
sentinel; a re-run skips finished chunks and re-does only the interrupted
one. Per-book scratch is kept until that book is stitched, then removed.
This matters because the whole reason for chunking is that big books are
where conversion falls over -- losing six converted chunks to a crash on
the seventh would defeat the point.

Scratch is keyed by book name, source file size, AND --chunk-pages, so
changing the chunk size (or replacing a PDF) starts that book's chunks
fresh instead of reusing incompatible parts; finished books are never
touched either way.

  python extract_chunked.py --raw raw --out md
  python extract_chunked.py --raw raw --out md --limit 1 --chunk-pages 10  # smoke test
  python extract_chunked.py --raw raw --out md --force-ocr    # for scanned books
  python extract_chunked.py --raw raw --out md --keep-scratch # keep chunk output to inspect
  python extract_chunked.py --raw raw --out md --log run.log  # tee everything to a file

VRAM: on cards under 14GB we cap surya's inference batch sizes ourselves,
because marker will not -- see BATCH_CAPS. Without that, an 8GB card runs
the full CUDA defaults and overcommits, and the driver pages the overflow
into host RAM over PCIe rather than raising OOM.

This caps memory, NOT time. Measured on a 150-page chunk of dense text:
57:30 with the caps against a 59:09 mean without, i.e. no effect, well
inside the 37:25-77:57 per-chunk spread. Runtime tracks content density
instead (0.042 pages/sec dense vs 0.93 light). Do not treat these caps as
a speed fix -- see the throughput gotcha in the README.

Requires marker-pdf 1.x. 2.x routes OCR through a VLM inference server
(vLLM in Docker), which this wrapper does not drive -- see README.
"""
import argparse, hashlib, os, re, shutil, subprocess, sys, time
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path

import pypdfium2 as pdfium

# Line-anchored: marker emits the page marker on its own line, and anchoring
# stops a coincidental "{12}-----" inside body text from being renumbered.
# Dash count is marker's page_separator ('-' * 48 by default) -- matched loosely
# so a changed separator width doesn't silently stop the offsetting.
PAGE_MARKER = re.compile(r"^\{(\d+)\}(-{10,})\s*$", re.M)
IMAGE_TOKEN = re.compile(r"_page_(\d+)_")
DONE = ".chunk_done"          # sentinel: this chunk converted completely
IMAGE_EXTS = (".jpeg", ".jpg", ".png", ".webp", ".gif")

# ---- VRAM caps -------------------------------------------------------------
# marker/utils/batch.py picks batch sizes like this:
#
#     workers = max(1, vram // 7)        # vram from nvidia-smi, int GB
#     if workers == 1:
#         return {}, workers            # <-- no caps at all
#
# so its conservative numbers are applied ONLY when the card can hold two or
# more 7GB workers, i.e. >=14GB. Below that you get surya's raw CUDA defaults
# (recognition 256, ocr_error 64, detection 36, layout/table_rec 32), which do
# not fit in 8GB. That is backwards, and marker's `--workers 1` does not help:
# convert.py computes the batch sizes BEFORE applying the workers override.
#
# These are marker's own safe values, sized for a ~7GB budget, which is what a
# single worker on an 8GB card actually has. Applied as env defaults so an
# explicit setting in the caller's shell still wins.
BATCH_CAPS = {
    "RECOGNITION_BATCH_SIZE": "64",
    "DETECTOR_BATCH_SIZE": "8",
    "LAYOUT_BATCH_SIZE": "12",
    "TABLE_REC_BATCH_SIZE": "12",
    "OCR_ERROR_BATCH_SIZE": "12",
}
CAP_BELOW_VRAM_GB = 14        # the window where marker skips its own caps
EQUATION_BATCH_SIZE = "16"    # marker-side; a CLI flag, not a surya env var


def slug(s: str) -> str:
    return re.sub(r"-+", "-", re.sub(r"[^A-Za-z0-9]+", "-", s)).strip("-")[:80] or "book"


def book_workdir(scratch: Path, src: Path, chunk_pages: int) -> Path:
    """Per-book scratch dir, keyed so stale resume state can never be misapplied.

    - name hash: keeps distinct books apart even when slug() collapses them
      (two all-CJK titles would otherwise both become "book" and cross-stitch
      each other's sentineled chunks);
    - file size: a replaced source PDF invalidates the old parts;
    - chunk size: parts and sentinels from a different --chunk-pages would
      stitch with silently missing pages (a 10-page part_00000 reused by a
      150-page run loses pages 10-149).
    """
    name_key = hashlib.md5(src.name.encode("utf-8")).hexdigest()[:8]
    return scratch / f"{slug(src.stem)}-{name_key}-{src.stat().st_size}-c{chunk_pages}"


# ---- marker capability probing --------------------------------------------
# Flags differ between marker 1.x and 2.x, and asserting one that doesn't exist
# makes click exit non-zero on EVERY chunk -- a long run that fails identically
# hundreds of times. Probe once instead of hardcoding.

@lru_cache(maxsize=1)
def _marker_help() -> str:
    try:
        r = subprocess.run(["marker", "--help"], capture_output=True,
                           text=True, timeout=180)
        return (r.stdout or "") + (r.stderr or "")
    except Exception:
        return ""


def marker_supports(flag: str) -> bool:
    return flag in _marker_help()


def check_marker_version() -> None:
    """Refuse to run against marker 2.x rather than failing per-chunk later."""
    if shutil.which("marker") is None:
        sys.exit("marker CLI not found. In JunieAirport:  pip install 'marker-pdf<2'")
    try:
        from importlib.metadata import version
        v = version("marker-pdf")
    except Exception:
        return
    head = v.split(".")[0]
    if head.isdigit() and int(head) >= 2:
        sys.exit(
            f"marker-pdf {v} detected; this wrapper targets 1.x.\n"
            "2.x runs OCR/layout through a VLM served by vLLM (Docker) or\n"
            "llama.cpp, which this script does not start.\n"
            "  pip install 'marker-pdf<2'")


@lru_cache(maxsize=1)
def gpu_vram_gb() -> int:
    """Total VRAM in whole GB, or 0 if there is no CUDA card to ask about.

    Deliberately shells out to nvidia-smi rather than importing torch: this
    runs in the parent process, and initialising CUDA here just to read a
    number would hold a context for the whole run, against a budget that is
    already the thing we are trying not to overspend.

    int() truncation matches marker's own reading of the same value, so our
    threshold and its `vram // 7` agree about which side of the line a card
    falls on (an 8151MiB card reads as 7GB to both).
    """
    if shutil.which("nvidia-smi") is None:
        return 0
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=60, check=True)
        return int(int(r.stdout.strip().splitlines()[0]) / 1024)
    except Exception:
        return 0


def marker_env() -> dict:
    """Environment for the marker subprocess, with batch caps applied.

    setdefault, not assignment: if you exported RECOGNITION_BATCH_SIZE
    yourself you meant it, and this should not quietly overrule you.
    """
    env = os.environ.copy()
    vram = gpu_vram_gb()
    if 0 < vram < CAP_BELOW_VRAM_GB:
        for k, v in BATCH_CAPS.items():
            env.setdefault(k, v)
    return env


# ---- logging ---------------------------------------------------------------

def stamp(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}")


class Tee:
    """Mirror a stream to a log file.

    Installed over sys.stdout/sys.stderr so the wrapper's own output, marker's
    output, and tracebacks all land in one file in the order they happened --
    the point being that a run that misbehaves overnight leaves evidence
    behind instead of requiring live process forensics.
    """

    def __init__(self, stream, fh):
        self.stream, self.fh = stream, fh

    def write(self, s):
        self.stream.write(s)
        self.stream.flush()
        self.fh.write(s)
        self.fh.flush()
        return len(s)

    def flush(self):
        self.stream.flush()
        self.fh.flush()

    def isatty(self):
        return self.stream.isatty()


def run_streaming(cmd: list[str], env: dict) -> None:
    """Run cmd, relaying output line by line, and raise on non-zero exit.

    Streamed rather than captured: a chunk takes the better part of an hour,
    and subprocess.run(capture_output=True) would withhold every line of it
    until the end -- exactly when it stops being useful. Writing to
    sys.stdout (not the file directly) keeps Tee as the single place that
    decides where output goes.
    """
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace",
                            bufsize=1, env=env)
    for line in proc.stdout:
        sys.stdout.write(line)
    rc = proc.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def fmt_dur(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


# ---- splitting -------------------------------------------------------------

def split_pdf(src: Path, chunk_pages: int, parts_root: Path) -> tuple[int, list[tuple[int, Path]]]:
    """Split src into per-chunk folders, each holding one PDF.

    Each chunk lives in its own directory because marker takes an input FOLDER;
    giving it a dedicated one avoids moving files around and keeps the split
    idempotent, so a resumed run reuses the parts already on disk.

    Returns (n_pages, [(page_offset, part_dir)]).
    """
    pdf = pdfium.PdfDocument(str(src))
    n = len(pdf)
    chunks: list[tuple[int, Path]] = []
    for start in range(0, n, chunk_pages):
        end = min(start + chunk_pages, n)
        part_dir = parts_root / f"part_{start:05d}"
        part_dir.mkdir(parents=True, exist_ok=True)
        out = part_dir / f"{part_dir.name}.pdf"
        if not out.is_file():
            piece = pdfium.PdfDocument.new()
            piece.import_pages(pdf, pages=list(range(start, end)))
            with open(out, "wb") as f:
                piece.save(f)
            piece.close()
        chunks.append((start, part_dir))
    pdf.close()
    return n, chunks


def convert_chunk(part_dir: Path, out_root: Path, extra_flags: list[str]) -> Path:
    """Convert one chunk folder, skipping it if a previous run finished it."""
    stem = part_dir.name
    chunk_out = out_root / stem
    md = chunk_out / f"{stem}.md"
    sentinel = chunk_out / DONE

    if sentinel.is_file() and md.is_file():
        return md

    # A directory without the sentinel is a partial write from an interrupted
    # run -- the .md may be truncated, so discard rather than trust it.
    shutil.rmtree(chunk_out, ignore_errors=True)

    cmd = ["marker", str(part_dir),
           "--output_dir", str(out_root),
           "--output_format", "markdown",
           "--paginate_output",
           "--workers", "1"]
    if marker_supports("--disable_tqdm"):
        cmd.append("--disable_tqdm")
    # Equation recognition has no surya env var -- it is a marker processor
    # config -- so it has to come in as a flag, and the flag is only present
    # on some 1.x builds. Probe rather than assume; asserting a flag that does
    # not exist makes click exit non-zero on every chunk.
    if gpu_vram_gb() and gpu_vram_gb() < CAP_BELOW_VRAM_GB:
        for flag in ("--equation_batch_size",
                     "--EquationProcessor_equation_batch_size"):
            if marker_supports(flag):
                cmd += [flag, EQUATION_BATCH_SIZE]
                break
    cmd += extra_flags

    run_streaming(cmd, marker_env())
    if not md.is_file():
        raise FileNotFoundError(f"marker produced no markdown at {md}")
    sentinel.write_text("ok\n", encoding="utf-8")
    return md


def offset_page_refs(text: str, offset: int) -> str:
    text = PAGE_MARKER.sub(lambda m: "{" + str(int(m.group(1)) + offset) + "}" + m.group(2), text)
    return IMAGE_TOKEN.sub(lambda m: f"_page_{int(m.group(1)) + offset}_", text)


def process_book(src: Path, out_root: Path, chunk_pages: int, scratch: Path,
                 extra_flags: list[str], keep_scratch: bool = False) -> None:
    book_out = out_root / src.stem
    final_md = book_out / f"{src.stem}.md"
    if final_md.is_file():
        print(f"[skip] {src.name} (already converted)")
        return

    book_t0 = time.monotonic()

    # Stable per-book workdir, NOT wiped on entry: that is what makes a
    # re-run after a crash resume instead of starting the book over.
    workdir = book_workdir(scratch, src, chunk_pages)
    parts_root, chunk_out_root = workdir / "parts", workdir / "out"
    workdir.mkdir(parents=True, exist_ok=True)

    n_pages, chunks = split_pdf(src, chunk_pages, parts_root)
    stamp(f"[book] {src.name}: {n_pages} pages -> {len(chunks)} chunk(s)")

    stitched: list[str] = []
    images: list[tuple[Path, str]] = []
    # Timings feed the ETA below. Only chunks actually converted in THIS run
    # count: a resumed run skips finished chunks in milliseconds, and letting
    # those into the mean would predict the rest of the book finishing almost
    # immediately.
    durations: list[float] = []
    for i, (offset, part_dir) in enumerate(chunks, 1):
        last = min(offset + chunk_pages, n_pages) - 1
        stamp(f"  [chunk {i}/{len(chunks)}] pages {offset}-{last}")
        t0 = time.monotonic()
        md = convert_chunk(part_dir, chunk_out_root, extra_flags)
        elapsed = time.monotonic() - t0

        if elapsed < 1.0:
            stamp(f"  [chunk {i}/{len(chunks)}] already done, skipped")
        else:
            durations.append(elapsed)
            mean = sum(durations) / len(durations)
            remaining = len(chunks) - i
            msg = f"  [chunk {i}/{len(chunks)}] took {fmt_dur(elapsed)}"
            if remaining:
                eta = datetime.now() + timedelta(seconds=mean * remaining)
                msg += (f" | mean {fmt_dur(mean)} | {remaining} left"
                        f" | book ETA ~{eta:%H:%M} (+{fmt_dur(mean * remaining)})")
            stamp(msg)

        # rstrip + "\n\n" guards against the next chunk's {N}---- marker landing
        # mid-line: PAGE_MARKER is line-anchored on both sides of the pipeline,
        # and a welded marker would silently inherit the previous page's number.
        body = offset_page_refs(md.read_text(encoding="utf-8"), offset)
        stitched.append(body.rstrip("\n") + "\n\n")

        for img in md.parent.iterdir():
            if img.suffix.lower() in IMAGE_EXTS:
                images.append((img, IMAGE_TOKEN.sub(
                    lambda m: f"_page_{int(m.group(1)) + offset}_", img.name)))

    book_out.mkdir(parents=True, exist_ok=True)
    for img, new_name in images:
        shutil.copy2(img, book_out / new_name)
    final_md.write_text("".join(stitched), encoding="utf-8")

    if not keep_scratch:
        shutil.rmtree(workdir, ignore_errors=True)
    stamp(f"  [done] {final_md}  ({len(images)} images,"
          f" {fmt_dur(time.monotonic() - book_t0)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("raw"))
    ap.add_argument("--out", type=Path, default=Path("md"))
    ap.add_argument("--chunk-pages", type=int, default=150,
                    help="pages per marker invocation; 150 peaks at ~2-3GB RAM")
    ap.add_argument("--limit", type=int, default=0,
                    help="only process the N smallest books (0 = all)")
    ap.add_argument("--keep-scratch", action="store_true",
                    help="keep per-chunk output for inspection instead of deleting it")
    ap.add_argument("--force-ocr", action="store_true")
    ap.add_argument("--use-llm", action="store_true")
    ap.add_argument("--log", type=Path, default=None,
                    help="tee all output (this script's and marker's) to this file")
    args = ap.parse_args()

    # Opened append, not write: a resumed run is the normal case for this
    # script, and truncating would throw away the log of the run that died --
    # which is the one you actually want to read.
    log_fh = None
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        log_fh = args.log.open("a", encoding="utf-8", errors="replace")
        sys.stdout = Tee(sys.stdout, log_fh)
        sys.stderr = Tee(sys.stderr, log_fh)
        print(f"\n{'=' * 70}\n"
              f"run started {datetime.now():%Y-%m-%d %H:%M:%S} :: "
              f"{' '.join(sys.argv)}\n{'=' * 70}")

    try:
        run(args)
    finally:
        if log_fh:
            sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
            log_fh.close()


def run(args) -> None:
    check_marker_version()

    extra = []
    if args.force_ocr:
        extra.append("--force_ocr")
    if args.use_llm:
        extra.append("--use_llm")

    # Scratch must live OUTSIDE --out: chunk_and_index.py rglobs md/ for *.md,
    # and an interrupted run would otherwise leave per-chunk fragments there
    # that index as extra documents with chunk-local (wrong) page numbers.
    #
    # JUNIELIB_SCRATCH names a PARENT to put our own _chunk_scratch/ inside --
    # it is never itself deleted. Treating a user-supplied path as ours to
    # rmtree would mean pointing this at D:\scratch wipes D:\scratch.
    scratch_root = Path(os.environ.get("JUNIELIB_SCRATCH") or args.raw.parent)
    scratch = scratch_root / "_chunk_scratch"

    books = sorted(args.raw.glob("*.pdf"), key=lambda p: p.stat().st_size)
    others = [p.name for p in args.raw.iterdir()
              if p.is_file() and p.suffix.lower() != ".pdf"]
    if others:
        print(f"note: {len(others)} non-PDF file(s) in {args.raw}/ skipped -- "
              f"page-splitting needs PDFs; convert these with extract.py:",
              *others, sep="\n  ")
    if args.limit:
        books = books[: args.limit]
    print(f"{len(books)} book(s) to process (smallest first)")
    print(f"scratch: {scratch}")

    # Record the batch sizes in effect. Whether the caps applied is the first
    # thing you want to know when a chunk runs long, and inferring it after
    # the fact means reading GPU counters on a live process.
    vram = gpu_vram_gb()
    if not vram:
        print("gpu: none detected (CPU path); leaving surya's CPU defaults alone")
    elif vram < CAP_BELOW_VRAM_GB:
        caps = ", ".join(f"{k.split('_BATCH')[0].lower()}={marker_env()[k]}"
                         for k in BATCH_CAPS)
        print(f"gpu: {vram}GB (<{CAP_BELOW_VRAM_GB}GB) -- capping batch sizes "
              f"marker would leave at CUDA defaults: {caps}")
    else:
        print(f"gpu: {vram}GB -- marker applies its own caps; not overriding")

    run_t0 = time.monotonic()
    failed = []
    for src in books:
        try:
            process_book(src, args.out, args.chunk_pages, scratch, extra, args.keep_scratch)
        except Exception as e:
            print(f"  [FAIL] {src.name}: {e}", file=sys.stderr)
            failed.append(src.name)

    if failed:
        print(f"\n{len(failed)} book(s) failed:", *failed, sep="\n  ", file=sys.stderr)
        print(f"Scratch kept at {scratch} -- re-run to resume from the last "
              f"completed chunk.", file=sys.stderr)
        sys.exit(1)

    if not args.keep_scratch:
        shutil.rmtree(scratch, ignore_errors=True)   # only ever our own subdir
    stamp(f"all books done in {fmt_dur(time.monotonic() - run_t0)}")
    print(f"\nDone -> {args.out}/  (open one file and confirm the page markers look like "
          f"'{{N}}----...' before you index)")


if __name__ == "__main__":
    main()
