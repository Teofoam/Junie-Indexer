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

  python extract_chunked.py --raw raw --out md
  python extract_chunked.py --raw raw --out md --limit 1 --chunk-pages 10  # smoke test
  python extract_chunked.py --raw raw --out md --force-ocr    # for scanned books
  python extract_chunked.py --raw raw --out md --keep-scratch # keep chunk output to inspect

Requires marker-pdf 1.x. 2.x routes OCR through a VLM inference server
(vLLM in Docker), which this wrapper does not drive -- see README.
"""
import argparse, os, re, shutil, subprocess, sys
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


def slug(s: str) -> str:
    return re.sub(r"-+", "-", re.sub(r"[^A-Za-z0-9]+", "-", s)).strip("-")[:80] or "book"


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
    cmd += extra_flags

    subprocess.run(cmd, check=True)
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

    # Stable per-book workdir, NOT wiped on entry: that is what makes a
    # re-run after a crash resume instead of starting the book over.
    workdir = scratch / slug(src.stem)
    parts_root, chunk_out_root = workdir / "parts", workdir / "out"
    workdir.mkdir(parents=True, exist_ok=True)

    n_pages, chunks = split_pdf(src, chunk_pages, parts_root)
    print(f"[book] {src.name}: {n_pages} pages -> {len(chunks)} chunk(s)")

    stitched: list[str] = []
    images: list[tuple[Path, str]] = []
    for i, (offset, part_dir) in enumerate(chunks, 1):
        last = min(offset + chunk_pages, n_pages) - 1
        print(f"  [chunk {i}/{len(chunks)}] pages {offset}-{last}")
        md = convert_chunk(part_dir, chunk_out_root, extra_flags)

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
    print(f"  [done] {final_md}  ({len(images)} images)")


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
    args = ap.parse_args()

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
    print(f"\nDone -> {args.out}/  (open one file and confirm the page markers look like "
          f"'{{N}}----...' before you index)")


if __name__ == "__main__":
    main()
