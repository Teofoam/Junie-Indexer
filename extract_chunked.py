#!/usr/bin/env python3
"""
extract_chunked.py -- RAM-safe batch PDF conversion with Marker 1.x.

Marker renders every page of a document into RAM before converting
(~13MB/page at 192 DPI), so 1000-page textbooks exhaust a 32GB machine.
This wrapper splits each PDF into fixed-size page chunks, converts each
chunk in a fresh `marker` subprocess (memory returns to the OS between
chunks), then stitches the chunk Markdown back together with global page
numbering -- `{N}----` markers and `_page_N_...` image names are offset to
the true page index, so citations match the original PDF.

Books that already have a stitched .md in --out are skipped, so the run
is resumable.

  python extract_chunked.py --raw raw --out md
  python extract_chunked.py --raw raw --out md --limit 1    # smoke test on smallest book
  python extract_chunked.py --raw raw --out md --force-ocr  # for scanned books
"""
import argparse, os, re, shutil, subprocess, sys
from pathlib import Path

import pypdfium2 as pdfium

PAGE_MARKER = re.compile(r"\{(\d+)\}(-{48})")
IMAGE_TOKEN = re.compile(r"_page_(\d+)_")


def split_pdf(src: Path, chunk_pages: int, workdir: Path) -> tuple[int, list[tuple[int, Path]]]:
    """Split src into chunk PDFs of chunk_pages each. Returns (n_pages, [(offset, path)])."""
    pdf = pdfium.PdfDocument(str(src))
    n = len(pdf)
    chunks = []
    for start in range(0, n, chunk_pages):
        end = min(start + chunk_pages, n)
        piece = pdfium.PdfDocument.new()
        piece.import_pages(pdf, pages=list(range(start, end)))
        out = workdir / f"part_{start:05d}.pdf"
        with open(out, "wb") as f:
            piece.save(f)
        piece.close()
        chunks.append((start, out))
    pdf.close()
    return n, chunks


def convert_chunk(chunk_pdf: Path, out_dir: Path, extra_flags: list[str]) -> Path:
    """Run marker on a folder containing only chunk_pdf. Returns the produced .md path."""
    in_dir = chunk_pdf.parent / f"{chunk_pdf.stem}_in"
    in_dir.mkdir(exist_ok=True)
    staged = in_dir / chunk_pdf.name
    shutil.move(str(chunk_pdf), staged)
    cmd = ["marker", str(in_dir),
           "--output_dir", str(out_dir),
           "--output_format", "markdown",
           "--paginate_output",
           "--workers", "1",
           "--disable_tqdm"] + extra_flags
    subprocess.run(cmd, check=True)
    md = out_dir / chunk_pdf.stem / f"{chunk_pdf.stem}.md"
    if not md.is_file():
        raise FileNotFoundError(f"marker produced no markdown at {md}")
    return md


def offset_page_refs(text: str, offset: int) -> str:
    text = PAGE_MARKER.sub(lambda m: "{" + str(int(m.group(1)) + offset) + "}" + m.group(2), text)
    text = IMAGE_TOKEN.sub(lambda m: f"_page_{int(m.group(1)) + offset}_", text)
    return text


def process_book(src: Path, out_root: Path, chunk_pages: int, scratch: Path,
                 extra_flags: list[str]) -> None:
    book_out = out_root / src.stem
    final_md = book_out / f"{src.stem}.md"
    if final_md.is_file():
        print(f"[skip] {src.name} (already converted)")
        return

    workdir = scratch / "current_book"
    shutil.rmtree(workdir, ignore_errors=True)
    workdir.mkdir(parents=True)

    n_pages, chunks = split_pdf(src, chunk_pages, workdir)
    print(f"[book] {src.name}: {n_pages} pages -> {len(chunks)} chunk(s)")

    stitched: list[str] = []
    images: list[tuple[Path, str]] = []  # (chunk image file, offset name)
    for i, (offset, chunk_pdf) in enumerate(chunks, 1):
        print(f"  [chunk {i}/{len(chunks)}] pages {offset}-{min(offset + chunk_pages, n_pages) - 1}")
        chunk_out = workdir / f"out_{offset:05d}"
        md = convert_chunk(chunk_pdf, chunk_out, extra_flags)
        stitched.append(offset_page_refs(md.read_text(encoding="utf-8"), offset))
        for img in md.parent.iterdir():
            if img.suffix.lower() in (".jpeg", ".jpg", ".png", ".webp", ".gif"):
                images.append((img, IMAGE_TOKEN.sub(
                    lambda m: f"_page_{int(m.group(1)) + offset}_", img.name)))

    book_out.mkdir(parents=True, exist_ok=True)
    for img, new_name in images:
        shutil.copy2(img, book_out / new_name)
    final_md.write_text("".join(stitched), encoding="utf-8")
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
    ap.add_argument("--force-ocr", action="store_true")
    ap.add_argument("--use-llm", action="store_true")
    args = ap.parse_args()

    if shutil.which("marker") is None:
        sys.exit("marker CLI not found. In JunieAirport:  pip install 'marker-pdf<2'")

    extra = []
    if args.force_ocr:
        extra.append("--force_ocr")
    if args.use_llm:
        extra.append("--use_llm")

    scratch = Path(os.environ.get("JUNIELIB_SCRATCH", args.out / "_chunk_scratch"))
    books = sorted(args.raw.glob("*.pdf"), key=lambda p: p.stat().st_size)
    if args.limit:
        books = books[: args.limit]
    print(f"{len(books)} book(s) to process (smallest first)")

    failed = []
    for src in books:
        try:
            process_book(src, args.out, args.chunk_pages, scratch, extra)
        except Exception as e:
            print(f"  [FAIL] {src.name}: {e}", file=sys.stderr)
            failed.append(src.name)

    shutil.rmtree(scratch, ignore_errors=True)
    if failed:
        print(f"\n{len(failed)} book(s) failed:", *failed, sep="\n  ", file=sys.stderr)
        sys.exit(1)
    print(f"\nDone -> {args.out}/  (open one file and confirm the page markers look like "
          f"'{{N}}----...' before you index)")


if __name__ == "__main__":
    main()
