#!/usr/bin/env python3
"""
extract.py -- Batch-convert PDFs/EPUBs to paginated Markdown with Marker.

Wraps Marker's CLI so every source is converted identically: structured
Markdown + page markers (--paginate_output), which the chunker turns into
page citations. Runs on the RTX 5060 in the JunieAirport env.

  python extract.py --raw raw --out md
  python extract.py --raw raw --out md --force-ocr    # for scanned books
"""
import argparse, shutil, subprocess, sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw",     type=Path, default=Path("raw"), help="folder of PDFs/EPUBs")
    ap.add_argument("--out",     type=Path, default=Path("md"))
    ap.add_argument("--workers", type=int,  default=1,
                    help="1 keeps Marker's ~5GB peak inside the 5060's 8GB VRAM")
    ap.add_argument("--force-ocr", action="store_true",
                    help="re-OCR every page (needed for scanned books; also improves math)")
    ap.add_argument("--use-llm", action="store_true",
                    help="LLM accuracy pass -- NOTE: calls a cloud model, so off by default to stay local")
    args = ap.parse_args()

    if shutil.which("marker") is None:
        sys.exit("marker CLI not found. In JunieAirport:  pip install marker-pdf")

    args.out.mkdir(parents=True, exist_ok=True)
    cmd = ["marker", str(args.raw),
           "--output_dir", str(args.out),
           "--output_format", "markdown",
           "--paginate_output",
           "--workers", str(args.workers)]
    if args.force_ocr:
        cmd.append("--force_ocr")
    if args.use_llm:
        cmd.append("--use_llm")

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"\nDone -> {args.out}/  (open one file and confirm the page markers look like "
          f"'{{N}}----...' before you index)")


if __name__ == "__main__":
    main()
