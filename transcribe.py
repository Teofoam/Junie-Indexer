#!/usr/bin/env python3
"""
transcribe.py -- Transcribe lecture videos/audio to Markdown with timestamps.

Uses faster-whisper on the RTX 5060. Emits one .md per source with [mm:ss]
markers so the same chunker attaches TIME-range citations, e.g.
"Stachniss, Photogrammetry I (12:30-14:05)".

  python transcribe.py --video video --out transcripts
  python transcribe.py --video video --out transcripts --model medium   # faster/lighter

Run this SEPARATELY from extract.py -- both want the GPU, and large-v3 + Marker
together will not fit in 8GB VRAM.
"""
import argparse, sys
from pathlib import Path


def fmt(t: float) -> str:
    m, s = divmod(int(t), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, default=Path("video"))
    ap.add_argument("--out",   type=Path, default=Path("transcripts"))
    ap.add_argument("--model", default="large-v3",
                    help="large-v3 (best) ~4.7GB VRAM; medium is a lighter fallback")
    ap.add_argument("--exts", nargs="*",
                    default=[".mp4", ".mkv", ".webm", ".mov", ".m4a", ".mp3", ".wav"])
    args = ap.parse_args()

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        sys.exit("pip install faster-whisper   (in lmme_dl)")

    model = WhisperModel(args.model, device="cuda", compute_type="float16")
    args.out.mkdir(parents=True, exist_ok=True)

    media = [p for p in sorted(args.video.rglob("*")) if p.suffix.lower() in args.exts]
    if not media:
        sys.exit(f"No media in {args.video}/ (looked for {args.exts})")

    for v in media:
        out_path = args.out / (v.stem + ".md")
        if out_path.exists():
            print("skip (exists):", v.name); continue
        print("transcribing:", v.name)
        segments, _ = model.transcribe(str(v), vad_filter=True)
        lines = [f"# {v.stem}", ""]
        for seg in segments:
            lines.append(f"[{fmt(seg.start)}] {seg.text.strip()}")
        out_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"\nDone -> {args.out}/")


if __name__ == "__main__":
    main()
