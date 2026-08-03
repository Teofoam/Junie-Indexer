#!/usr/bin/env python3
"""
transcribe.py -- Transcribe lecture videos/audio to Markdown with timestamps.

Uses faster-whisper on the RTX 5060 in the JunieAirport env. Emits one .md per
source with [mm:ss] markers so the same chunker attaches TIME-range citations,
e.g. "Stachniss, Photogrammetry I (12:30-14:05)".

  python transcribe.py --video video --out transcripts
  python transcribe.py --video video --out transcripts --model large-v3-turbo
  python transcribe.py --video video --out transcripts \
      --prompt "Lecture on epipolar geometry, homography, bundle adjustment."

Run this SEPARATELY from extract.py -- both want the GPU, and large-v3 + Marker
together will not fit in 8GB VRAM.

ACCURACY NOTE: for a citation index, the failure that matters is not word error
rate in general -- it's domain jargon. Ordinary prose transcribes fine while the
exact terms Junie will search on get mangled ("epipolar" -> "epi-polar", XPBD ->
"X-P-B-D", surnames anywhere). A wrong article costs nothing; a wrong technical
term costs the retrieval. --prompt biases decoding toward vocabulary you supply
and is a bigger lever than model size. Use it per lecture series.
"""
import argparse, sys, time
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
                    help="large-v3 (best accuracy) ~4.7GB VRAM; large-v3-turbo is a "
                         "4-decoder-layer distillation -- several times faster and "
                         "lighter, slightly less accurate; medium is a fallback")
    ap.add_argument("--compute-type", default="float16",
                    help="float16 required on Blackwell/sm_120; int8 fails with "
                         "CUBLAS_STATUS_NOT_SUPPORTED via CTranslate2")
    ap.add_argument("--prompt", default="",
                    help="initial_prompt: seed vocabulary for this run, e.g. author "
                         "surnames and domain terms. Strongly recommended per series.")
    ap.add_argument("--beam-size", type=int, default=5,
                    help="faster-whisper's default; raising it buys little")
    ap.add_argument("--condition-on-previous", action="store_true",
                    help="re-enable Whisper's default text conditioning. OFF here "
                         "because it can drive repetition loops on long lectures, "
                         "and a looped segment poisons its timestamps too")
    ap.add_argument("--exts", nargs="*",
                    default=[".mp4", ".mkv", ".webm", ".mov", ".m4a", ".mp3", ".wav"])
    args = ap.parse_args()

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        sys.exit("pip install faster-whisper   (in JunieAirport)")

    model = WhisperModel(args.model, device="cuda", compute_type=args.compute_type)
    args.out.mkdir(parents=True, exist_ok=True)

    media = [p for p in sorted(args.video.rglob("*")) if p.suffix.lower() in args.exts]
    if not media:
        sys.exit(f"No media in {args.video}/ (looked for {args.exts})")

    if args.prompt:
        print(f"initial_prompt: {args.prompt[:120]}")

    for v in media:
        out_path = args.out / (v.stem + ".md")
        if out_path.exists():
            print("skip (exists):", v.name); continue
        print("transcribing:", v.name)
        t0 = time.time()

        segments, info = model.transcribe(
            str(v),
            vad_filter=True,
            beam_size=args.beam_size,
            initial_prompt=args.prompt or None,
            condition_on_previous_text=args.condition_on_previous,
        )

        lines = [f"# {v.stem}", ""]
        for seg in segments:
            lines.append(f"[{fmt(seg.start)}] {seg.text.strip()}")
        out_path.write_text("\n".join(lines), encoding="utf-8")

        # Wall time vs. audio duration tells you whether the model is actually
        # the bottleneck -- decide on large-v3-turbo from this, not in advance.
        dur = getattr(info, "duration", 0) or 0
        took = time.time() - t0
        rt = f", {dur / took:.1f}x realtime" if dur and took else ""
        print(f"  -> {out_path.name}  ({took:.0f}s{rt})")

    print(f"\nDone -> {args.out}/")


if __name__ == "__main__":
    main()
