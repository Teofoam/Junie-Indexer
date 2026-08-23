# JunieLib — GPU crash investigation

**System:** Lenovo laptop, Intel Arrow Lake (20 cores), NVIDIA GeForce RTX 5060 Laptop
GPU (8 GB, PCI 01:00.0), Windows 11 Pro 26100.6584, BIOS 1.57 (current)
**Workload:** `marker-pdf` 1.10.2 + `surya-ocr` 0.17.1, sustained GPU inference over
multi-hour document OCR
**Period:** 2026-08-12 → 2026-08-23
**Status:** unresolved. Twenty-two bugchecks. Root cause narrowed but not identified.
**Workaround in use:** CPU-only inference. 21 of 23 books complete.

---

## Summary

Sustained OCR inference bugchecks this machine, reproducibly, within minutes. Two
bugcheck families, both with the NVIDIA kernel driver `nvlddmkm.sys` implicated:

| Code | Name | Count | Signature |
|---|---|---|---|
| `0x00020001` | `HYPERVISOR_ERROR` | 8 | `arg1=0x28` — *internal error in the I/O MMU module* |
| `0x00000101` | `CLOCK_WATCHDOG_TIMEOUT` | 11+ | `nvlddmkm` on the stack, `Stack.Pointer: ISR` |

Every `0x101` dump resolves to the identical failure bucket and hash:

```
Failure.Bucket:  CLOCK_WATCHDOG_TIMEOUT_nvlddmkm!unknown_function
Failure.Hash:    {5efc291b-e569-a793-5506-abbfebba7835}
Stack.Pointer:   ISR
```

Sample stack (082226-13421-01.dmp):

```
nt!KiDpcInterrupt+0x39f
nvlddmkm+0xb7e94f
nvlddmkm+0x11abbd
nvlddmkm+0x1011b8
```

**The WHEA log is empty.** Zero hardware error records across all twenty crashes. This
conclusively clears CPU and RAM — a genuine electrical or logic fault in either would be
recorded. It says nothing about the GPU, whose internal faults are handled inside the
NVIDIA driver and never reach WHEA.

---

## Mechanism

`CLOCK_WATCHDOG_TIMEOUT` fires when a core fails to service the clock interrupt for ~10
ticks. An ISR runs above clock IRQL, so while a core is inside one the clock interrupt
cannot preempt it. Every dump places the hung core inside `nvlddmkm`'s interrupt service
routine. The core is not asleep and not faulty — it is stuck in the graphics driver at a
priority nothing can interrupt.

The hung core index varied across crashes: **0, 1, 6, 7, 18, 19**. Not a defective core;
whichever one took the interrupt.

---

## Variables eliminated, each by direct test

| Variable | Tested | Result |
|---|---|---|
| Driver build | 610.88 (Jul 2026) and 610.47 (May 2026), DDU clean install in Safe Mode | Identical failure hash |
| VBS / hypervisor | Enabled and fully disabled (`hypervisorlaunchtype off`) | Crashes both ways |
| Memory Integrity (HVCI) | On and off | Crashes both ways |
| CPU idle states | Default and `IDLEDISABLE=1` | Crashes both ways |
| GPU core clock | Free (3090 MHz) and locked to 1492 MHz | Crashes both ways |
| GPU memory clock | Free (12001 MHz) and locked to 9001 MHz | Crashes both ways |
| Surya batch sizes | 64/8/12/12/12 and 16/4/6/6/6 | Crashes both ways; peak VRAM differed by 1 MiB |
| Page content | Probe on a different page range in a separate PDF | Crashed at the same mark |
| HAGS | Default (on) and `HwSchMode=1` (off) | See below — best result obtained, not a fix |
| BIOS | 1.57, confirmed current | n/a |
| Thermals | 51–74 °C observed | Never near limit |
| Power draw | 15–79 W of a 115 W limit | Never near limit |
| NVIDIA extras | ShadowPlay / NVIDIA App / telemetry removed | No change |

### Hardware-Accelerated GPU Scheduling

Disabling HAGS produced the single best result of the investigation, and it is the only
change that ever did:

| | HAGS on | HAGS off |
|---|---|---|
| Time to crash under load | 90 s – 3 min | **2 h 04 m 35 s** (one chunk completed) |

It was not reproducible. The next four attempts crashed at 3m17s, 3m25s, 3m35s, 3m42s.
The two-hour run stands alone as an outlier.

---

## Determinism

Attempts on book 20 (419-page scan, `--force-ocr`), all with HAGS off, driver 610.47,
VBS off:

| Attempt | Pages | Duration |
|---|---|---|
| chunk 1 | 0–149 | **2h 04m 35s — completed** |
| chunk 2 | 150–299 | 3m 25s |
| chunk 2 | 150–299 | 3m 35s |
| chunk 2 | 150–299 | 3m 42s |
| probe | **300–418, separate PDF** | 3m 17s |

Four consecutive failures inside a 25-second window, across two different page ranges and
two different files. Static analysis found the pages homogeneous — every page a single
full-page scan, median render 2.50 MP at 192 DPI, no page deviating more than 25% from
median, no outsized embedded image. Content is not the trigger.

---

## Telemetry at the moment of a crash

Captured at 2-second intervals, flushed per line so the record survives a bugcheck:

| Metric | At crash | Limit |
|---|---|---|
| SM clock | 1492 MHz (locked) | 3090 MHz |
| Power draw | 15.2 W | 115 W |
| Temperature | 51 °C | ~90 °C |
| GPU utilisation | 100% | — |
| VRAM | 7754 MiB | 8151 MiB |

The GPU died cool, quiet, and at under half its clock ceiling. Thermal, electrical, and
frequency causes are all excluded by direct measurement.

---

## Synthetic stress tests — both passed

Generic PyTorch (2.13.0+cu130), no marker, no surya:

| Test | Configuration | Result |
|---|---|---|
| v1 | fp16 matmul loop, 0.97 GB resident, negligible PCIe traffic | **Survived 12 min**, 22,000 iterations |
| v2 | 7789 MiB VRAM (marker-equivalent) + pinned-memory DMA both directions | **Survived 12 min**, 659 iterations, **126.5 GB over PCIe** |

This is the most awkward result in the investigation. The GPU sustains marker-equivalent
memory occupancy and 126 GB of DMA without incident, yet marker itself fails within
minutes. Raw compute, memory occupancy, and DMA volume are all cleared.

---

## Attention backends — tested and eliminated

The stress tests exercised cuBLAS matmul, never `scaled_dot_product_attention`. Surya
calls SDPA directly (`surya/common/adetr/decoder.py:210,333`,
`surya/common/donut/encoder.py:427`) and auto-selects the `sdpa` implementation, so
PyTorch dispatches to a fused attention kernel on every page. On sm_120 with cuDNN 9.20
those are among the newest kernels in the stack, which made them a strong suspect.

Backends were disabled via a `sitecustomize.py` on `PYTHONPATH` — imported automatically
at interpreter startup, so it reaches marker's subprocess without patching marker or
surya. Each run logged proof of application to `sdpa_patch.log`; both runs confirmed the
setting active in all four processes.

| Round | Configuration | Result |
|---|---|---|
| 1 | cuDNN off; flash + mem-efficient still active | **Crashed at ~1m11s** |
| 2 | cuDNN, flash **and** mem-efficient all off — `math` backend only, zero fused attention kernels anywhere | **Crashed at ~2m36s** |

Round 2 is decisive: with no fused attention kernel in the process at all, the machine
still bugchecks. Fused attention is not the trigger. Batch sizes were reduced to 8/2/4/4
for round 2 to give the math backend VRAM headroom (it materialises the full attention
matrix); this is not a confound, since 16 and 64 had already been shown to crash
identically. No CUDA OOM occurred, so the result is valid.

Also verified during this work: `torch.cuda.get_arch_list()` contains `sm_120`, so
Blackwell kernels are natively compiled and there is **no PTX JIT fallback**. torch
2.13.0+cu130 against a driver exposing CUDA 13.3 is the correct, compatible direction.
The CUDA installation is healthy.

**Remaining untested axis:** GPU command submission rate. Marker issues thousands of small
kernel launches per second with dynamic tensor shapes (variable-size text crops) and swaps
models between detection, layout, and recognition stages. Both stress tests issued few,
large, statically-shaped operations. Submission rate drives interrupt rate, and the fault
is in the interrupt handler.

---

## Assessment

The evidence does not cleanly support either conclusion:

**For hardware:** twenty crashes; two independent driver builds behave identically; every
software and firmware variable eliminated by test; the fault is in the driver's interrupt
path where a hardware handshake would manifest.

**Against hardware:** the GPU passes sustained synthetic load at matching VRAM and heavy
DMA; thermals, power, and clocks all have large headroom; WHEA is empty.

The most consistent remaining hypothesis is a fault in the driver's interrupt handling
under **high command-submission rate** — a workload profile that sustained transformer
inference produces and that gaming, which this driver path is primarily validated
against, does not. Whether the root cause is marginal silicon exposed only by that
pattern, or a driver defect, is not determined by the evidence collected here.

## Reproduction

1. `marker-pdf` 1.10.2 with `--force-ocr` on a scanned PDF, 150-page chunks
2. RTX 5060 Laptop, 8 GB, driver 610.47 or 610.88
3. Bugcheck within 90 s – 4 min of inference starting, reliably

## Artefacts

- `C:\Windows\Minidump\` — dumps from 081226 onward
- `gpu_telemetry.log.*` — 2-second GPU telemetry across four crashes
- `gpu_stress.log`, `gpu_stress2.log` — synthetic stress test records
- `run.log` — full extraction history with timings

## Current workaround

CPU-only inference (`TORCH_DEVICE=cpu`, see `run_extract_cpu.cmd`). Slower, but the fault
does not occur. 21 of 23 books completed; the remaining two are running on CPU.
