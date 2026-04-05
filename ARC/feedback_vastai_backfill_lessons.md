---
name: Vast.ai backfill lessons learned
description: Hard lessons from 95M patent embedding run — script versioning, batch sizing, fleet management for the 400M journal run
type: feedback
---

Lessons from the patent backfill run (2026-04-04/05) — apply to journal embedding run.

**Why:** ~60-80% of GPU spend was wasted on duplicate processing, OOM crashes, and old script versions.

**How to apply:**

1. **Never mix script versions.** When updating the worker script, kill ALL instances and relaunch clean. "Kill half, keep half" leaves old-script workers that lack claims, skip-existing, and correct batch size. This caused 20 workers to re-process already-completed files.

2. **Auto-detect batch size from VRAM.** batch=1024 OOM'd on 24GB cards, batch=256 OOM'd on 12GB. Use `_auto_batch_size()` that checks `torch.cuda.get_device_properties(0).total_mem`. Safe values: 256 for 20GB+, 128 for 14GB+, 96 for 10GB+, 64 for 8GB+.

3. **fp16 from the start.** Halves model memory. `SentenceTransformer(model_name, model_kwargs={"torch_dtype": "float16"})`

4. **csv.field_size_limit(10_000_000)** — some patent/journal abstracts exceed the 128KB default.

5. **Claim mechanism from day 1.** R2-based claims (`backfill/claimed/{filename}`) prevent duplicate processing. Old workers without claims = invisible duplicate work.

6. **GPU compute capability >= 7.0 (Volta+).** Pascal (6.x) crashes FAISS GPU with unrecoverable C++ abort. Blackwell (RTX 5060/5070) lacks PyTorch 2.6 kernel support. Safe: Turing (2080 Ti), Ampere (3060+), Ada (4060+).

7. **Install order matters.** torch LAST with --force-reinstall (from arc_cloud_sentinel.py pattern). sentence-transformers pulls wrong torch version as dep.

8. **~20% instance failure rate on vast.ai.** Plan for it. Monitor agents check SSH + worker process + GPU util. Dead instances: destroy immediately.

9. **Large files dominate wall time.** G06F (1.8M docs) took one worker 20+ hours at batch=64. Sort files largest-first or assign big files to high-VRAM workers.

10. **Idle timeout configurable.** `--idle-passes 20` (20 min) lets workers wait for new files arriving (e.g., DOCDB front file export). Default 3 passes (90s) too aggressive.

11. **No-CPC files: batch 50K not 5K.** Per-file R2 overhead dominates at 5K docs/file. 50K reduces file count 10x.

12. **Download poller needs retry on ETag mismatch.** S3 download fails if file is being uploaded simultaneously. Add try/except with retry.
