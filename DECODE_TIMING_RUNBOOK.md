# DECODE_TIMING runbook (issue-#365 long-video TTFT attribution)

Goal: for long videos, attribute per-request TTFT between (a) PyNvVideoCodec HW
video decode wall-time and (b) vLLM prefill compute. This overlay adds an
opt-in, behavior-preserving decode timer; prefill is derived host-side.

## What changed and why

File: `vllm/multimodal/video.py` (branch `dev/jbottleson/hwdecode-decode-prefill-timing`,
based on `bb4ff49e` "memfix" HEAD).

1. `import time` (top of file) — needed for `time.monotonic()`.
2. New module constant `PYNVC_DECODE_TIMING` (next to the other env-tunable
   PyNvVideoCodec constants). Reads env `PYNVC_DECODE_TIMING`; OFF unless set to
   anything other than `0`/empty/`false`/`False`. Default OFF = zero behavior
   change for production runs.
3. In `PyNvVideoCodecVideoBackendMixin._decode_to_pinned_host` (the single
   PyNvVideoCodec frame-decode entry point used by `decode_frames_pynvvideocodec`):
   - capture `decode_t0 = time.monotonic()` immediately before
     `decoder.get_batch_frames_by_index(frame_idx)`;
   - after the existing `stream.synchronize()` (which forces the async NVDEC
     decode + device->host copy to complete — so the measured interval is true
     decode wall-time, not just kernel-launch time), emit one INFO line:
     `DECODE_TIMING elapsed_ms=<f> frame_count=<n> source=<basename>`.

Both the timer read and the log are guarded by `if PYNVC_DECODE_TIMING:`, so
when disabled there is no extra work beyond one already-true boolean check.

Identifier note: `_decode_to_pinned_host` does not have the vLLM `request_id`
in scope (it operates on raw `bytes` written to a temp `.mp4`). The greppable
field is `source=<temp mp4 basename>`. For the synthetic-video AIPerf workload
in the test cell there is effectively one distinct video, so the join to
per-request TTFT is by request ordering / count, not by a shared id. If a future
need arises to thread the real request id down here, that is a larger change
through the multimodal processor call chain and is intentionally out of scope.

### Why no PREFILL_TIMING log was added

Investigated the v1 engine/worker forward path. There is **no clean, low-risk,
accurate** single-point hook for per-request prefill wall-time:
- The GPU model runner (`vllm/v1/worker/gpu_model_runner.py`) runs ONE batched
  `forward()` over all scheduled requests, so wall-time cannot be attributed to
  an individual request there.
- Chunked prefill splits a single request's prefill across multiple engine
  steps, so there is no single "prefill done" instant in the runner.
- The only place prefill completion is cleanly detected per request is the
  scheduler (`vllm/v1/core/sched/scheduler.py`, `update_from_output`, where
  `num_output_tokens_before == 0`), but a wall-clock delta there
  (`time.time() - request.arrival_time`) includes queue wait + scheduler/KV
  overhead, i.e. it is ~TTFT, not pure prefill compute. Logging it would be
  redundant with the TTFT already exported by aiperf and Prometheus.

Per the task instruction ("if NOT clean/low-risk, do NOT force it"), prefill is
derived host-side instead — see below.

## Host-side derivation: prefill_ms = TTFT - decode_ms

For a long-video run where TTFT dominates latency (~97%):

- `TTFT` (per request, ms): from the aiperf artifact
  `profile_export_aiperf.json` — the `time_to_first_token` records (aiperf
  reports in ms). Equivalent aggregate is also in the vLLM Prometheus
  `vllm:time_to_first_token_seconds_*` captured in the cell metrics.
- `decode_ms` (per video, ms): from the `DECODE_TIMING elapsed_ms=...` log
  lines emitted by this overlay.
- `prefill_ms ≈ TTFT - decode_ms` (compute-bound prefill time), per request.

Joining per request: the test cell uses `--num-dataset-entries 1` +
`--dataset-sampling-strategy sequential`, so every request decodes the same
single synthetic video. There is therefore one `DECODE_TIMING` line per request
(decode is on the request's critical path before prefill). Match them by order
of appearance / by count:
- aggregate: `mean(decode_ms)` vs `mean(TTFT)` → mean prefill share.
- per request: zip the Nth `DECODE_TIMING` line to the Nth completed request in
  arrival order if a tighter attribution is wanted (the synthetic workload makes
  them interchangeable in practice).

## How the host validates

(All steps run by the host — sub-agent is sandbox-locked and cannot push/ssh/benchy.)

1. Push the branch:
   `git push <fork-remote> dev/jbottleson/hwdecode-decode-prefill-timing`
2. Capture the pushed SHA and fill it into the test cell:
   in `/home/user/projects/nvcv-agents/.scratch/issue365-hwdecode-sweep/cell-decode-timing.json`,
   replace `vllm_overlay.expected_sha` value `TBD_HOST_PUSH_SHA` with the real
   commit SHA (must equal what `fetch_ref` resolves to, else benchy rejects).
3. Run on a free H100 box via benchy with `cell-decode-timing.json`. The overlay
   already sets `env.PYNVC_DECODE_TIMING=1` (alongside
   `VLLM_VIDEO_LOADER_BACKEND=pynvvideocodec`), so the timer is active.
4. For a long-video run, raise `modality_config.duration` (default 5.0s in this
   cell) to a long value so TTFT is decode+prefill dominated.
5. Grep the vLLM container stdout / `vllm_server.log` in the run artifacts:
   - `grep DECODE_TIMING vllm_server.log` → confirm one line per video with a
     non-trivial `elapsed_ms` and the expected `frame_count`.
   - `grep PREFILL_TIMING vllm_server.log` → expected EMPTY (no prefill log by
     design; see above).
6. Confirm `decode_ms + prefill_ms ≈ TTFT`: take `mean(elapsed_ms)` from the
   `DECODE_TIMING` lines, take mean TTFT (ms) from `profile_export_aiperf.json`,
   and verify `mean(TTFT) - mean(decode_ms)` is a plausible positive prefill
   time (and that decode_ms is the expected large share for long videos).

## Expected overhead

Negligible. When `PYNVC_DECODE_TIMING` is unset (production default) the only
added cost is one boolean check per decode. When enabled, cost per video is two
`time.monotonic()` calls + one formatted INFO log — microseconds against a
decode that takes milliseconds-to-seconds for long videos.
