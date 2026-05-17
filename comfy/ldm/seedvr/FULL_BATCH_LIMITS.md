# SeedVR2 Native Full-Batch Limits

Native SeedVR2 full-batch execution is not a parity claim for every pixel load.

The native path has a documented full-batch ceiling for this clean execution shape:

- input video shape: `320x240x100`
- upscale target shape: `1280x960x100`
- chunk geometry: `frames_per_chunk=101`
- encode tiling: `false`
- decode tiling: `false`
- fallback: `false`
- sampling: `steps=1`, `cfg=1.0`, `sampler=euler`
- attention caller shape: inline `optimized_var_attention(q=concat_win(...), k=concat_win(...), v=concat_win(...))`
- repeat-concat shape: `lambda vid, txt: torch.cat([vid, txt])[tgt_idx]`

Under that clean native execution shape, DiT inference does not complete before CUDA OOM. The full-batch native path is therefore ceiling-limited at this recorded pixel load. Use a smaller native chunk, tiling, or a separately verified attention backend when this ceiling is reached.
