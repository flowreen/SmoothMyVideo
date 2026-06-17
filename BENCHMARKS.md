# SmoothMyVideo GMFSS speed benchmarks

Core GMFSS inference timing at 1080p (warmup + cuda.synchronize), excludes model load and ffmpeg I/O.
Lower is better. Each entry is a progress point as speedups land (fp16, cupy, torch.compile, TensorRT).

## 2026-06-15 01:44  (commit n/a)
- GPU: NVIDIA GeForce RTX 5090 Laptop GPU | torch 2.11.0+cu128 | model load 0.5s | sample 1920x1080
- **fp32**: reuse 653.0ms, inference 357.1ms/frame, pair@16x 6008.9ms, est 360f@16x ~36.0min
- **fp16**: reuse 466.1ms, inference 276.2ms/frame, pair@16x 4609.4ms, est 360f@16x ~27.6min

## 2026-06-15 01:54  (commit n/a)
- GPU: NVIDIA GeForce RTX 5090 Laptop GPU | torch 2.11.0+cu128 | model load 0.5s | sample 1920x1080
- **fp32**: reuse 777.1ms, inference 300.6ms/frame, pair@16x 5285.8ms, est 360f@16x ~31.6min
- **fp16**: reuse 472.6ms, inference 160.3ms/frame, pair@16x 2877.4ms, est 360f@16x ~17.2min
