# Third-party notices

Smooth My Video is MIT licensed (see [LICENSE](LICENSE)). It builds on the following
third-party work.

## Vendored in this repository

* **GMFSS_Fortuna** (the interpolation model: GMFlow flow network, IFNet/RIFE refiner,
  MetricNet, FeatureNet, FusionNet, softmax-splatting glue, and the `train_log` weights),
  vendored under `engine/GMFSS_Fortuna/`. MIT License, Copyright (c) 2023 98mxr; see
  `engine/GMFSS_Fortuna/LICENSE`. Upstream components it builds on: GMFlow (Apache-2.0,
  Haofei Xu et al.), RIFE/IFNet (MIT, hzwer et al.), softmax splatting (Simon Niklaus).
* **Real-ESRGAN** `realesr-animevideov3` architecture and weights (the `--restore` pass),
  vendored in `engine/realesr.py` / `engine/realesr-animevideov3.pth`. BSD 3-Clause License,
  Copyright (c) 2021 Xintao Wang.
* **AMD FidelityFX RCAS** (the sharpen pass), reimplemented in `engine/rcas.py` from AMD's
  FidelityFX-FSR reference. MIT License, Copyright (c) 2021 Advanced Micro Devices, Inc.
* **NVIDIA RTX Video SDK sample code**: the compiled bridge `engine/rtxvideo/rtxvideo_cuda.dll`
  is built from NVIDIA's SDK convenience layer (sources in `engine/rtxvideo/build_src/`),
  used under the NVIDIA RTX Video SDK license. The SDK's AI feature models (`nvngx_vsr.dll`,
  `nvngx_truehdr.dll`) are NOT redistributed; users install them from NVIDIA directly.

## Bundled in the release zip (not in this repository)

* **FFmpeg** (LGPL v2.1+ shared build by BtbN, `engine/bin/`): source code at
  https://github.com/BtbN/FFmpeg-Builds (LGPL builds link their exact sources per release).
* **CPython** via python-build-standalone (PSF License and component licenses).
* **PyTorch** (BSD-style), **CuPy** (MIT), **NumPy** (BSD), **OpenCV** (Apache-2.0),
  **ONNX / onnxscript** (Apache-2.0 / MIT).
* **NVIDIA TensorRT, cuDNN, CUDA runtime libraries**: redistributed as permitted by the
  NVIDIA software license agreements covering runtime redistribution.

## Optional, user-installed (never bundled or redistributed)

* **dovi_tool** and **hdr10plus_tool** (MIT, quietvoid): installed by the user via the app's
  Dolby Vision / HDR10+ panels.
* **NVIDIA RTX Video feature models**: installed by the user from NVIDIA's RTX Video SDK
  download under NVIDIA's EULA.

Dolby and Dolby Vision are trademarks of Dolby Laboratories. HDR10+ is a trademark of HDR10+
Technologies, LLC. This project is not affiliated with, endorsed by, or certified by either.
