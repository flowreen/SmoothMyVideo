# Third-party notices

Smooth My Video is MIT licensed (see [LICENSE](LICENSE)). It builds on the following
third-party work.

## Vendored in this repository

* **GMFSS_Fortuna** (the interpolation model: GMFlow flow network, IFNet/RIFE refiner,
  MetricNet, FeatureNet, FusionNet, softmax-splatting glue, and the `train_log` weights),
  vendored under `engine/GMFSS_Fortuna/`. MIT License, Copyright (c) 2023 98mxr; see
  `engine/GMFSS_Fortuna/LICENSE`. Upstream components it builds on: GMFlow (Apache-2.0,
  Haofei Xu et al.), RIFE/IFNet (MIT, hzwer et al.), softmax splatting (Simon Niklaus).
* **DRBA / Practical-RIFE 4.26 heavy** (the RIFE interpolation model and its optional DRBA
  anime-pacing timing: IFNet 4.26-heavy network + `flownet.pkl` weights, DistanceRatioMap
  adjuster, softmax splatting), vendored under `engine/rife/` + `engine/rife_backend.py` from
  routineLife1's DRBA repository. MIT License; see `engine/rife/DRBA-LICENSE.txt`. Upstream
  components: Practical-RIFE (MIT, hzwer et al.), softmax splatting (Simon Niklaus).
* **Real-ESRGAN** `realesr-animevideov3` architecture and weights (the `--restore` pass),
  vendored in `engine/realesr.py` / `engine/realesr-animevideov3.pth`. BSD 3-Clause License,
  Copyright (c) 2021 Xintao Wang.
* **Adaptive Sharpen** (the Sharpen pass): bacondither's DX11 two-pass HQ version (2021-09-10, from
  github.com/bacondither/Miscellaneous-shaders), ported to CUDA in the native host
  (`engine/live/build_src/smv-live-native.inl`, `k_sharpPlanar` / `k_sharpThdrIn`). BSD 2-Clause License:

  > Copyright (c) 2015-2021, bacondither
  > All rights reserved.
  >
  > Redistribution and use in source and binary forms, with or without
  > modification, are permitted provided that the following conditions
  > are met:
  > 1. Redistributions of source code must retain the above copyright
  >    notice, this list of conditions and the following disclaimer
  >    in this position and unchanged.
  > 2. Redistributions in binary form must reproduce the above copyright
  >    notice, this list of conditions and the following disclaimer in the
  >    documentation and/or other materials provided with the distribution.
  >
  > THIS SOFTWARE IS PROVIDED BY THE AUTHORS ``AS IS'' AND ANY EXPRESS OR
  > IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
  > OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
  > IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
  > INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
  > NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  > DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  > THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  > (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
  > THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
* **AMD FidelityFX SDK v2.3.0: FSR 4 frame generation and FSR upscaling** (the AMD FSR 4 frame generation model and
  the AMD FSR 4 upscaler): AMD's signed binaries `amd_fidelityfx_loader_dx12.dll` with
  `amd_fidelityfx_framegeneration_dx12.dll` (in `engine/fsrfg/`) and with `amd_fidelityfx_upscaler_dx12.dll` (in
  `engine/fsrup/`), redistributed unmodified beside our bridges `smv_fsrfg_bridge.dll` and `smv_fsrup_bridge.dll`
  (the recipes: `engine/fsrfg/build_src/`, `engine/fsrup/build_src/`). At run time `smv_fsrup_bridge.dll` changes
  one function of the upscaler DLL in memory (its GPU check), so FSR 4's model runs on NVIDIA GPUs; the files on
  disk are AMD's. Every one of these files is listed in the MIT section of the SDK's
  `Kits/FidelityFX/docs/license.md` (reproduced in `engine/fsrfg/licenses/`). MIT License:

  > Copyright (C) Advanced Micro Devices, Inc.
  >
  > Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
  > documentation files (the "Software"), to deal in the Software without restriction, including without limitation
  > the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
  > permit persons to whom the Software is furnished to do so, subject to the following conditions:
  >
  > The above copyright notice and this permission notice shall be included in all copies or substantial portions of
  > the Software.
  >
  > THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
  > THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  > AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
  > TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  > SOFTWARE.
* **vkd3d-proton** (D3D12 on Vulkan: it runs AMD's FSR 4 frame generation on NVIDIA GPUs):
  `engine/fsrfg/smv_vkd3d_d3d12.dll` and `engine/fsrfg/smv_vkd3d_d3d12core.dll`, built from vkd3d-proton
  (github.com/HansKristian-Work/vkd3d-proton) commit 2230755878b01993b4b82d0ac0c0624f3ab4d333 with its dxil-spirv at
  ab47c3df1a4746f36c958f0262bc9e314eb566eb, both modified: the patches ship beside the DLLs in `engine/fsrfg/source/`,
  the build steps are in `engine/fsrfg/build_src/build.py`. GNU Lesser General Public License 2.1 (vkd3d-proton's
  COPYING, LICENSE and AUTHORS in `engine/fsrfg/licenses/`); the app loads the two DLLs at run time, so a rebuild from
  the corresponding source replaces them. Built into them, with their notices in `engine/fsrfg/licenses/`: dxil-spirv
  (MIT), its bc-decoder (MIT, Copyright (c) 2019-2020 Baldur Karlsson), the glslang SPIR-V builder (BSD 3-Clause) and
  SPIRV-Headers (MIT).
* `engine/fsrfg/amdxc64.dll` is this project's own code (`engine/fsrfg/build_src/amdxc64_shim.cpp`, MIT): it answers
  the driver-extension queries of AMD's frame generation DLL so the DLL offers FSR 4; it contains no AMD code.
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
* **NVIDIA TensorRT for RTX, cuDNN, CUDA runtime libraries**: redistributed as permitted by the
  NVIDIA software license agreements covering runtime redistribution.

## Optional, user-installed (never bundled or redistributed)

* **dovi_tool** and **hdr10plus_tool** (MIT, quietvoid): installed by the user via the app's
  Dolby Vision / HDR10+ panels.
* **NVIDIA RTX Video feature models**: installed by the user from NVIDIA's RTX Video SDK
  download under NVIDIA's EULA.

Dolby and Dolby Vision are trademarks of Dolby Laboratories. HDR10+ is a trademark of HDR10+
Technologies, LLC. This project is not affiliated with, endorsed by, or certified by either.
