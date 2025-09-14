# REAL Video Enhancer: Elevate Your Videos with AI-Powered Upscaling and Frame Interpolation

**Instantly enhance your videos on Windows, Linux, and macOS with AI-powered frame interpolation and upscaling using [REAL Video Enhancer](https://github.com/TNTwise/REAL-Video-Enhancer)**

[![Visitors](https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fgithub.com%2FTNTwise%2FREAL-Video-enhancer%2F&countColor=%23263759)](https://github.com/TNTwise/REAL-Video-enhancer/)
[![Discord](https://img.shields.io/discord/1041502781808328704?label=Discord&logo=discord)](https://discord.gg/hwGHXga8ck)
[![License](https://img.shields.io/github/license/tntwise/real-video-enhancer)](https://github.com/TNTwise/REAL-Video-Enhancer/blob/main/LICENSE)
[![Version](https://img.shields.io/badge/Version-2.3.6-blue)](https://github.com/TNTwise/REAL-Video-Enhancer/releases/latest)
[![Downloads](https://img.shields.io/github/downloads/tntwise/REAL-Video-Enhancer/total.svg?label=Downloads%40Total)](https://github.com/TNTwise/REAL-Video-Enhancer/releases/latest)
[![FlatHub](https://dl.flathub.org/assets/badges/flathub-badge-en.svg)](https://flathub.org/apps/io.github.tntwise.REAL-Video-Enhancer)

<p align=center>
  <img src="https://github.com/TNTwise/REAL-Video-Enhancer/raw/2.0/icons/logo-v2.svg" width = "25%">
</p>

## Key Features

*   **Cross-Platform Support:** Available on Windows, Linux, and macOS.
*   **Frame Interpolation:** Smoothly increase frame rates for fluid motion.
*   **Video Upscaling:** Enhance video resolution with AI-powered models.
*   **Efficient Inference:** Utilize TensorRT and NCNN for optimized performance on various GPUs.
*   **Scene Change Detection:** Preserves sharp transitions in your videos.
*   **Discord RPC Integration:** Displays your current processing status.
*   **User-Friendly Preview:** See the latest rendered frame in real-time.

## Table of Contents

*   [Introduction](#introduction)
*   [Key Features](#key-features)
*   [Hardware and Software Requirements](#hardware-and-software-requirements)
*   [Models](#models)
    *   [Interpolate Models](#interpolate-models)
    *   [Upscale Models](#upscale-models)
    *   [Decompression Models](#decompression-models)
*   [Backends](#backends)
*   [FAQ](#faq)
    *   [General Application Usage](#general-application-usage)
    *   [TensorRT Related Questions](#tensorrt-related-questions)
    *   [ROCm Related Questions](#rocm-related-questions)
    *   [NCNN Related Questions](#ncnn-related-questions)
*   [Cloning](#cloning)
*   [Building](#building)
*   [Colab Notebook](#colab-notebook)
*   [Credits](#credits)

## Hardware and Software Requirements

| Feature         | Minimum                  | Recommended                 |
| :-------------- | :----------------------- | :-------------------------- |
| CPU             | Dual Core x64 bit        | Quad Core x64 bit           |
| GPU             | Vulkan 1.3 capable       | Nvidia RTX GPU (20 series+) |
| VRAM            | 4 GB - NCNN             | 8 GB - TensorRT             |
| RAM             | 16 GB                    | 32 GB                       |
| Storage         | 1 GB free - NCNN         | 16 GB free - TensorRT       |
| Operating System | Windows 10/11 / MacOS 14+ | Ubuntu 22.04+ (Recommended) |

## Models

### Interpolate Models

| Model      | Author  | Link                                                     |
| :--------- | :------ | :------------------------------------------------------- |
| RIFE 4.6+  | Hzwer   | [Practical-RIFE](https://github.com/hzwer/Practical-RIFE) |
| GMFSS      | 98mxr   | [GMFSS_Fortuna](https://github.com/98mxr/GMFSS_Fortuna)   |
| GIMM       | GSeanCDAT | [GIMM](https://github.com/GSeanCDAT/GIMM-VFI)       |
| IFRNet     | ltkong218 | [IFRnet](https://github.com/ltkong218/IFRNet)          |

### Upscale Models

| Model                 | Author         | Link                                                                     |
| :-------------------- | :------------- | :----------------------------------------------------------------------- |
| 4x-SPANkendata        | Crustaceous D  | [4x-SPANkendata](https://openmodeldb.info/models/4x-SPANkendata)           |
| 4x-ClearRealityV1     | Kim2091        | [4x-ClearRealityV1](https://openmodeldb.info/models/4x-ClearRealityV1)    |
| 4x-Nomos8k-SPAN series | Helaman        | [4x-Nomos8k-SPAN series](https://openmodeldb.info/models/4x-Nomos8k-span-otf-strong) |
| 2x-OpenProteus        | SiroSky        | [OpenProteus](https://github.com/Sirosky/Upscale-Hub/releases/tag/OpenProteus) |
| 2x-AnimeJaNai         | The Database   | [AnimeJanai](https://github.com/the-database/mpv-upscale-2x_animejanai)    |
| 2x-AniSD              | SiroSky        | [AniSD](https://github.com/Sirosky/Upscale-Hub/releases/tag/AniSD)         |

### Decompression Models

| Model    | Author  | Link                                                                |
| :------- | :------ | :------------------------------------------------------------------ |
| DeH264   | Helaman | [1xDeH264_realplksr](https://github.com/Phhofm/models/releases/tag/1xDeH264_realplksr) |

## Backends

| Backend   | Hardware                             |
| :-------- | :----------------------------------- |
| TensorRT  | NVIDIA RTX GPUs                      |
| PyTorch   | CUDA 12.6 and ROCm 6.2 capable GPUs |
| NCNN      | Vulkan 1.3 capable GPUs              |

## FAQ

### General Application Usage

| Question                                                                | Answer                                                                                                                                         |
| :---------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------- |
| What does this program attempt to accomplish?                           | Fast, efficient, and accessible video interpolation (e.g., 24->48FPS) and video upscaling (e.g., 1920->3840).                            |
| Why is it failing to recognize installed backends?                     | REAL Video Enhancer uses PIP and portable python for inference, which can sometimes have issues. Please attempt reinstalling the app. |

### TensorRT Related Questions

| Question                                                              | Answer                                                                                                                                     |
| :-------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------- |
| Why does it take so long to begin inference?                          | TensorRT uses advanced optimization at the beginning of inference based on your device, done only once per resolution.                      |
| Why does the optimization and inference fail?                         | The most common cause of optimization failure is **Limited VRAM**. There is no fix besides using CUDA or NCNN instead.                      |

### ROCm Related Questions

| Question                                 | Answer                                                                                               |
| :--------------------------------------- | :--------------------------------------------------------------------------------------------------- |
| Why am I getting (Insert Error here)?   | ROCm can be buggy.  Please consult the [ROCm Help](https://github.com/TNTwise/REAL-Video-Enhancer/wiki/ROCm-Help). |

### NCNN Related Questions

| Question                                    | Answer                                                                                                                                                                |
| :------------------------------------------ | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Why am I getting (Insert Vulkan Error here)? | This is usually an OOM (Out Of Memory) error. It can indicate a weak iGPU or old GPU.  Consider using the [Colab Notebook](https://github.com/TNTwise/REAL-Video-Enhancer-Colab) instead. |

## Cloning

```bash
# Nightly
git clone --recurse-submodules https://github.com/TNTwise/REAL-Video-Enhancer

# Stable
git clone --recurse-submodules https://github.com/TNTwise/REAL-Video-Enhancer --branch 2.3.4
```

## Building

*   **Supported Build Methods:**
    *   pyinstaller (recommended for Win/Mac)
    *   cx_freeze (recommended for Linux)
    *   nuitka (experimental)
*   **Supported Python Versions:** 3.10, 3.11, 3.12

```bash
python3 build.py --build BUILD_OPTION --copy_backend
```

## Colab Notebook

[Colab Notebook](https://github.com/tntwise/REAL-Video-Enhancer-Colab)

## Credits

### People

| Person            | Contribution                                                                  | Link                                         |
| :---------------- | :---------------------------------------------------------------------------- | :------------------------------------------- |
| NevermindNilas    | Some backend and reference code, working with me on many projects           | [NevermindNilas](https://github.com/NevermindNilas/)    |
| Styler00dollar    | RIFE ncnn models (4.1-4.5, 4.7-4.12-lite), Sudo Shuffle Span and benchmarking | [styler00dollar](https://github.com/styler00dollar) |
| HolyWu            | TensorRT engine generation code, inference optimizations, and RIFE fixes     | [HolyWu](https://github.com/HolyWu/)         |
| Rick Astley       | Amazing music                                                                 | [YouTube](https://www.youtube.com/watch?v=dQw4w9WgXcQ) |

### Software

| Software Used                 | Purpose                                                                 | Link                                                                                               |
| :---------------------------- | :---------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------- |
| FFmpeg                        | Multimedia framework                                                   | [FFmpeg](https://ffmpeg.org/)                                                                      |
| FFmpeg Builds                 | Pre-compiled builds of FFMpeg.                                         | Windows/Linux: [FFmpeg-Builds](https://github.com/BtbN/FFmpeg-Builds), MacOS: [mpv-mac](https://github.com/eko5624/mpv-mac) |
| PyTorch                       | Neural Network Inference (CUDA/ROCm/TensorRT)                          | [PyTorch](https://pytorch.org/)                                                                      |
| NCNN                          | Neural Network Inference (Vulkan)                                        | [NCNN](https://github.com/tencent/ncnn)                                                              |
| RIFE                          | Real-Time Intermediate Flow Estimation for Video Frame Interpolation     | [RIFE](https://github.com/hzwer/Practical-RIFE)                                                        |
| rife-ncnn-vulkan              | Video frame interpolation implementation using NCNN and Vulkan           | [rife-ncnn-vulkan](https://github.com/nihui/rife-ncnn-vulkan)                                        |
| rife ncnn vulkan python       | Python bindings for RIFE NCNN Vulkan implementation                   | [rife-ncnn-vulkan-python](https://github.com/media2x/rife-ncnn-vulkan-python)                         |
| GMFSS                         | GMFlow based Anime VFI                                                  | [GMFSS](https://github.com/98mxr/GMFSS_Fortuna)  |
| GIMM                          | Motion Modeling Realistic VFI                                           | [GIMM](https://github.com/GSeanCDAT/GIMM-VFI)                                                            |
| ncnn python                   | Python bindings for NCNN Vulkan framework                             | [ncnn](https://pypi.org/project/ncnn)                                                              |
| Real-ESRGAN                   | Upscaling                                                              | [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)                                                    |
| SPAN                          | Upscaling                                                              | [SPAN](https://github.com/hongyuanyu/SPAN)                                                              |
| Spandrel                      | CUDA upscaling model architecture support                            | [Spandrel](https://github.com/chaiNNer-org/spandrel)                                                   |
| ChaiNNer                      | Model Scale Detection                                                  | [ChaiNNer](https://github.com/chaiNNer-org/chainner)                                                    |
| cx_Freeze                     | Tool for creating standalone executables from Python scripts (Linux build) | [cx_Freeze](https://github.com/marcelotduarte/cx_Freeze)                                              |
| PyInstaller                   | Tool for creating standalone executables from Python scripts (Win/Mac builds) | [PyInstaller](https://github.com/pyinstaller/pyinstaller)                                            |
| Feather Icons                 | Open source icons library                                               | [Feather Icons](https://github.com/feathericons/feather)                                                |
| PySceneDetect                 | Transition detection library for python                               | [PySceneDetect](https://github.com/Breakthrough/PySceneDetect/)                                         |
| Python Standalone Builds      | Backend inference using portable python                                 | [python-build-standalone](https://github.com/indygreg/python-build-standalone)                       |

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=tntwise/real-video-enhancer&type=Date)](https://star-history.com/#tntwise/real-video-enhancer&Date)