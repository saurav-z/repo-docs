# REAL Video Enhancer: Supercharge Your Videos with AI-Powered Upscaling and Frame Interpolation

**Enhance video quality with ease!** REAL Video Enhancer is a versatile application that leverages cutting-edge AI to upscale video resolution and create smooth, high-frame-rate videos, offering a significant upgrade over tools like Flowframes.

[![Visitors](https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fgithub.com%2FTNTwise%2FREAL-Video-enhancer%2F&countColor=%23263759)](https://github.com/TNTwise/REAL-Video-Enhancer)
[![pypresence](https://img.shields.io/badge/using-pypresence-00bb88.svg?style=for-the-badge&logo=discord&logoWidth=20)](https://github.com/qwertyquerty/pypresence)
[![License](https://img.shields.io/github/license/tntwise/real-video-enhancer)](https://github.com/TNTwise/REAL-Video-Enhancer/blob/main/LICENSE)
[![Version](https://img.shields.io/badge/Version-2.3.6-blue)](https://github.com/TNTwise/REAL-Video-Enhancer/releases/latest)
[![Downloads](https://img.shields.io/github/downloads/tntwise/REAL-Video-Enhancer/total.svg?label=Downloads@Total)](https://github.com/TNTwise/REAL-Video-Enhancer/releases/latest)
[<img src="https://img.shields.io/discord/1041502781808328704?label=Discord" alt="Discord Shield"/>](https://discord.gg/hwGHXga8ck)
<a href="https://flathub.org/apps/io.github.tntwise.REAL-Video-Enhancer">
    <img src="https://dl.flathub.org/assets/badges/flathub-badge-en.svg" height="50px"/>
  </a>
<p align=center>
  <img src="https://github.com/TNTwise/REAL-Video-Enhancer/raw/2.0/icons/logo-v2.svg" width = "25%">
</p>

**[➡️ View the Source Code on GitHub](https://github.com/TNTwise/REAL-Video-Enhancer)**

## Key Features:

*   **Cross-Platform Support:** Works seamlessly on Windows, Linux, and macOS.
*   **Frame Interpolation:** Smoothly increases frame rates for a cinematic experience.
*   **Video Upscaling:** Enhances video resolution for sharper details.
*   **Scene Change Detection:** Preserves sharp transitions by intelligently adapting to scene changes.
*   **Efficient Inference:** Utilizes TensorRT and NCNN backends for optimized performance on various GPUs.
*   **Discord Rich Presence:**  Displays application status within Discord.
*   **Preview Functionality:** Provides a real-time preview of the processed frames.

## Table of Contents

*   [Introduction](#introduction)
*   [Key Features](#key-features)
*   [Hardware & Software Requirements](#hardware-software-requirements)
*   [Models](#models)
    *   [Interpolate Models](#interpolate-models)
    *   [Upscale Models](#upscale-models)
    *   [Decompression Models](#decompression-models)
*   [Backends](#backends)
*   [FAQ](#faq)
    *   [General Application Usage](#general-application-usage)
    *   [TensorRT-Related Questions](#tensorrt-related-questions)
    *   [ROCm-Related Questions](#rocm-related-questions)
    *   [NCNN-Related Questions](#ncnn-related-questions)
*   [Cloning](#cloning)
*   [Building](#building)
*   [Colab Notebook](#colab-notebook)
*   [Credits](#credits)
    *   [People](#people)
    *   [Software](#software)
*   [Star History](#star-history)

## Hardware & Software Requirements

| Feature           | Minimum                          | Recommended                          |
| :---------------- | :------------------------------- | :----------------------------------- |
| CPU               | Dual Core x64 bit                 | Quad Core x64 bit                    |
| GPU               | Vulkan 1.3 capable device        | Nvidia RTX GPU (20 series and up)    |
| VRAM              | 4 GB - NCNN                      | 8 GB - TensorRT (or more)              |
| RAM               | 16 GB                            | 32 GB                                |
| Storage           | 1 GB free - NCNN                | 16 GB free - TensorRT                |
| Operating System  | Windows 10/11 64bit / MacOS 14+ | Any modern Linux distro (Ubuntu 22.04+) |

## Models

### Interpolate Models

| Model        | Author    | Link                                                        |
| :----------- | :-------- | :---------------------------------------------------------- |
| RIFE         | Hzwer     | [Practical-RIFE](https://github.com/hzwer/Practical-RIFE)     |
| GMFSS        | 98mxr     | [GMFSS_Fortuna](https://github.com/98mxr/GMFSS_Fortuna)      |
| GIMM         | GSeanCDAT | [GIMM](https://github.com/GSeanCDAT/GIMM-VFI)                |
| IFRNet       | ltkong218 | [IFRnet](https://github.com/ltkong218/IFRNet)               |

### Upscale Models

| Model                    | Author          | Link                                                                  |
| :----------------------- | :-------------- | :-------------------------------------------------------------------- |
| 4x-SPANkendata           | Crustaceous D   | [4x-SPANkendata](https://openmodeldb.info/models/4x-SPANkendata)      |
| 4x-ClearRealityV1        | Kim2091         | [4x-ClearRealityV1](https://openmodeldb.info/models/4x-ClearRealityV1) |
| 4x-Nomos8k-SPAN series    | Helaman         | [4x-Nomos8k-SPAN series](https://openmodeldb.info/models/4x-nomos8k-span-otf-strong) |
| 2x-OpenProteus           | SiroSky         | [OpenProteus](https://github.com/Sirosky/Upscale-Hub/releases/tag/OpenProteus)         |
| 2x-AnimeJaNai V2 and V3 Sharp | The Database | [AnimeJanai](https://github.com/the-database/mpv-upscale-2x_animejanai)                 |
| 2x-AniSD                 | SiroSky         | [AniSD](https://github.com/Sirosky/Upscale-Hub/releases/tag/AniSD)       |

### Decompression Models

| Model     | Author  | Link                                                     |
| :-------- | :------ | :------------------------------------------------------- |
| DeH264    | Helaman | [1xDeH264_realplksr](https://github.com/Phhofm/models/releases/tag/1xDeH264_realplksr) |

## Backends

| Backend   | Hardware                             |
| :-------- | :----------------------------------- |
| TensorRT  | NVIDIA RTX GPUs                      |
| PyTorch   | CUDA 12.6 and ROCm 6.2 capable GPUs  |
| NCNN      | Vulkan 1.3 capable GPUs              |

## FAQ

### General Application Usage

| Question                                  | Answer                                                                                                                                      |
| :---------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------ |
| What does this program attempt to accomplish? | Fast, efficient, and accessible video interpolation (e.g., 24->48FPS) and video upscaling (e.g., 1920->3840).                        |
| Why is it failing to recognize installed backends? | REAL Video Enhancer uses PIP and portable python for inference. This can sometimes cause installation issues. Reinstall the app before creating an issue. |

### TensorRT-Related Questions

| Question                                  | Answer                                                                                                                                                                   |
| :---------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Why does it take so long to begin inference? | TensorRT optimizes at the beginning of inference based on your device, done only once per video resolution.                                                                 |
| Why does the optimization and inference fail? | The most common cause of optimization failure is **Limited VRAM**. There's no direct fix; try CUDA or NCNN instead.                                                                    |

### ROCm-Related Questions

| Question                    | Answer                                                                                  |
| :-------------------------- | :-------------------------------------------------------------------------------------- |
| Why am I getting (Insert Error here)? | ROCm can be buggy. See the [ROCm Help](https://github.com/TNTwise/REAL-Video-Enhancer/wiki/ROCm-Help) for troubleshooting. |

### NCNN-Related Questions

| Question                                  | Answer                                                                                                                                           |
| :---------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------- |
| Why am I getting (Insert Vulkan Error here)? | This usually indicates an OOM (Out Of Memory) error. This could mean you have a weak iGPU or very old GPU, try the [Colab Notebook](#colab-notebook) instead. |

## Cloning

```bash
# Nightly
git clone --recurse-submodules https://github.com/TNTwise/REAL-Video-Enhancer

# Stable (e.g., v2.3.6)
git clone --recurse-submodules https://github.com/TNTwise/REAL-Video-Enhancer --branch 2.3.6
```

## Building

*   **Build Methods:**
    *   pyinstaller (Recommended for Win/Mac)
    *   cx_freeze (Recommended for Linux)
    *   nuitka (Experimental)
*   **Supported Python Versions:** 3.10, 3.11, 3.12

```bash
python3 build.py --build BUILD_OPTION --copy_backend
```

## Colab Notebook

*   [Colab Notebook](https://github.com/tntwise/REAL-Video-Enhancer-Colab)

## Credits

### People

| Person             | For                                                        | Link                                               |
| :----------------- | :--------------------------------------------------------- | :------------------------------------------------- |
| NevermindNilas     | Backend and reference code, project collaboration             | https://github.com/NevermindNilas/                 |
| Styler00dollar     | RIFE ncnn models, SPAN and benchmarking                    | https://github.com/styler00dollar                 |
| HolyWu             | TensorRT engine generation, inference optimizations, RIFE fixes | https://github.com/HolyWu/                         |
| Rick Astley          | Amazing music | https://www.youtube.com/watch?v=dQw4w9WgXcQ |

### Software

| Software Used         | For                                                                   | Link                                                                                                    |
| :-------------------- | :-------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------ |
| FFmpeg                | Multimedia framework                                                  | https://ffmpeg.org/                                                                                     |
| FFmpeg Builds         | Pre-compiled FFmpeg binaries                                          | Windows/Linux: https://github.com/BtbN/FFmpeg-Builds, MacOS: https://github.com/eko5624/mpv-mac         |
| PyTorch               | Neural Network Inference (CUDA/ROCm/TensorRT)                        | https://pytorch.org/                                                                                   |
| NCNN                  | Neural Network Inference (Vulkan)                                      | https://github.com/tencent/ncnn                                                                        |
| RIFE                  | Real-Time Intermediate Flow Estimation                                 | https://github.com/hzwer/Practical-RIFE                                                               |
| rife-ncnn-vulkan     | RIFE implementation with NCNN and Vulkan                               | https://github.com/nihui/rife-ncnn-vulkan                                                                |
| rife ncnn vulkan python | Python bindings for RIFE NCNN Vulkan                                   | https://pypi.org/project/rife-ncnn-vulkan                                                               |
| GMFSS                 | GMFlow based Anime VFI                                              | https://github.com/98mxr/GMFSS_Fortuna                                                                 |
| GIMM                  | Motion Modeling Realistic VFI                                        | https://github.com/GSeanCDAT/GIMM-VFI                                                                    |
| ncnn python           | Python bindings for NCNN Vulkan                                        | https://pypi.org/project/ncnn                                                                          |
| Real-ESRGAN           | Upscaling                                                             | https://github.com/xinntao/Real-ESRGAN                                                                 |
| SPAN                  | Upscaling                                                             | https://github.com/hongyuanyu/SPAN                                                                     |
| Spandrel              | CUDA upscaling model architecture support                             | https://github.com/chaiNNer-org/spandrel                                                                |
| ChaiNNer              | Model Scale Detection                                                 | https://github.com/chaiNNer-org/chainner                                                                  |
| cx_Freeze             | Creates standalone executables (Linux build)                          | https://github.com/marcelotduarte/cx_Freeze                                                             |
| PyInstaller           | Creates standalone executables (Windows/Mac builds)                    | https://github.com/pyinstaller/pyinstaller                                                               |
| Feather Icons         | Open-source icons                                                     | https://github.com/feathericons/feather                                                                 |
| PySceneDetect         | Transition detection                                                  | https://github.com/Breakthrough/PySceneDetect/                                                          |
| Python Standalone Builds | Backend inference with portable python | https://github.com/indygreg/python-build-standalone |


## Star History
[![Star History Chart](https://api.star-history.com/svg?repos=tntwise/real-video-enhancer&type=Date)](https://star-history.com/#tntwise/real-video-enhancer&Date)