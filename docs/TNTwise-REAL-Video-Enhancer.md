# REAL Video Enhancer: Elevate Your Videos with AI-Powered Enhancement

Tired of blurry or low-quality videos? **REAL Video Enhancer** offers a user-friendly solution to upscale and interpolate your videos, breathing new life into your favorite content. ([Original Repository](https://github.com/TNTwise/REAL-Video-Enhancer))

[![Visitors](https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fgithub.com%2FTNTwise%2FREAL-Video-enhancer%2F&countColor=%23263759)](https://github.com/TNTwise/REAL-Video-enhancer/)
[![License](https://img.shields.io/github/license/tntwise/real-video-enhancer)](https://github.com/TNTwise/REAL-Video-Enhancer/blob/main/LICENSE)
[![Version](https://img.shields.io/badge/Version-2.3.6-blue)](https://github.com/TNTwise/REAL-Video-Enhancer/releases)
[![Downloads](https://img.shields.io/github/downloads/tntwise/REAL-Video-Enhancer/total.svg?label=Downloads@Total)](https://github.com/TNTwise/REAL-Video-Enhancer/releases)
<a href="https://discord.gg/hwGHXga8ck">
    <img src="https://img.shields.io/discord/1041502781808328704?label=Discord" alt="Discord Shield"/></a>
<br/>
<a href="https://flathub.org/apps/io.github.tntwise.REAL-Video-Enhancer">
    <img src="https://dl.flathub.org/assets/badges/flathub-badge-en.svg" height="50px"/>
</a>
<p align=center>
  <img src="https://github.com/TNTwise/REAL-Video-Enhancer/raw/2.0/icons/logo-v2.svg" width = "25%">
</p>

## Key Features:

*   **Cross-Platform Support:** Works on Windows, macOS (arm/x86), and Linux.
*   **Enhanced Performance:** Leverage TensorRT and NCNN for efficient processing on NVIDIA GPUs and Vulkan-capable devices.
*   **Frame Interpolation:** Smoothly increase frame rates (e.g., 24fps to 48fps) using cutting-edge models.
*   **Upscaling:** Improve video resolution with advanced AI models.
*   **Scene Change Detection:** Preserves sharp transitions for a better viewing experience.
*   **Discord Rich Presence:** Displays your activity within the Discord application.
*   **User-Friendly Interface:** Includes a preview to check your progress.

## Table of Contents

*   [Introduction](#introduction)
*   [Key Features](#key-features)
*   [Hardware/Software Requirements](#hardware-software-requirements)
*   [Models](#models)
    *   [Interpolate Models](#interpolate-models)
    *   [Upscale Models](#upscale-models)
*   [Backends](#backends)
*   [FAQ](#faq)
    *   [General Application Usage](#general-application-usage)
    *   [TensorRT](#tensorrt-related-questions)
    *   [ROCm](#rocm-related-questions)
    *   [NCNN](#ncnn-related-questions)
*   [Cloning](#cloning)
*   [Building](#building)
*   [Colab Notebook](#colab-notebook)
*   [Credits](#credits)
    *   [People](#people)
    *   [Software](#software)
*   [Star History](#star-history)

## Hardware/Software Requirements

| Feature          | Minimum                    | Recommended                        |
| ---------------- | -------------------------- | ---------------------------------- |
| CPU              | Dual Core x64 bit           | Quad Core x64 bit                  |
| GPU              | Vulkan 1.3 capable device  | NVIDIA RTX GPU (20 series and up) |
| VRAM             | 4 GB - NCNN                | 8 GB - TensorRT                    |
| RAM              | 16 GB                      | 32 GB                              |
| Storage          | 1 GB free - NCNN           | 16 GB free - TensorRT              |
| Operating System | Windows 10/11 64bit / MacOS 14+ | Any modern Linux distro (Ubuntu 22.04+) |

## Models

### Interpolate Models

| Model        | Author   | Link                                                       |
| ------------- | -------- | ---------------------------------------------------------- |
| RIFE 4.6,4.7,4.15,4.18,4.22,4.22-lite,4.25 | Hzwer    | [Practical-RIFE](https://github.com/hzwer/Practical-RIFE) |
| GMFSS       | 98mxr    | [GMFSS_Fortuna](https://github.com/98mxr/GMFSS_Fortuna)     |
| GIMM        | GSeanCDAT | [GIMM](https://github.com/GSeanCDAT/GIMM-VFI)               |
| IFRNet      | ltkong218 | [IFRnet](https://github.com/ltkong218/IFRNet)              |

### Upscale Models

| Model                   | Author         | Link                                                                                             |
| ----------------------- | -------------- | ------------------------------------------------------------------------------------------------ |
| 4x-SPANkendata          | Crustaceous D  | [4x-SPANkendata](https://openmodeldb.info/models/4x-SPANkendata)                                   |
| 4x-ClearRealityV1      | Kim2091        | [4x-ClearRealityV1](https://openmodeldb.info/models/4x-ClearRealityV1)                              |
| 4x-Nomos8k-SPAN series | Helaman        | [4x-Nomos8k-SPAN series](https://openmodeldb.info/models/4x-Nomos8k-span-otf-strong)                |
| 2x-OpenProteus         | SiroSky        | [OpenProteus](https://github.com/Sirosky/Upscale-Hub/releases/tag/OpenProteus)                      |
| 2x-AnimeJaNai V2 and V3 Sharp | The Database | [AnimeJanai](https://github.com/the-database/mpv-upscale-2x_animejanai)                         |
| 2x-AniSD                | SiroSky        | [AniSD](https://github.com/Sirosky/Upscale-Hub/releases/tag/AniSD)                                |

### Decompression Models

| Model    | Author  | Link                                                                |
| -------- | ------- | ------------------------------------------------------------------- |
| DeH264   | Helaman | [1xDeH264_realplksr](https://github.com/Phhofm/models/releases/tag/1xDeH264_realplksr) |

## Backends

| Backend   | Hardware                                       |
| --------- | ---------------------------------------------- |
| TensorRT  | NVIDIA RTX GPUs                                |
| PyTorch   | CUDA 12.6 and ROCm 6.2 capable GPUs          |
| NCNN      | Vulkan 1.3 capable GPUs                        |

## FAQ

### General Application Usage

| Question                                                              | Answer                                                                                                                                                                                                     |
| --------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| What does this program attempt to accomplish?                         | Fast, efficient, and easily accessible video interpolation (e.g., 24->48FPS) and video upscaling (e.g., 1920->3840).                                                                               |
| Why is it failing to recognize installed backends?                  | REAL Video Enhancer uses PIP and portable python for inference, this can sometimes have issues installing. Please attempt reinstalling the app before creating an issue. |

### TensorRT related questions

| Question                                                              | Answer                                                                                                                                                      |
| --------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Why does it take so long to begin inference?                         | TensorRT uses advanced optimization at the beginning of inference based on your device, this is only done once per resolution of video input.           |
| Why does the optimization and inference fail?                         | The most common way an optimization can fail is **Limited VRAM**. There is no fix to this except using CUDA or NCNN instead.                              |

### ROCm related questions

| Question                                                  | Answer                                                                              |
| --------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| Why am I getting (Insert Error here)?                   | ROCm is buggy, please take a look at [ROCm Help](https://github.com/TNTwise/REAL-Video-Enhancer/wiki/ROCm-Help). |

### NCNN related questions

| Question                                                              | Answer                                                                                                                                                 |
| --------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Why am I getting (Insert Vulkan Error here)?                         | This usually is an OOM (Out Of Memory) error, this can indicate a weak iGPU or very old GPU, I recommend trying out the [Colab Notebook](https://github.com/TNTwise/REAL-Video-Enhancer-Colab)  instead. |

## Cloning:

```bash
# Nightly
git clone --recurse-submodules https://github.com/TNTwise/REAL-Video-Enhancer

# Stable
git clone --recurse-submodules https://github.com/TNTwise/REAL-Video-Enhancer --branch 2.3.6
```

## Building:

3 supported build methods:

*   pyinstaller (recommended for Win/Mac)
*   cx_freeze (recommended for Linux)
*   nuitka (experimental)

supported python versions:

*   3.10 3.11, 3.12

```bash
python3 build.py --build BUILD_OPTION --copy_backend
```

## Colab Notebook

[Colab Notebook](https://github.com/tntwise/REAL-Video-Enhancer-Colab)

## Credits:

### People:

| Person             | For                                                                          | Link                                 |
| ------------------ | ---------------------------------------------------------------------------- | ------------------------------------ |
| NevermindNilas    | Some backend and reference code and working with me on many projects             | https://github.com/NevermindNilas/    |
| Styler00dollar   | RIFE ncnn models (4.1-4.5, 4.7-4.12-lite), Sudo Shuffle Span and benchmarking | https://github.com/styler00dollar    |
| HolyWu            | TensorRT engine generation code, inference optimizations, and RIFE jagged lines fixes | https://github.com/HolyWu/           |
| Rick Astley        | Amazing music                                                                 | https://www.youtube.com/watch?v=dQw4w9WgXcQ |

### Software:

| Software Used         | For                                                                         | Link                                                                                                   |
| --------------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| FFmpeg                | Multimedia framework for handling video, audio, and other media files        | https://ffmpeg.org/                                                                                     |
| FFmpeg Builds         | Pre-compiled builds of FFmpeg.                                                | Windows/Linux:  https://github.com/BtbN/FFmpeg-Builds, MacOS: https://github.com/eko5624/mpv-mac        |
| PyTorch               | Neural Network Inference (CUDA/ROCm/TensorRT)                               | https://pytorch.org/                                                                                   |
| NCNN                  | Neural Network Inference (Vulkan)                                           | https://github.com/tencent/ncnn                                                                       |
| RIFE                  | Real-Time Intermediate Flow Estimation for Video Frame Interpolation           | https://github.com/hzwer/Practical-RIFE                                                                |
| rife-ncnn-vulkan      | Video frame interpolation implementation using NCNN and Vulkan              | https://github.com/nihui/rife-ncnn-vulkan                                                             |
| rife ncnn vulkan python | Python bindings for RIFE NCNN Vulkan implementation                        | https://github.com/media2x/rife-ncnn-vulkan-python                                                       |
| GMFSS                 | GMFlow based Anime VFI                                                     | https://github.com/98mxr/GMFSS_Fortuna                                                                 |
| GIMM                  | Motion Modeling Realistic VFI                                               | https://github.com/GSeanCDAT/GIMM-VFI                                                                  |
| ncnn python           | Python bindings for NCNN Vulkan framework                                   | https://pypi.org/project/ncnn                                                                        |
| Real-ESRGAN           | Upscaling                                                                   | https://github.com/xinntao/Real-ESRGAN                                                                |
| SPAN                  | Upscaling                                                                   | https://github.com/hongyuanyu/SPAN                                                                      |
| Spandrel              | CUDA upscaling model architecture support                                   | https://github.com/chaiNNer-org/spandrel                                                              |
| ChaiNNer              | Model Scale Detection                                                         | https://github.com/chaiNNer-org/chainner                                                              |
| cx_Freeze             | Tool for creating standalone executables from Python scripts (Linux build)    | https://github.com/marcelotduarte/cx_Freeze                                                          |
| PyInstaller           | Tool for creating standalone executables from Python scripts (Windows/Mac builds) | https://github.com/pyinstaller/pyinstaller                                                            |
| Feather Icons         | Open source icons library                                                   | https://github.com/feathericons/feather                                                                |
| PySceneDetect         | Transition detection library for python                                     | https://github.com/Breakthrough/PySceneDetect/                                                          |
| Python Standalone Builds | Backend inference using portable python, helps when porting to different platforms.        | https://github.com/indygreg/python-build-standalone |

## Star History
[![Star History Chart](https://api.star-history.com/svg?repos=tntwise/real-video-enhancer&type=Date)](https://star-history.com/#tntwise/real-video-enhancer&Date)