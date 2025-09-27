# REAL Video Enhancer: Elevate Your Videos with AI-Powered Enhancement

**Transform your videos into stunning, high-quality content with REAL Video Enhancer, a versatile application for frame interpolation and upscaling.**  ([Original Repo](https://github.com/TNTwise/REAL-Video-Enhancer))

[![Visitors](https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fgithub.com%2FTNTwise%2FREAL-Video-enhancer%2F&countColor=%23263759)](https://github.com/TNTwise/REAL-Video-enhancer/)
[![pypresence](https://img.shields.io/badge/using-pypresence-00bb88.svg?style=for-the-badge&logo=discord&logoWidth=20)](https://github.com/qwertyquerty/pypresence)
[![license](https://img.shields.io/github/license/tntwise/real-video-enhancer)](https://github.com/TNTwise/REAL-Video-Enhancer/blob/main/LICENSE)
[![Version](https://img.shields.io/badge/Version-2.3.6-blue)](https://github.com/TNTwise/REAL-Video-Enhancer/releases)
[![downloads_total](https://img.shields.io/github/downloads/tntwise/REAL-Video-Enhancer/total.svg?label=downloads%40total)](https://github.com/TNTwise/REAL-Video-Enhancer/releases)
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

*   **Cross-Platform Support:** Works seamlessly on Windows, Linux (including Flatpak), and MacOS (arm/x86).
*   **Discord Integration:** Supports Discord Rich Presence to showcase your enhancement activities.
*   **Intelligent Scene Detection:** Preserves sharp transitions by detecting and processing scene changes accurately.
*   **Real-time Preview:** View the latest rendered frame for instant feedback.
*   **Optimized Backends:** Leverages TensorRT and NCNN for efficient performance on NVIDIA and Vulkan-capable GPUs.

## Table of Contents

*   [Introduction](#introduction)
*   [Features](#key-features)
*   [Hardware Requirements](#hardware-requirements)
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
    *   [People](#people)
    *   [Software](#software)

## Introduction

REAL Video Enhancer is a powerful and user-friendly application, offering frame interpolation (e.g., 24fps to 48fps) and upscaling capabilities. It's a great alternative to outdated software like Flowframes or enhancr, providing convenient access to cutting-edge video enhancement technologies across multiple platforms.

<p align=center>
  <img src="https://github.com/TNTwise/REAL-Video-Enhancer/blob/v2-main/screenshots/demo.png?raw=true" width = "100%">
</p>

## Hardware Requirements

| Feature        | Minimum                    | Recommended                      |
| -------------- | -------------------------- | -------------------------------- |
| CPU            | Dual Core x64 bit          | Quad Core x64 bit                |
| GPU            | Vulkan 1.3 capable device  | Nvidia RTX GPU (20 series and up) |
| VRAM           | 4 GB - NCNN                | 8 GB - TensorRT                  |
| RAM            | 16 GB                      | 32 GB                            |
| Storage        | 1 GB free - NCNN           | 16 GB free - TensorRT            |
| Operating System | Windows 10/11 64bit / MacOS 14+ | Any modern Linux distro (Ubuntu 22.04+) |

## Models

### Interpolate Models

| Model          | Author     | Link                                                       |
| -------------- | ---------- | ---------------------------------------------------------- |
| RIFE 4.6,4.7,4.15,4.18,4.22,4.22-lite,4.25 | Hzwer      | [Practical-RIFE](https://github.com/hzwer/Practical-RIFE) |
| GMFSS          | 98mxr      | [GMFSS_Fortuna](https://github.com/98mxr/GMFSS_Fortuna)   |
| GIMM           | GSeanCDAT  | [GIMM](https://github.com/GSeanCDAT/GIMM-VFI)             |
| IFRNet         | ltkong218  | [IFRnet](https://github.com/ltkong218/IFRNet)              |

### Upscale Models

| Model                  | Author          | Link                                                                               |
| ---------------------- | --------------- | ---------------------------------------------------------------------------------- |
| 4x-SPANkendata         | Crustaceous D   | [4x-SPANkendata](https://openmodeldb.info/models/4x-SPANkendata)                 |
| 4x-ClearRealityV1      | Kim2091         | [4x-ClearRealityV1](https://openmodeldb.info/models/4x-ClearRealityV1)             |
| 4x-Nomos8k-SPAN series | Helaman         | [4x-Nomos8k-SPAN series](https://openmodeldb.info/models/4x-nomos8k-span-otf-strong) |
| 2x-OpenProteus         | SiroSky         | [OpenProteus](https://github.com/Sirosky/Upscale-Hub/releases/tag/OpenProteus)      |
| 2x-AnimeJaNai V2 and V3 Sharp | The Database  | [AnimeJanai](https://github.com/the-database/mpv-upscale-2x_animejanai)           |
| 2x-AniSD               | SiroSky         | [AniSD](https://github.com/Sirosky/Upscale-Hub/releases/tag/AniSD)                   |

### Decompression Models

| Model     | Author  | Link                                                                   |
| --------- | ------- | ---------------------------------------------------------------------- |
| DeH264    | Helaman | [1xDeH264_realplksr](https://github.com/Phhofm/models/releases/tag/1xDeH264_realplksr) |

## Backends

| Backend   | Hardware                   |
| --------- | -------------------------- |
| TensorRT  | NVIDIA RTX GPUs            |
| PyTorch   | CUDA 12.6 and ROCm 6.2 capable GPUs |
| NCNN      | Vulkan 1.3 capable GPUs    |

## FAQ

### General Application Usage

| Question                                       | Answer                                                                                                                                     |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| What does this program attempt to accomplish?   | Fast, efficient, and easily accessible video interpolation (Ex: 24->48FPS) and video upscaling (Ex: 1920->3840).                         |
| Why is it failing to recognize installed backends? | REAL Video Enhancer uses PIP and portable python for inference, which can sometimes have installation issues. Please try reinstalling. |

### TensorRT Related Questions

| Question                                    | Answer                                                                                                                               |
| ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| Why does it take so long to begin inference? | TensorRT uses advanced optimization at the beginning of inference based on your device; this is only done once per input video resolution. |
| Why does the optimization and inference fail? | The most common cause of optimization failure is **Limited VRAM**. There's no solution to this except using CUDA or NCNN instead.    |

### ROCm Related Questions

| Question                                   | Answer                                                                                                                                                                       |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Why am I getting (Insert Error here)? | ROCm is buggy, please take a look at [ROCm Help](https://github.com/TNTwise/REAL-Video-Enhancer/wiki/ROCm-Help). |

### NCNN Related Questions

| Question                                        | Answer                                                                                                                                                                               |
| ----------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Why am I getting (Insert Vulkan Error here)? | This usually indicates an OOM (Out Of Memory) error, which can be caused by a weak iGPU or very old GPU. Consider trying the [Colab Notebook](https://github.com/TNTwise/REAL-Video-Enhancer-Colab) instead. |

## Cloning

```bash
# Nightly
git clone --recurse-submodules https://github.com/TNTwise/REAL-Video-Enhancer

# Stable
git clone --recurse-submodules https://github.com/TNTwise/REAL-Video-Enhancer --branch 2.3.6
```

## Building

Three supported build methods are available:

*   **pyinstaller** (Recommended for Windows/Mac)
*   **cx\_freeze** (Recommended for Linux)
*   **nuitka** (Experimental)

Supported Python versions: 3.10, 3.11, and 3.12

```bash
python3 build.py --build BUILD_OPTION --copy_backend
```

## Colab Notebook

[Colab Notebook](https://github.com/tntwise/REAL-Video-Enhancer-Colab)

## Credits

### People

| Person            | For                                                                | Link                                                 |
| ----------------- | ------------------------------------------------------------------ | ---------------------------------------------------- |
| NevermindNilas    | Backend, reference code, and collaboration on multiple projects  | [NevermindNilas](https://github.com/NevermindNilas/)    |
| Styler00dollar    | RIFE ncnn models (4.1-4.5, 4.7-4.12-lite), Sudo Shuffle Span, and benchmarking | [styler00dollar](https://github.com/styler00dollar)    |
| HolyWu            | TensorRT engine generation code, inference optimizations, and RIFE jagged lines fixes | [HolyWu](https://github.com/HolyWu/)              |
| Rick Astley       | Amazing music                                                      | [Rick Astley - Never Gonna Give You Up](https://www.youtube.com/watch?v=dQw4w9WgXcQ) |

### Software

| Software Used                       | For                                                                          | Link                                                                      |
| ----------------------------------- | ---------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| FFmpeg                              | Multimedia framework for handling video, audio, and other media files        | [FFmpeg](https://ffmpeg.org/)                                               |
| FFMpeg Builds                     | Pre-compiled builds of FFMpeg.                                                | Windows/Linux: [BtbN/FFmpeg-Builds](https://github.com/BtbN/FFmpeg-Builds), MacOS: [eko5624/mpv-mac](https://github.com/eko5624/mpv-mac)      |
| PyTorch                             | Neural Network Inference (CUDA/ROCm/TensorRT)                                 | [PyTorch](https://pytorch.org/)                                               |
| NCNN                                | Neural Network Inference (Vulkan)                                            | [NCNN](https://github.com/tencent/ncnn)                                    |
| RIFE                                | Real-Time Intermediate Flow Estimation for Video Frame Interpolation           | [Practical-RIFE](https://github.com/hzwer/Practical-RIFE)                   |
| rife-ncnn-vulkan                    | Video frame interpolation implementation using NCNN and Vulkan                  | [rife-ncnn-vulkan](https://github.com/nihui/rife-ncnn-vulkan)               |
| rife ncnn vulkan python             | Python bindings for RIFE NCNN Vulkan implementation                           | [rife-ncnn-vulkan-python](https://github.com/media2x/rife-ncnn-vulkan-python) |
| GMFSS                               | GMFlow based Anime VFI                                                       | [GMFSS_Fortuna](https://github.com/98mxr/GMFSS_Fortuna)                    |
| GIMM                                | Motion Modeling Realistic VFI                                                  | [GIMM](https://github.com/GSeanCDAT/GIMM-VFI)                              |
| ncnn python                         | Python bindings for NCNN Vulkan framework                                      | [ncnn](https://pypi.org/project/ncnn)                                        |
| Real-ESRGAN                         | Upscaling                                                                    | [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)                        |
| SPAN                                | Upscaling                                                                    | [SPAN](https://github.com/hongyuanyu/SPAN)                                  |
| Spandrel                            | CUDA upscaling model architecture support                                    | [Spandrel](https://github.com/chaiNNer-org/spandrel)                       |
| ChaiNNer                            | Model Scale Detection                                                         | [ChaiNNer](https://github.com/chaiNNer-org/chainner)                        |
| cx\_Freeze                          | Tool for creating standalone executables from Python scripts (Linux build)    | [cx_Freeze](https://github.com/marcelotduarte/cx_Freeze)                     |
| PyInstaller                         | Tool for creating standalone executables from Python scripts (Windows/Mac builds) | [PyInstaller](https://github.com/pyinstaller/pyinstaller)                 |
| Feather Icons                       | Open source icons library                                                    | [Feather Icons](https://github.com/feathericons/feather)                     |
| PySceneDetect                       | Transition detection library for python                                       | [PySceneDetect](https://github.com/Breakthrough/PySceneDetect/)               |
| Python Standalone Builds            | Backend inference using portable python, helps when porting to different platforms. | [python-build-standalone](https://github.com/indygreg/python-build-standalone) |

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=tntwise/real-video-enhancer&type=Date)](https://star-history.com/#tntwise/real-video-enhancer&Date)