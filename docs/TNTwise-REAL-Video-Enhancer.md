# REAL Video Enhancer: Elevate Your Videos with AI-Powered Enhancement

Enhance and upscale your videos with ease using **REAL Video Enhancer**, a powerful and user-friendly application for Windows, Linux, and macOS. ([View on GitHub](https://github.com/TNTwise/REAL-Video-Enhancer))

**Key Features:**

*   **Cross-Platform Support:** Works seamlessly on Windows, Linux, and macOS.
*   **Frame Interpolation:** Smoothly increases video frame rates (e.g., 24fps to 48fps).
*   **Video Upscaling:** Enhances video resolution and detail.
*   **Scene Change Detection:** Preserves sharp transitions in your videos.
*   **Multiple Backends:** Supports TensorRT, PyTorch (CUDA/ROCm), and NCNN for optimal performance across different hardware.
*   **Discord Integration:** Discord RPC support for system package and flatpak.
*   **Preview Feature:** See real-time updates of the latest rendered frame.
*   **Wide Model Support:** Compatible with a variety of interpolation, upscaling, and decompression models.

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
    *   [TensorRT Related Questions](#tensorrt-related-questions)
    *   [ROCm Related Questions](#rocm-related-questions)
    *   [NCNN Related Questions](#ncnn-related-questions)
*   [Cloning](#cloning)
*   [Building](#building)
*   [Colab Notebook](#colab-notebook)
*   [Credits](#credits)

## Hardware & Software Requirements

| Component       | Minimum                             | Recommended                          |
| :-------------- | :---------------------------------- | :----------------------------------- |
| CPU             | Dual Core x64 bit                  | Quad Core x64 bit                    |
| GPU             | Vulkan 1.3 capable device           | Nvidia RTX GPU (20 series and up)    |
| VRAM            | 4 GB - NCNN                      | 8 GB - TensorRT                    |
| RAM             | 16 GB                             | 32 GB                              |
| Storage         | 1 GB free - NCNN                   | 16 GB free - TensorRT                |
| Operating System | Windows 10/11 64bit / MacOS 14+ | Any modern Linux distro (Ubuntu 22.04+) |

## Models

### Interpolate Models

| Model            | Author     | Link                                                            |
| :--------------- | :--------- | :-------------------------------------------------------------- |
| RIFE 4.6, 4.7, 4.15, 4.18, 4.22, 4.22-lite, 4.25 | Hzwer      | [Practical-RIFE](https://github.com/hzwer/Practical-RIFE)      |
| GMFSS            | 98mxr      | [GMFSS\_Fortuna](https://github.com/98mxr/GMFSS_Fortuna)        |
| GIMM             | GSeanCDAT  | [GIMM](https://github.com/GSeanCDAT/GIMM-VFI)                  |
| IFRNet           | ltkong218  | [IFRnet](https://github.com/ltkong218/IFRNet)                   |

### Upscale Models

| Model                | Author        | Link                                                                     |
| :------------------- | :------------ | :----------------------------------------------------------------------- |
| 4x-SPANkendata       | Crustaceous D | [4x-SPANkendata](https://openmodeldb.info/models/4x-SPANkendata)            |
| 4x-ClearRealityV1   | Kim2091       | [4x-ClearRealityV1](https://openmodeldb.info/models/4x-ClearRealityV1)    |
| 4x-Nomos8k-SPAN series | Helaman       | [4x-Nomos8k-SPAN series](https://openmodeldb.info/models/4x-Nomos8k-span-otf-strong) |
| 2x-OpenProteus       | SiroSky       | [OpenProteus](https://github.com/Sirosky/Upscale-Hub/releases/tag/OpenProteus)    |
| 2x-AnimeJaNai V2 and V3 Sharp | The Database | [AnimeJanai](https://github.com/the-database/mpv-upscale-2x_animejanai) |
| 2x-AniSD             | SiroSky       | [AniSD](https://github.com/Sirosky/Upscale-Hub/releases/tag/AniSD)          |

### Decompression Models

| Model     | Author  | Link                                                      |
| :-------- | :------ | :-------------------------------------------------------- |
| DeH264    | Helaman | [1xDeH264\_realplksr](https://github.com/Phhofm/models/releases/tag/1xDeH264_realplksr) |

## Backends

| Backend   | Hardware                    |
| :-------- | :-------------------------- |
| TensorRT  | NVIDIA RTX GPUs             |
| PyTorch   | CUDA 12.6 and ROCm 6.2 capable GPUs |
| NCNN      | Vulkan 1.3 capable GPUs     |

## FAQ

### General Application Usage

| Question                                                      | Answer                                                                                                                                                         |
| :------------------------------------------------------------ | :------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| What does this program attempt to accomplish?                 | Fast, efficient and easily accessible video interpolation (Ex: 24->48FPS) and video upscaling (Ex: 1920->3840)                                                 |
| Why is it failing to recognize installed backends?           | REAL Video Enhancer uses PIP and portable python for inference; this can sometimes have issues installing. Please attempt reinstalling the app before creating an issue. |

### TensorRT Related Questions

| Question                                                              | Answer                                                                                                                                                |
| :-------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------- |
| Why does it take so long to begin inference?                         | TensorRT uses advanced optimization at the beginning of inference based on your device; this is only done once per resolution of video inputed.          |
| Why does the optimization and inference fail?                          | The most common way an optimization can fail is **Limited VRAM**. There is no fix to this except using CUDA or NCNN instead.                            |

### ROCm Related Questions

| Question                                        | Answer                                                        |
| :---------------------------------------------- | :------------------------------------------------------------ |
| Why am I getting (Insert Error here)?           | ROCM is buggy; please take a look at [ROCm Help](https://github.com/TNTwise/REAL-Video-Enhancer/wiki/ROCm-Help). |

### NCNN Related Questions

| Question                                             | Answer                                                                                                                                               |
| :--------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------- |
| Why am I getting (Insert Vulkan Error here)?          | This usually is an OOM (Out Of Memory) error, this can indicate a weak iGPU or a very old GPU. I recommend trying out the [Colab Notebook](https://github.com/TNTwise/REAL-Video-Enhancer-Colab) instead. |

## Cloning

```bash
# Nightly
git clone --recurse-submodules https://github.com/TNTwise/REAL-Video-Enhancer

# Stable
git clone --recurse-submodules https://github.com/TNTwise/REAL-Video-Enhancer --branch 2.3.4
```

## Building

<p>Supported build methods: </p>
<ul>
    <li>pyinstaller (recommended for Win/Mac)</li>
    <li>cx_freeze (recommended for Linux)</li>
    <li>nuitka (experimental)</li>
</ul>
<p>Supported python versions: </p>
<ul>
    <li>3.10, 3.11, 3.12</li>
</ul>

```bash
python3 build.py --build BUILD_OPTION --copy_backend
```

## Colab Notebook

[Colab Notebook](https://github.com/tntwise/REAL-Video-Enhancer-Colab)

## Credits

### People

| Person            | For                                                                               | Link                                                |
| :---------------- | :-------------------------------------------------------------------------------- | :-------------------------------------------------- |
| NevermindNilas    | Some backend and reference code and working with me on many projects               | [NevermindNilas' GitHub](https://github.com/NevermindNilas/)    |
| Styler00dollar    | RIFE ncnn models (4.1-4.5, 4.7-4.12-lite), Sudo Shuffle Span and benchmarking | [Styler00dollar's GitHub](https://github.com/styler00dollar)    |
| HolyWu            | TensorRT engine generation code, inference optimizations, and RIFE jagged lines fixes  | [HolyWu's GitHub](https://github.com/HolyWu/)  |
| Rick Astley       | Amazing music                                                                      | [Rick Astley on YouTube](https://www.youtube.com/watch?v=dQw4w9WgXcQ) |

### Software

| Software Used                   | For                                                                    | Link                                                                      |
| :------------------------------ | :--------------------------------------------------------------------- | :------------------------------------------------------------------------ |
| FFmpeg                          | Multimedia framework for handling video, audio, and other media files   | [FFmpeg](https://ffmpeg.org/)                                            |
| FFMpeg Builds                   | Pre-compiled builds of FFMpeg.                                          | Windows/Linux: [FFmpeg-Builds](https://github.com/BtbN/FFmpeg-Builds), MacOS: [mpv-mac](https://github.com/eko5624/mpv-mac) |
| PyTorch                         | Neural Network Inference (CUDA/ROCm/TensorRT)                          | [PyTorch](https://pytorch.org/)                                          |
| NCNN                            | Neural Network Inference (Vulkan)                                      | [NCNN](https://github.com/tencent/ncnn)                                   |
| RIFE                            | Real-Time Intermediate Flow Estimation for Video Frame Interpolation    | [Practical-RIFE](https://github.com/hzwer/Practical-RIFE)                      |
| rife-ncnn-vulkan                | Video frame interpolation implementation using NCNN and Vulkan           | [rife-ncnn-vulkan](https://github.com/nihui/rife-ncnn-vulkan)            |
| rife ncnn vulkan python         | Python bindings for RIFE NCNN Vulkan implementation                    | [rife-ncnn-vulkan-python](https://pypi.org/project/ncnn)             |
| GMFSS                           | GMFlow based Anime VFI                                                 | [GMFSS_Fortuna](https://github.com/98mxr/GMFSS_Fortuna)                |
| GIMM                            | Motion Modeling Realistic VFI                                          | [GIMM](https://github.com/GSeanCDAT/GIMM-VFI)                            |
| ncnn python                     | Python bindings for NCNN Vulkan framework                            | [ncnn](https://pypi.org/project/ncnn)                                        |
| Real-ESRGAN                     | Upscaling                                                              | [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)                     |
| SPAN                            | Upscaling                                                              | [SPAN](https://github.com/hongyuanyu/SPAN)                                 |
| Spandrel                        | CUDA upscaling model architecture support                            | [Spandrel](https://github.com/chaiNNer-org/spandrel)                      |
| ChaiNNer                        | Model Scale Detection                                                  | [ChaiNNer](https://github.com/chaiNNer-org/chainner)                         |
| cx_Freeze                       | Tool for creating standalone executables from Python scripts (Linux build) | [cx_Freeze](https://github.com/marcelotduarte/cx_Freeze)                  |
| PyInstaller                     | Tool for creating standalone executables from Python scripts (Windows/Mac builds) | [PyInstaller](https://github.com/pyinstaller/pyinstaller)                |
| Feather Icons                   | Open source icons library                                              | [Feather Icons](https://github.com/feathericons/feather)                 |
| PySceneDetect                   | Transition detection library for python                                | [PySceneDetect](https://github.com/Breakthrough/PySceneDetect/)          |
| Python Standalone Builds        | Backend inference using portable python, helps when porting to different platforms.     | [python-build-standalone](https://github.com/indygreg/python-build-standalone) |

---

<!-- Add a star history chart here if desired -->