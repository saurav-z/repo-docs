<div align="center">

# ComfyUI: The Most Powerful and Modular Visual AI Engine

**Unleash your creativity with ComfyUI, a graph-based interface that empowers you to build and customize intricate AI image generation workflows.**

[![Website](https://img.shields.io/badge/ComfyOrg-4285F4?style=flat)](https://www.comfy.org/)
[![Discord](https://img.shields.io/badge/Discord-Join-green)](https://www.comfy.org/discord)
[![Twitter](https://img.shields.io/twitter/follow/ComfyUI)](https://x.com/ComfyUI)
[![Matrix](https://img.shields.io/badge/Matrix-Join-000000?style=flat&logo=matrix&logoColor=white)](https://app.element.io/#/room/%23comfyui_space%3Amatrix.org)
<br>
[![GitHub Release](https://img.shields.io/github/v/release/comfyanonymous/ComfyUI?style=flat&sort=semver)](https://github.com/comfyanonymous/ComfyUI/releases)
[![GitHub Release Date](https://img.shields.io/github/release-date/comfyanonymous/ComfyUI?style=flat)](https://github.com/comfyanonymous/ComfyUI/releases)
[![GitHub Downloads](https://img.shields.io/github/downloads/comfyanonymous/ComfyUI/total?style=flat)](https://github.com/comfyanonymous/ComfyUI/releases)

![ComfyUI Screenshot](https://github.com/user-attachments/assets/7ccaf2c1-9b72-41ae-9a89-5688c94b7abe)
</div>

ComfyUI offers a node-based interface for designing and executing advanced Stable Diffusion pipelines. It is available on Windows, Linux, and macOS, providing unparalleled flexibility and control over your AI image generation.

## Key Features

*   **Node-Based Workflow:** Create complex Stable Diffusion workflows using a visual, graph-based interface, eliminating the need for code.
*   **Extensive Model Support:** Works with a wide range of image, video, and audio models including:
    *   SD1.x, SD2.x, SDXL, SDXL Turbo, Stable Cascade, SD3 and SD3.5, and more.
    *   Video models like Stable Video Diffusion, Mochi, LTX-Video, and others.
    *   Audio models like Stable Audio and ACE Step.
    *   3D models (Hunyuan3D 2.0)
*   **Advanced Techniques:** Includes features like:
    *   Asynchronous Queue System for efficient processing.
    *   Smart Memory Management for running large models on GPUs with limited VRAM.
    *   Support for ckpt, safetensors, embeddings, LoRAs, hypernetworks, and more.
    *   Workflow loading/saving in various formats (PNG, WebP, FLAC, JSON).
    *   ControlNet, T2I-Adapter, Upscale Models, GLIGEN, and Model Merging.
    *   Latent previews with TAESD for high-quality previews.
*   **Offline Functionality:** Operates fully offline, ensuring privacy and control over your data.
*   **Modular and Customizable:** Offers an open-source platform with a config file (extra_model_paths.yaml.example) for model paths.

For examples of workflows, visit the [Examples page](https://comfyanonymous.github.io/ComfyUI_examples/).

## Installation

Choose the installation method that best suits your needs:

*   **Desktop Application:**
    *   The easiest way to get started.
    *   Available on Windows & macOS.
*   **Windows Portable Package:**
    *   Get the latest commits and completely portable.
    *   Available on Windows.
*   **Manual Install:**
    *   Supports all operating systems and GPU types (NVIDIA, AMD, Intel, Apple Silicon, Ascend).
    *   See [Manual Install](#manual-install-windows-linux) section for more information.

## Get Started

### [Desktop Application](https://www.comfy.org/download)
- The easiest way to get started.
- Available on Windows & macOS.

### [Windows Portable Package](#installing)
- Get the latest commits and completely portable.
- Available on Windows.

### [Manual Install](#manual-install-windows-linux)
Supports all operating systems and GPU types (NVIDIA, AMD, Intel, Apple Silicon, Ascend).

## Running

```bash
python main.py
```

## Additional Information

*   **[ComfyUI Wiki](https://github.com/comfyanonymous/ComfyUI/wiki/Which-GPU-should-I-buy-for-ComfyUI)**: For GPU recommendations.
*   **Support:** Get help and engage with the community on [Discord](https://comfy.org/discord) or [Matrix](https://app.element.io/#/room/%23comfyui_space%3Amatrix.org).

## Contributing

ComfyUI is open-source and welcomes contributions. Explore the [ComfyUI Core](https://github.com/comfyanonymous/ComfyUI) repository for code contributions. For any bugs, issues, or feature requests related to the frontend, please use the [ComfyUI Frontend repository](https://github.com/Comfy-Org/ComfyUI_frontend).

**[Original Repository](https://github.com/comfyanonymous/ComfyUI)**