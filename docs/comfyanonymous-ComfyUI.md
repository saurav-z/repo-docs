<div align="center">

# ComfyUI: The Ultimate Visual AI Engine for Stable Diffusion

**Unleash your creativity with ComfyUI, a powerful and modular visual AI engine that allows you to design and execute complex Stable Diffusion workflows with ease.**  Explore the original [ComfyUI repository](https://github.com/comfyanonymous/ComfyUI) for more information.

[![Website][website-shield]][website-url]
[![Discord][discord-shield]][discord-url]
[![Twitter][twitter-shield]][twitter-url]
[![Matrix][matrix-shield]][matrix-url]
<br>
[![GitHub Release][github-release-shield]][github-release-link]
[![GitHub Release Date][github-release-date-shield]][github-release-link]
[![GitHub Downloads][github-downloads-shield]][github-downloads-link]
[![GitHub Downloads Latest][github-downloads-latest-shield]][github-downloads-link]

[matrix-shield]: https://img.shields.io/badge/Matrix-000000?style=flat&logo=matrix&logoColor=white
[matrix-url]: https://app.element.io/#/room/%23comfyui_space%3Amatrix.org
[website-shield]: https://img.shields.io/badge/ComfyOrg-4285F4?style=flat
[website-url]: https://www.comfy.org/
[discord-shield]: https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fdiscord.com%2Fapi%2Finvites%2Fcomfyorg%3Fwith_counts%3Dtrue&query=%24.approximate_member_count&logo=discord&logoColor=white&label=Discord&color=green&suffix=%20total
[discord-url]: https://www.comfy.org/discord
[twitter-shield]: https://img.shields.io/twitter/follow/ComfyUI
[twitter-url]: https://x.com/ComfyUI

[github-release-shield]: https://img.shields.io/github/v/release/comfyanonymous/ComfyUI?style=flat&sort=semver
[github-release-link]: https://github.com/comfyanonymous/ComfyUI/releases
[github-release-date-shield]: https://img.shields.io/github/release-date/comfyanonymous/ComfyUI?style=flat
[github-downloads-shield]: https://img.shields.io/github/downloads/comfyanonymous/ComfyUI/total?style=flat
[github-downloads-latest-shield]: https://img.shields.io/github/downloads/comfyanonymous/ComfyUI/latest/total?style=flat&label=downloads%40latest
[github-downloads-link]: https://github.com/comfyanonymous/ComfyUI/releases

![ComfyUI Screenshot](https://github.com/user-attachments/assets/7ccaf2c1-9b72-41ae-9a89-5688c94b7abe)
</div>

## Key Features

*   **Node-Based Workflow:**  Build complex Stable Diffusion pipelines with a visual, node-based interface, eliminating the need for coding.
*   **Extensive Model Support:**  Compatible with a vast range of image, video, audio and 3D models including:
    *   SD1.x, SD2.x, SDXL, SDXL Turbo, Stable Cascade, SD3/3.5 and more.
    *   Omnigen 2, Flux Kontext, HiDream E1.1 for image editing.
    *   Stable Video Diffusion, Mochi, LTX-Video, Nvidia Cosmos, and others for video.
    *   Stable Audio and ACE Step for audio generation.
    *   Hunyuan3D 2.0 for 3D content creation.
*   **Asynchronous Queue:** Efficiently manage and execute multiple generation tasks.
*   **Optimized Performance:** Leverage optimizations like asynchronous queuing and smart memory management.
*   **Broad Hardware Support:** Runs on Windows, Linux, and macOS, and supports NVIDIA, AMD, Intel, and Apple Silicon GPUs, or even CPU-only mode.
*   **Workflow Management:** Load, save, and share complete workflows via PNG, WebP, FLAC and JSON files.
*   **Advanced Features:** Utilize advanced techniques like ControlNet, T2I-Adapter, model merging, LCM models, and more.

## Getting Started

*   **Desktop Application:** The easiest way to get started, available for Windows and macOS. [Download](https://www.comfy.org/download)
*   **Windows Portable Package:** Get the latest commits with a fully portable package.
*   **Manual Install:** Supports all operating systems and GPU types.

## Examples

Explore the possibilities with example workflows. [Examples](https://comfyanonymous.github.io/ComfyUI_examples/)

## Installation

### [Windows Portable](#windows-portable)

*   Download, extract, and run from the releases page.
*   Place your Stable Diffusion checkpoints/models in `ComfyUI\models\checkpoints`.

### [comfy-cli](https://docs.comfy.org/comfy-cli/getting-started)

```bash
pip install comfy-cli
comfy install
```

### [Manual Install (Windows, Linux)](#manual-install-windows-linux)

1.  Clone the repository.
2.  Place your models (ckpt/safetensors, VAE, etc.) in the designated `models` subfolders.
3.  Install dependencies using `pip install -r requirements.txt`.
4.  Run `python main.py`.

#### GPU-Specific Instructions

*   **AMD GPUs (Linux):** Install ROCm and PyTorch with the commands.
*   **Intel GPUs (Windows and Linux):**  Install PyTorch with XPU or use IPEX for Intel Extension for PyTorch (IPEX).
*   **NVIDIA:** Install stable or nightly PyTorch versions with CUDA support.
*   **Apple Mac silicon:** Install Pytorch nightly and follow the ComfyUI installation instructions.

## Running

Run ComfyUI with the command:

```bash
python main.py
```

### Advanced Options

*   AMD: HSA_OVERRIDE_GFX_VERSION and experimental memory efficient attention.

## Useful Notes
*   Only parts of the graph that have an output with all the correct inputs will be executed.
*   If you submit the same graph twice only the first will be executed.
*   Dragging a generated png on the webpage or loading one will give you the full workflow including seeds that were used to create it.
*   You can use () to change emphasis of a word or phrase like: (good code:1.2) or (bad code:0.8). The default emphasis for () is 1.1. To use () characters in your actual prompt escape them like \\( or \\).
*   You can use {day|night}, for wildcard/dynamic prompts. With this syntax "{wild|card|test}" will be randomly replaced by either "wild", "card" or "test" by the frontend every time you queue the prompt. To use {} characters in your actual prompt escape them like: \\{ or \\}.
*   Dynamic prompts also support C-style comments, like `// comment` or `/* comment */`.
*   To use a textual inversion concepts/embeddings in a text prompt put them in the models/embeddings directory and use them in the CLIPTextEncode node like this (you can omit the .pt extension):
```embedding:embedding_filename.pt```

## High-Quality Previews
Enable high-quality previews with TAESD.  Download taesd_decoder.pth, taesdxl_decoder.pth, taesd3_decoder.pth and taef1_decoder.pth from the [TAESD Github](https://github.com/madebyollin/taesd/) and place them in the `models/vae_approx` folder.  Restart ComfyUI and run with `--preview-method taesd`.

## Support and Community

*   [Discord](https://comfy.org/discord): Get help and discuss ComfyUI.
*   [Matrix Space](https://app.element.io/#/room/%23comfyui_space%3Amatrix.org): Another open-source community channel.
*   [Website](https://www.comfy.org/): For more resources.

## Frontend Development

The ComfyUI frontend is now in a separate repository [ComfyUI Frontend](https://github.com/Comfy-Org/ComfyUI_frontend). 

*   Use the [ComfyUI Frontend repository](https://github.com/Comfy-Org/ComfyUI_frontend) for frontend related issues and feature requests.
*   Use the main ComfyUI repo with the `--front-end-version` command to use the latest daily frontend builds.

## QA

See the [wiki page](https://github.com/comfyanonymous/ComfyUI/wiki/Which-GPU-should-I-buy-for-ComfyUI) for some GPU recommendations.