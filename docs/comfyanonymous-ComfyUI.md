<div align="center">

# ComfyUI: Unleash Your AI Creativity with Node-Based Workflows

**ComfyUI empowers you to create stunning images, videos, and more with its flexible, node-based visual AI engine – get ready to visualize your imagination!**

[![Website][website-shield]][website-url]
[![Discord][discord-shield]][discord-url]
[![Twitter][twitter-shield]][twitter-url]
[![Matrix][matrix-shield]][matrix-url]
<br>
[![GitHub Release][github-release-shield]][github-release-link]
[![Release Date][github-release-date-shield]][github-release-link]
[![Downloads][github-downloads-shield]][github-downloads-link]
[![Latest Downloads][github-downloads-latest-shield]][github-downloads-link]

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

ComfyUI is a powerful and modular visual AI engine and application that allows users to design and execute advanced Stable Diffusion pipelines. It utilizes a graph/nodes/flowchart based interface, providing a flexible and intuitive way to experiment with and create complex AI workflows. It is available on Windows, Linux, and macOS.

## Key Features

*   **Node-Based Workflow:** Design complex Stable Diffusion pipelines without coding using an intuitive node-based interface.
*   **Model Support:** Extensive support for a wide range of image, video, and audio models, including:
    *   **Image Models:** SD1.x, SD2.x, SDXL, SDXL Turbo, Stable Cascade, SD3 and SD3.5, Pixart Alpha and Sigma, AuraFlow, HunyuanDiT, Flux, Lumina Image 2.0, HiDream, Qwen Image, and Hunyuan Image 2.1.
    *   **Image Editing Models:** Omnigen 2, Flux Kontext, HiDream E1.1, Qwen Image Edit.
    *   **Video Models:** Stable Video Diffusion, Mochi, LTX-Video, Hunyuan Video, Wan 2.1, and Wan 2.2.
    *   **Audio Models:** Stable Audio and ACE Step.
    *   **3D Models:** Hunyuan3D 2.0.
*   **Optimized Performance:**  Features asynchronous queue system, memory optimization for low-VRAM GPUs, and only re-executes modified parts of the workflow.
*   **Checkpoint and Model Loading:** Supports loading ckpt and safetensors files, including diffusion models, VAEs, and CLIP models.
*   **Advanced Features:** Includes support for embeddings/textual inversions, LoRAs, Hypernetworks, area composition, inpainting, ControlNet/T2I-Adapter, and upscale models.
*   **Workflow Management:** Load and save workflows as JSON files; load workflows from generated PNG, WebP, and FLAC files.
*   **Offline Functionality:** Works fully offline, ensuring privacy and control over your AI workflows.
*   **API Integration:** Optional API nodes available to use paid models from external providers via the Comfy API.

For workflow examples, visit the [Examples page](https://comfyanonymous.github.io/ComfyUI_examples/).

## Getting Started

Choose your preferred installation method:

*   **[Desktop Application](https://www.comfy.org/download)**: Easiest way to get started, available for Windows & macOS.
*   **[Windows Portable Package](#installing)**: Portable and up-to-date, for Windows.
*   **[Manual Install](#manual-install-windows-linux)**: Supports all operating systems and GPU types (NVIDIA, AMD, Intel, Apple Silicon, Ascend).

## Installation

### [Windows Portable](https://github.com/comfyanonymous/ComfyUI/releases/latest/download/ComfyUI_windows_portable_nvidia.7z)

1.  Download the latest portable package from the releases page.
2.  Extract the contents using 7-Zip.
3.  Place your Stable Diffusion checkpoints/models in the `ComfyUI\models\checkpoints` directory.
4.  Run the executable.

### [comfy-cli](https://docs.comfy.org/comfy-cli/getting-started)

```bash
pip install comfy-cli
comfy install
```

### Manual Install (Windows, Linux, macOS)

1.  **Clone the repository:** `git clone https://github.com/comfyanonymous/ComfyUI`
2.  **Place models:** Put your SD checkpoints (ckpt/safetensors files) in `models/checkpoints` and VAEs in `models/vae`.
3.  **Install dependencies:** Navigate to the ComfyUI directory in your terminal and run: `pip install -r requirements.txt`
4.  **Run ComfyUI:** Execute `python main.py`

Follow the OS-specific instructions for GPU setup (AMD, Intel, NVIDIA, Apple Silicon, Ascend, Cambricon, Iluvatar).

### Running

```bash
python main.py
```

## Useful Links

*   **[ComfyUI Core Repository](https://github.com/comfyanonymous/ComfyUI)**
*   **[ComfyUI Desktop Repository](https://github.com/Comfy-Org/desktop)**
*   **[ComfyUI Frontend Repository](https://github.com/Comfy-Org/ComfyUI_frontend)**
*   **[ComfyUI Examples](https://comfyanonymous.github.io/ComfyUI_examples/)**
*   **[ComfyUI Wiki - GPU Recommendations](https://github.com/comfyanonymous/ComfyUI/wiki/Which-GPU-should-I-buy-for-ComfyUI)**
*   **[ComfyUI Support and Development Channel](https://www.comfy.org/)**

**[View the original repository](https://github.com/comfyanonymous/ComfyUI)**