<div align="center">

# ComfyUI: Unleash the Power of Visual AI

**ComfyUI is a powerful, modular, and open-source visual AI engine, enabling you to design and execute complex Stable Diffusion workflows with ease.**

[![Website][website-shield]][website-url]
[![Discord][discord-shield]][discord-url]
[![Twitter][twitter-shield]][twitter-url]
[![Matrix][matrix-shield]][matrix-url]
<br>
[![GitHub Release][github-release-shield]][github-release-link]
[![Release Date][github-release-date-shield]][github-release-link]
[![Downloads][github-downloads-shield]][github-downloads-link]
[![Downloads Latest][github-downloads-latest-shield]][github-downloads-link]

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

ComfyUI provides a node-based interface for creating and experimenting with Stable Diffusion pipelines, offering unparalleled flexibility and control.  [**Explore the ComfyUI repository on GitHub**](https://github.com/comfyanonymous/ComfyUI).

## Key Features

*   **Node-Based Workflow:** Design and execute complex Stable Diffusion workflows visually, without the need for code.
*   **Extensive Model Support:** Compatibility with a wide range of models, including SD1.x, SD2.x, SDXL, Stable Cascade, SD3/3.5, and more.
*   **Image Editing Capabilities:** Supports advanced image editing models like Omnigen 2, Flux Kontext, and Qwen Image Edit.
*   **Video Generation:** Offers video models like Stable Video Diffusion, Mochi, LTX-Video, and more.
*   **Audio Generation:** Integrated support for audio models such as Stable Audio and ACE Step.
*   **3D Model Support:** Includes support for Hunyuan3D 2.0.
*   **Asynchronous Queue System:** Efficiently manages and processes multiple generation tasks.
*   **Memory Optimization:** Smart memory management allows you to run large models even on GPUs with limited VRAM.
*   **CPU Support:** Works even without a GPU using the `--cpu` flag.
*   **Checkpoint and Model Loading:** Loads ckpt, safetensors, and other model formats.
*   **Workflow Management:** Load, save, and share workflows as JSON files, and supports loading workflows from PNG, WebP, and FLAC files.
*   **Customization:** Supports embeddings, LoRAs, Hypernetworks, ControlNet, T2I-Adapter, and Upscale Models.
*   **Offline Functionality:** Core functionality works entirely offline, ensuring privacy and control over your data.
*   **API Integration (Optional):**  Offers optional API nodes for integrating with paid models from external providers.

## Getting Started

### Installation

*   **Desktop Application:** The easiest way to get started. Available for Windows & macOS. Download from [ComfyUI Website](https://www.comfy.org/download).
*   **Windows Portable Package:** Get the latest commits and completely portable.  Available on Windows from the [Releases Page](https://github.com/comfyanonymous/ComfyUI/releases).
*   **Manual Install:** Supports all operating systems (Windows, Linux, macOS) and GPU types (NVIDIA, AMD, Intel, Apple Silicon, Ascend). See the instructions below for manual installation.
    *   **[comfy-cli](https://docs.comfy.org/comfy-cli/getting-started)**
    ```bash
    pip install comfy-cli
    comfy install
    ```
    *   **Manual Install (Windows, Linux)**
        1.  Clone the repository:
            ```bash
            git clone https://github.com/comfyanonymous/ComfyUI
            cd ComfyUI
            ```
        2.  Place your Stable Diffusion checkpoints (ckpt/safetensors) in the `models/checkpoints` directory.
        3.  Place your VAE files in the `models/vae` directory.
        4.  Install dependencies:
            ```bash
            pip install -r requirements.txt
            ```
        5.  Run ComfyUI:
            ```bash
            python main.py
            ```
*   **AMD GPUs (Linux only)**:  Install PyTorch with ROCm: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.4` (for stable) or `--pre` for nightly.
*   **Intel GPUs (Windows and Linux)**: Install PyTorch xpu: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu` (for stable) or `--pre` for nightly. Alternatively leverage IPEX.
*   **NVIDIA**: Install PyTorch with CUDA:  `pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu129` (for stable) or `--pre` for nightly.
*   **Apple Mac silicon**: Install PyTorch nightly as per the [Accelerated PyTorch training on Mac](https://developer.apple.com/metal/pytorch/) guide and then follow the [Manual Install instructions](#manual-install-windows-linux).

### Examples

Explore example workflows to jumpstart your creativity: [ComfyUI Examples](https://comfyanonymous.github.io/ComfyUI_examples/)

## Additional Resources

*   **Support and Community:** [Discord Server](https://www.comfy.org/discord), [Matrix Space](https://app.element.io/#/room/%23comfyui_space%3Amatrix.org), [ComfyUI Website](https://www.comfy.org/)
*   **Frontend Development:** For frontend-related issues, use the [ComfyUI Frontend repository](https://github.com/Comfy-Org/ComfyUI_frontend).

This improved README offers a concise overview, focuses on key SEO terms, and includes clear headings and bullet points for readability.