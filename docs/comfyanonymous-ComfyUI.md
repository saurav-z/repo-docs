<div align="center">

# ComfyUI: The Powerful, Modular Visual AI Engine

**Design and execute advanced Stable Diffusion pipelines with an intuitive node-based interface.**

[![Website][website-shield]][website-url]
[![Discord][discord-shield]][discord-url]
[![Twitter][twitter-shield]][twitter-url]
[![Matrix][matrix-shield]][matrix-url]
<br>
[![GitHub Release][github-release-shield]][github-release-link]
[![Release Date][github-release-date-shield]][github-release-link]
[![Downloads][github-downloads-shield]][github-downloads-link]
[![Latest Download][github-downloads-latest-shield]][github-downloads-link]

[matrix-shield]: https://img.shields.io/badge/Matrix-000000?style=flat&logo=matrix&logoColor=white
[matrix-url]: https://app.element.io/#/room/%23comfyui_space%3Amatrix.org
[website-shield]: https://img.shields.io/badge/ComfyOrg-4285F4?style=flat
[website-url]: https://www.comfy.org/
<!-- Workaround to display total user from https://github.com/badges/shields/issues/4500#issuecomment-2060079995 -->
[discord-shield]: https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fdiscord.com%2Fapi%2Finvites%2Fcomfyorg%3Fwith_counts%3Dtrue&query=%24.approximate_member_count&logo=discord&logoColor=white&label=Discord&color=green&suffix=%20total
[discord-url]: https://www.comfy.org/discord
[twitter-shield]: https://img.shields.io/twitter/follow/ComfyUI
[twitter-url]: https://x.com/ComfyUI

[github-release-shield]: https://img.shields.io/github/v/release/comfyanonymous/ComfyUI?style=flat&sort=semver
[github-release-link]: https://github.com/comfyanonymous/ComfyUI/releases
[github-release-date-shield]: https://img.shields.io/github/release-date/comfyanonymous/ComfyUI?style=flat
[github-downloads-shield]: https://img.shields/github/downloads/comfyanonymous/ComfyUI/total?style=flat
[github-downloads-latest-shield]: https://img.shields.io/github/downloads/comfyanonymous/ComfyUI/latest/total?style=flat&label=downloads%40latest
[github-downloads-link]: https://github.com/comfyanonymous/ComfyUI/releases

![ComfyUI Screenshot](https://github.com/user-attachments/assets/7ccaf2c1-9b72-41ae-9a89-5688c94b7abe)
</div>

ComfyUI is a powerful and flexible visual AI engine that allows users to build and run advanced Stable Diffusion workflows with a node-based interface. **[Visit the original repository](https://github.com/comfyanonymous/ComfyUI) for more details.**

## Key Features

*   **Node-Based Workflow:** Create complex Stable Diffusion pipelines visually without coding, using a drag-and-drop interface.
*   **Extensive Model Support:**
    *   Supports various Stable Diffusion models: SD1.x, SD2.x, SDXL, SDXL Turbo, Stable Cascade, SD3 and SD3.5, etc.
    *   Supports Image Editing Models: Omnigen 2, Flux Kontext, HiDream E1.1, Qwen Image Edit, etc.
    *   Supports Video Models: Stable Video Diffusion, Mochi, LTX-Video, etc.
    *   Supports Audio Models: Stable Audio, ACE Step
    *   Supports 3D Models: Hunyuan3D 2.0
*   **Optimized Performance:**
    *   Asynchronous queue system for efficient task management.
    *   Optimized execution: only re-executes changed parts of workflows.
    *   Smart memory management for running large models on GPUs with limited VRAM.
    *   CPU fallback option (`--cpu`) for users without GPUs.
*   **Model Compatibility:** Loads various checkpoint formats (ckpt, safetensors), VAEs, CLIP models, embeddings, LoRAs (regular, locon, loha), and Hypernetworks.
*   **Workflow Management:**
    *   Loads and saves complete workflows from PNG, WebP, and FLAC files, preserving seeds.
    *   Saves and loads workflows as JSON files.
*   **Advanced Features:**
    *   Area Composition for intricate image creation.
    *   Inpainting capabilities.
    *   ControlNet and T2I-Adapter support.
    *   Upscale models (ESRGAN, SwinIR, etc.).
    *   GLIGEN support.
    *   Model merging.
    *   LCM models and Loras.
    *   High-quality previews with TAESD.
    *   Fully offline functionality.
    *   Optional API nodes for paid models (via [Comfy API](https://docs.comfy.org/tutorials/api-nodes/overview)).
*   **Customization:**
    *   Config file (`extra_model_paths.yaml.example`) for setting model search paths.

## Getting Started

Choose from several options to start creating with ComfyUI:

*   **Desktop Application:** The easiest way to get started, available for Windows and macOS. ([Download](https://www.comfy.org/download))
*   **Windows Portable Package:** Get the latest commits, completely portable, available on Windows.
*   **Manual Install:** Supports all operating systems and GPU types.

## Examples

Explore what's possible with ComfyUI by viewing the [example workflows](https://comfyanonymous.github.io/ComfyUI_examples/).

## Release Process

ComfyUI follows a weekly release cycle, with three interconnected repositories:

1.  **[ComfyUI Core](https://github.com/comfyanonymous/ComfyUI)**: New stable releases.
2.  **[ComfyUI Desktop](https://github.com/Comfy-Org/desktop)**: Builds releases using the latest core version.
3.  **[ComfyUI Frontend](https://github.com/Comfy-Org/ComfyUI_frontend)**: Handles UI updates.

## Useful Keyboard Shortcuts

See the original README for a comprehensive list of keyboard shortcuts.

## Installing

### Windows Portable

Download, extract with [7-Zip](https://7-zip.org), and run. Place Stable Diffusion checkpoints/models in: `ComfyUI\models\checkpoints`.

#### [Direct link to download](https://github.com/comfyanonymous/ComfyUI/releases/latest/download/ComfyUI_windows_portable_nvidia.7z)

### [comfy-cli](https://docs.comfy.org/comfy-cli/getting-started)

Install with `pip install comfy-cli` and run with `comfy install`.

### Manual Install (Windows, Linux, macOS)

1.  Clone the repository.
2.  Place SD checkpoints in `models/checkpoints` and VAEs in `models/vae`.
3.  Install dependencies: `pip install -r requirements.txt`
4.  Run: `python main.py`

### GPU-Specific Instructions

*   **AMD GPUs (Linux):** Install rocm and pytorch with pip using the commands provided.
*   **Intel GPUs (Windows and Linux):** Follow the installation instructions for either the Intel Extension for PyTorch (IPEX) or PyTorch xpu for improved performance.
*   **NVIDIA:** Install stable pytorch using the provided command.
*   **macOS (M1/M2):** Install the latest PyTorch nightly and follow the manual installation steps.

### Other Installation Methods

*   DirectML (AMD Cards on Windows): Install torch-directml and run with `--directml`.
*   Ascend NPUs: Follow the instructions for your platform.
*   Cambricon MLUs: Follow the instructions for your platform.
*   Iluvatar Corex: Follow the installation instructions for your platform.

## Running

Run ComfyUI with the command `python main.py`.

### Troubleshooting

*   For AMD cards not officially supported by ROCm: use the command as given in the main README.
*   AMD ROCm Tips: Use the command as given in the main README to enable experimental features.

## How to show high-quality previews?

Use the command `--preview-method auto` to enable previews.  To use higher quality previews, download the `taesd_decoder.pth, taesdxl_decoder.pth, taesd3_decoder.pth and taef1_decoder.pth` files and place them in `models/vae_approx` folder, then run with `--preview-method taesd`.

## How to use TLS/SSL?

Generate a self-signed certificate and key and use `--tls-keyfile key.pem --tls-certfile cert.pem` to enable TLS/SSL.

## Support and Dev Channel

*   [Discord](https://comfy.org/discord): Try the #help or #feedback channels.
*   [Matrix space: #comfyui_space:matrix.org](https://app.element.io/#/room/%23comfyui_space%3Amatrix.org)
*   [https://www.comfy.org/](https://www.comfy.org/)

## Frontend Development

As of August 15, 2024, frontend development is now in the [ComfyUI Frontend repository](https://github.com/Comfy-Org/ComfyUI_frontend).

### Reporting Issues and Requesting Features

Please report frontend-specific issues in the [ComfyUI Frontend repository](https://github.com/Comfy-Org/ComfyUI_frontend).

### Using the Latest Frontend

*   For the latest release: `--front-end-version Comfy-Org/ComfyUI_frontend@latest`
*   For a specific version: `--front-end-version Comfy-Org/ComfyUI_frontend@<version>`

### Accessing the Legacy Frontend

Use the command line argument `--front-end-version Comfy-Org/ComfyUI_legacy_frontend@latest`.

## QA

### Which GPU should I buy for this?

[See this page for recommendations](https://github.com/comfyanonymous/ComfyUI/wiki/Which-GPU-should-I-buy-for-ComfyUI)