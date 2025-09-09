<div align="center">

# ComfyUI: Unleash Your AI Creativity with a Visual Workflow Engine

**ComfyUI is the ultimate visual AI engine, empowering you to create and execute advanced Stable Diffusion pipelines with an intuitive node-based interface.** Explore the power of AI image generation with this versatile and modular application.

[![Website][website-shield]][website-url]
[![Dynamic JSON Badge][discord-shield]][discord-url]
[![Twitter][twitter-shield]][twitter-url]
[![Matrix][matrix-shield]][matrix-url]
<br>
[![][github-release-shield]][github-release-link]
[![][github-release-date-shield]][github-release-link]
[![][github-downloads-shield]][github-downloads-link]
[![][github-downloads-latest-shield]][github-downloads-link]

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
[github-downloads-shield]: https://img.shields.io/github/downloads/comfyanonymous/ComfyUI/total?style=flat
[github-downloads-latest-shield]: https://img.shields.io/github/downloads/comfyanonymous/ComfyUI/latest/total?style=flat&label=downloads%40latest
[github-downloads-link]: https://github.com/comfyanonymous/ComfyUI/releases

![ComfyUI Screenshot](https://github.com/user-attachments/assets/7ccaf2c1-9b72-41ae-9a89-5688c94b7abe)
</div>

## Key Features of ComfyUI:

*   **Node-Based Interface:**  Design and execute complex Stable Diffusion workflows visually without coding.
*   **Extensive Model Support:** Compatible with a vast array of image, video, audio, and 3D models, including:
    *   SD1.x, SD2.x (and [unCLIP](https://comfyanonymous.github.io/ComfyUI_examples/unclip/))
    *   SDXL, SDXL Turbo
    *   Stable Cascade
    *   SD3 and SD3.5
    *   Pixart Alpha and Sigma
    *   And many more, with dedicated examples for each!
*   **Comprehensive Workflow Capabilities:**
    *   Asynchronous Queue system for efficient processing.
    *   Smart memory management, allowing use on GPUs with limited VRAM.
    *   Supports CPU fallback mode if no GPU is available.
    *   Loads ckpt, safetensors, embeddings, LoRAs, and Hypernetworks.
    *   Full workflow loading and saving (JSON, PNG, WebP, FLAC).
    *   Advanced features: Hires fix, inpainting, ControlNet, and more.
*   **Easy to Get Started:** Available as a [Desktop Application](https://www.comfy.org/download), [Windows Portable Package](#installing), and through [Manual Install](#manual-install-windows-linux) for various OS.
*   **Optimized for Performance:** Only re-executes the parts of the workflow that change.
*   **Rich Ecosystem:** Access to various examples and community-created workflows through the [Examples page](https://comfyanonymous.github.io/ComfyUI_examples/).
*   **Flexible & Offline-Friendly:** Works fully offline; only downloads are user-initiated.
*   **API Integration:** Optional API nodes allow using paid models.
*   **Customization:** Easily set search paths for models via the [Config file](extra_model_paths.yaml.example).

## Getting Started

Choose the best installation method for you:

*   **[Desktop Application](https://www.comfy.org/download)**: Easiest option, available for Windows & macOS.
*   **[Windows Portable Package](#installing)**: Get the latest updates, fully portable.
*   **[Manual Install](#manual-install-windows-linux)**: For all operating systems and GPU types (NVIDIA, AMD, Intel, Apple Silicon, Ascend, Cambricon, Iluvatar).

## Explore ComfyUI Examples

Discover what's possible with ComfyUI by exploring a wide range of workflows at the [Examples page](https://comfyanonymous.github.io/ComfyUI_examples/).

## Release Process

ComfyUI follows a weekly release cycle.  The project has three interconnected repositories:

1.  **[ComfyUI Core](https://github.com/comfyanonymous/ComfyUI)**:  Releases stable versions.
2.  **[ComfyUI Desktop](https://github.com/Comfy-Org/desktop)**:  Builds releases using the latest core.
3.  **[ComfyUI Frontend](https://github.com/Comfy-Org/ComfyUI_frontend)**:  Hosts the frontend, with updates merged into the core.

## Shortcuts
... (Keep the shortcuts table here)

## Installing

### Windows Portable
... (Keep the Windows Portable instructions here)

### [comfy-cli](https://docs.comfy.org/comfy-cli/getting-started)
... (Keep the comfy-cli instructions here)

### Manual Install (Windows, Linux)
... (Keep the Manual Install instructions here)

## Running
... (Keep the Running instructions here)

## Notes
... (Keep the Notes section here)

## How to show high-quality previews?
... (Keep the High-Quality Previews instructions here)

## How to use TLS/SSL?
... (Keep the TLS/SSL instructions here)

## Support and dev channel
... (Keep the Support and dev channel information here)

## Frontend Development
... (Keep the Frontend Development information here)

## QA
... (Keep the QA section here)