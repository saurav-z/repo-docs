<div align="center">

# ComfyUI: Unleash Your AI Artistry with a Powerful Visual Workflow Engine

**ComfyUI is a groundbreaking, node-based visual AI engine that empowers you to craft intricate and stunning AI-generated art using Stable Diffusion and beyond.**

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

ComfyUI offers a flexible, modular, and powerful environment for creating and manipulating AI art, all within an intuitive, visual interface. Designed for Windows, Linux, and macOS, ComfyUI leverages a node-based workflow, empowering users to build intricate pipelines without writing any code.

## Key Features:

*   **Node-Based Workflow:** Design and execute advanced Stable Diffusion pipelines visually using a graph/nodes/flowchart interface.
*   **Extensive Model Support:**
    *   SD1.x, SD2.x, SDXL, SDXL Turbo, Stable Cascade, SD3 & SD3.5, and more.
    *   Image Editing Models: Omnigen 2, Flux Kontext, HiDream E1.1, Qwen Image Edit
    *   Video Models: Stable Video Diffusion, Mochi, LTX-Video, Hunyuan Video, Wan 2.1/2.2
    *   Audio Models: Stable Audio, ACE Step
    *   3D Models: Hunyuan3D 2.0
*   **Advanced Capabilities:**
    *   Asynchronous Queue system for efficient processing.
    *   Smart memory management for low-VRAM GPUs, supports down to 1GB vram.
    *   Supports various model formats (ckpt, safetensors, etc.).
    *   Embeddings, textual inversions, LoRAs, and Hypernetworks.
    *   Workflow loading and saving (PNG, WebP, JSON).
    *   Inpainting, ControlNet, T2I-Adapter.
    *   Upscale Models (ESRGAN, SwinIR, etc.), GLIGEN, Model Merging, and LCM models.
*   **Flexible and Modular:**
    *   Offline functionality: core will never download unless you want it to.
    *   API nodes for paid models from external providers via [Comfy API](https://docs.comfy.org/tutorials/api-nodes/overview).
    *   Config file (`extra_model_paths.yaml.example`) for managing model search paths.
*   **High-Quality Previews:** Utilize TAESD for improved preview quality.
*   **Customization:**  Use the available [examples](https://comfyanonymous.github.io/ComfyUI_examples/) to start creating unique and complex workflows.

**[Explore ComfyUI's Capabilities in Detail](https://github.com/comfyanonymous/ComfyUI)**

## Getting Started

ComfyUI is available through the following options:

### Desktop Application ([Download](https://www.comfy.org/download))

*   Simplest way to get started.
*   Available on Windows & macOS.

### Windows Portable Package

*   Portable and up-to-date version.
*   Download from the [Releases page](https://github.com/comfyanonymous/ComfyUI/releases).

### Manual Install (Windows, Linux, macOS)

Supports all operating systems and GPU types (NVIDIA, AMD, Intel, Apple Silicon, Ascend, Cambricon MLUs, Iluvatar Corex).  See [Manual Install](#manual-install-windows-linux) for detailed instructions.

### [comfy-cli](https://docs.comfy.org/comfy-cli/getting-started)

You can install and start ComfyUI using comfy-cli:

```bash
pip install comfy-cli
comfy install
```

## Core Concepts

*   **Nodes:** Building blocks of your AI workflows.
*   **Graph/Flowchart:** Visual representation of your pipeline.
*   **Workflows:** Customizable pipelines for image, video, and audio generation.

## Release Process
ComfyUI follows a weekly release cycle targeting Friday but this regularly changes because of model releases or large changes to the codebase. There are three interconnected repositories:

1.  **[ComfyUI Core](https://github.com/comfyanonymous/ComfyUI)**
    *   Releases a new stable version (e.g., v0.7.0)
    *   Serves as the foundation for the desktop release

2.  **[ComfyUI Desktop](https://github.com/Comfy-Org/desktop)**
    *   Builds a new release using the latest stable core version

3.  **[ComfyUI Frontend](https://github.com/Comfy-Org/ComfyUI_frontend)**
    *   Weekly frontend updates are merged into the core repository
    *   Features are frozen for the upcoming core release
    *   Development continues for the next release cycle

## Additional Resources

*   [Examples](https://comfyanonymous.github.io/ComfyUI_examples/)
*   [Wiki](https://github.com/comfyanonymous/ComfyUI/wiki) (e.g. Which GPU should I buy for this?)

## Support

*   [Discord](https://comfy.org/discord): Get help in the #help or #feedback channels.
*   [Matrix](https://app.element.io/#/room/%23comfyui_space%3Amatrix.org)

---

**[Get Started with ComfyUI Today!](https://github.com/comfyanonymous/ComfyUI)**