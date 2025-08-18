<div align="center">

# ComfyUI: The Modular Visual AI Engine

**Unleash the power of visual AI with ComfyUI, a node-based interface for crafting and executing advanced Stable Diffusion workflows.**  Explore the innovative world of AI image and video generation with the most powerful and modular visual AI engine!

[![Website](https://img.shields.io/badge/ComfyOrg-4285F4?style=flat)][website-url]
[![Discord](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fdiscord.com%2Fapi%2Finvites%2Fcomfyorg%3Fwith_counts%3Dtrue&query=%24.approximate_member_count&logo=discord&logoColor=white&label=Discord&color=green&suffix=%20total)][discord-url]
[![Twitter](https://img.shields.io/twitter/follow/ComfyUI)][twitter-url]
[![Matrix](https://img.shields.io/badge/Matrix-000000?style=flat&logo=matrix&logoColor=white)][matrix-url]
<br>
[![GitHub Release](https://img.shields.io/github/v/release/comfyanonymous/ComfyUI?style=flat&sort=semver)][github-release-link]
[![Release Date](https://img.shields.io/github/release-date/comfyanonymous/ComfyUI?style=flat)][github-release-link]
[![Total Downloads](https://img.shields.io/github/downloads/comfyanonymous/ComfyUI/total?style=flat)][github-downloads-link]
[![Latest Downloads](https://img.shields.io/github/downloads/comfyanonymous/ComfyUI/latest/total?style=flat&label=downloads%40latest)][github-downloads-link]

[matrix-url]: https://app.element.io/#/room/%23comfyui_space%3Amatrix.org
[website-url]: https://www.comfy.org/
[discord-url]: https://www.comfy.org/discord
[twitter-url]: https://x.com/ComfyUI
[github-release-link]: https://github.com/comfyanonymous/ComfyUI/releases
[github-downloads-link]: https://github.com/comfyanonymous/ComfyUI/releases

![ComfyUI Screenshot](https://github.com/user-attachments/assets/7ccaf2c1-9b72-41ae-9a89-5688c94b7abe)
</div>

**[Explore the ComfyUI Repository](https://github.com/comfyanonymous/ComfyUI)**

ComfyUI provides a flexible, node-based interface for designing and executing complex Stable Diffusion pipelines. Available on Windows, Linux, and macOS, it empowers both beginners and advanced users to experiment and create AI-generated content visually.

**Key Features:**

*   **Visual Workflow Creation:** Design intricate Stable Diffusion workflows using a node-based, flowchart interface, eliminating the need for extensive coding.
*   **Broad Model Support:** Compatibility with a wide range of image, video, and audio models, including SD1.x, SD2.x, SDXL, Stable Cascade, SD3, and more, along with video and audio models.
*   **Advanced Features:** Includes an asynchronous queue system, smart memory management for low-VRAM GPUs, support for loading checkpoints, embeddings, LoRAs, and Hypernetworks.
*   **Workflow Flexibility:** Load, save, and share workflows as JSON files and utilize features like inpainting, ControlNet, upscaling, and model merging for enhanced creative control.
*   **Customization:** Utilize the config file to set the search paths for models.
*   **Modular Design:** Only re-executes the parts of the workflow that change, optimizing performance.
*   **Offline Functionality:** Works fully offline and optionally can utilize paid models from external providers through the online Comfy API.
*   **High-Quality Previews:** Supports TAESD for higher-quality preview generation.
*   **Cross-Platform Compatibility:** Works on Windows, Linux, and macOS, supporting various GPU types and CPU fallback.

**Get Started:**

*   **Desktop Application:** Simplest method, available for Windows & macOS ([Download](https://www.comfy.org/download))
*   **Windows Portable Package:** Latest commits, fully portable. ([Releases](https://github.com/comfyanonymous/ComfyUI/releases/latest/download/ComfyUI_windows_portable_nvidia.7z))
*   **Manual Install:** Supports all OS and GPU types (NVIDIA, AMD, Intel, Apple Silicon, Ascend)

**Examples:**

*   Explore the possibilities with pre-made workflows: [Examples](https://comfyanonymous.github.io/ComfyUI_examples/)

**Installation:**

*   **Windows Portable:** Download and extract the standalone build from the releases page. Place models in the `ComfyUI/models/checkpoints` folder.
*   **comfy-cli:** `pip install comfy-cli` then `comfy install`
*   **Manual Install:**
    1.  Clone the repository.
    2.  Place SD checkpoints and VAE in the respective `models` folders.
    3.  Install dependencies: `pip install -r requirements.txt`
    4.  Follow specific instructions for AMD, Intel, NVIDIA, Apple Silicon, DirectML, Ascend NPUs, Cambricon MLUs, and Iluvatar Corex for optimal setup.

**Running:**

*   Open your terminal in the ComfyUI directory and execute: `python main.py`

**Shortcuts:**

*   A comprehensive list of keyboard shortcuts is provided within the original README, as well as details about running the system.

**Support:**

*   **Discord:** Try the #help or #feedback channels.
*   **Matrix:** [Matrix space: #comfyui_space:matrix.org](https://app.element.io/#/room/%23comfyui_space%3Amatrix.org)
*   **Website:** [https://www.comfy.org/](https://www.comfy.org/)

**Frontend Development**

*   The frontend is hosted in a separate repository: [ComfyUI Frontend](https://github.com/Comfy-Org/ComfyUI_frontend)
*   Reporting Issues and Requesting Features: [ComfyUI Frontend repository](https://github.com/Comfy-Org/ComfyUI_frontend)
*   Use the command line argument `--front-end-version` to select a specific version of the frontend.