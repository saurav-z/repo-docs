<div align="center">

# ComfyUI: The Most Powerful and Modular Visual AI Engine

**Unleash your creativity with ComfyUI, a revolutionary visual AI engine that empowers you to design and execute complex Stable Diffusion workflows with an intuitive, node-based interface.** ([Original Repository](https://github.com/comfyanonymous/ComfyUI))

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

ComfyUI allows you to design and execute advanced stable diffusion pipelines using a graph/nodes/flowchart based interface, offering unparalleled flexibility and control. Compatible with Windows, Linux, and macOS.

## Key Features:

*   **Node-Based Workflow:** Create intricate Stable Diffusion workflows using a visual, node-based interface without coding.
*   **Broad Model Support:**
    *   **Image Generation:** SD1.x, SD2.x, SDXL, SDXL Turbo, Stable Cascade, SD3 and SD3.5, and more (Pixart Alpha and Sigma, AuraFlow, HunyuanDiT, Flux, Lumina Image 2.0, HiDream, Cosmos Predict2, Qwen Image).
    *   **Image Editing:** Omnigen 2, Flux Kontext, HiDream E1.1, Qwen Image Edit.
    *   **Video Generation:** Stable Video Diffusion, Mochi, LTX-Video, Hunyuan Video, Nvidia Cosmos and Cosmos Predict2, Wan 2.1, Wan 2.2
    *   **Audio Generation:** Stable Audio, ACE Step.
    *   **3D Models:** Hunyuan3D 2.0
*   **Optimized Performance:** Asynchronous queue system and smart memory management for efficient operation, even on lower-spec GPUs (down to 1GB VRAM with smart offloading).
*   **Model Compatibility:** Loads ckpt, safetensors, and various diffusion models, VAEs, and CLIP models.
*   **Workflow Flexibility:** Support for embeddings, LoRAs, Hypernetworks, and full workflow loading from PNG, WebP and FLAC files.
*   **Advanced Features:** Includes area composition, inpainting, ControlNet, T2I-Adapter, upscaling, GLIGEN, model merging, LCM models and Loras.
*   **High-Quality Previews:** Utilize [TAESD](https://github.com/madebyollin/taesd) for enhanced preview quality.
*   **Offline Functionality:** Fully functional offline; no downloads unless required.
*   **API Integration:** Optional API nodes for paid model access via the online [Comfy API](https://docs.comfy.org/tutorials/api-nodes/overview).

## Getting Started:

*   **Desktop Application:** [Download](https://www.comfy.org/download) for the easiest setup (Windows & macOS).
*   **Windows Portable Package:** Get the latest commits (Windows).
*   **Manual Install:** Supports all OS and GPU types (NVIDIA, AMD, Intel, Apple Silicon, Ascend).

## Examples:

Explore what's possible with [example workflows](https://comfyanonymous.github.io/ComfyUI_examples/).

## Shortcuts

See the list of keyboard shortcuts in the original README.

## Installation Instructions:

*   **Windows Portable:** Download, extract, and run from the releases page.  Place models in `ComfyUI\models\checkpoints`.
*   **comfy-cli:**  `pip install comfy-cli; comfy install`
*   **Manual Install (Windows, Linux, macOS):**
    1.  Clone the repository.
    2.  Place your model files (ckpt/safetensors) in `models/checkpoints` and VAEs in `models/vae`.
    3.  Follow the instructions for [AMD](#amd-gpus-linux-only), [Intel](#intel-gpus-windows-and-linux), [NVIDIA](#nvidia), and other [Hardware](#others) in the original README.
    4.  Install dependencies: `pip install -r requirements.txt`
    5.  Run: `python main.py`

## Advanced Topics and Troubleshooting:

*   **Running ComfyUI**  with specific GPU configurations (AMD, Intel, NVIDIA) and troubleshooting steps can be found in the original README.
*   **Preview Quality:** Use `--preview-method auto` for fast previews, and configure TAESD for high-quality previews (instructions in original README).
*   **TLS/SSL:**  Instructions on generating self-signed certificates for secure access (original README).
*   **Support:**  Get help and connect with the community on [Discord](https://comfy.org/discord) and [Matrix](https://app.element.io/#/room/%23comfyui_space%3Amatrix.org).
*   **GPU Recommendations:**  Find GPU recommendations at  [Which-GPU-should-I-buy-for-ComfyUI](https://github.com/comfyanonymous/ComfyUI/wiki/Which-GPU-should-I-buy-for-ComfyUI).