# ComfyUI: The Most Powerful and Modular Visual AI Engine

**Unleash your creativity with ComfyUI, a node-based interface for designing and executing advanced AI image pipelines.**  ([Original Repository](https://github.com/comfyanonymous/ComfyUI))

[![Website](https://img.shields.io/badge/ComfyOrg-4285F4?style=flat)](https://www.comfy.org/)
[![Discord](https://img.shields.io/badge/Discord-total?style=flat&logo=discord&logoColor=white)](https://www.comfy.org/discord)
[![Twitter](https://img.shields.io/twitter/follow/ComfyUI?style=flat)](https://x.com/ComfyUI)
[![Matrix](https://img.shields.io/badge/Matrix-000000?style=flat&logo=matrix&logoColor=white)](https://app.element.io/#/room/%23comfyui_space%3Amatrix.org)
[![GitHub Release](https://img.shields.io/github/v/release/comfyanonymous/ComfyUI?style=flat&sort=semver)](https://github.com/comfyanonymous/ComfyUI/releases)
[![GitHub Release Date](https://img.shields.io/github/release-date/comfyanonymous/ComfyUI?style=flat)](https://github.com/comfyanonymous/ComfyUI/releases)
[![GitHub Downloads](https://img.shields.io/github/downloads/comfyanonymous/ComfyUI/total?style=flat)](https://github.com/comfyanonymous/ComfyUI/releases)
[![GitHub Downloads Latest](https://img.shields.io/github/downloads/comfyanonymous/ComfyUI/latest/total?style=flat&label=downloads%40latest)](https://github.com/comfyanonymous/ComfyUI/releases)

![ComfyUI Screenshot](https://github.com/user-attachments/assets/7ccaf2c1-9b72-41ae-9a89-5688c94b7abe)

ComfyUI empowers you to create intricate Stable Diffusion workflows using a user-friendly, node-based interface, offering unparalleled flexibility and control.  Compatible with Windows, Linux, and macOS.

## Key Features

*   **Node-Based Workflow:** Design complex AI image generation pipelines visually without coding.
*   **Extensive Model Support:**  Supports a wide range of image, video, and audio models, including:
    *   SD1.x, SD2.x, SDXL, SDXL Turbo, Stable Cascade, SD3/SD3.5, and more.
    *   Video models like Stable Video Diffusion, Mochi, LTX-Video, and others.
    *   Audio models like Stable Audio and ACE Step.
    *   3D models like Hunyuan3D 2.0
*   **Performance and Optimization:**  Leverages an asynchronous queue system, memory optimization, and smart offloading for efficient use on various hardware configurations, including GPUs with as little as 1GB VRAM, and CPU fallback.
*   **Model and File Compatibility:**  Loads various checkpoint and safetensors files, VAEs, CLIP models, embeddings, LoRAs, and hypernetworks.
*   **Workflow Management:**  Load and save complete workflows as PNG, WebP, FLAC and JSON files, preserving seeds and settings.
*   **Advanced Features:** Includes features like Inpainting, ControlNet, T2I-Adapter, upscaling, GLIGEN, and Model Merging.
*   **Preview Enhancements:**  Offers high-quality preview options with TAESD for better visual results.
*   **Offline Functionality:** Operates fully offline, ensuring privacy and control, with optional API integration for paid models.

## Getting Started

Choose your preferred method:

*   **[Desktop Application](https://www.comfy.org/download):**  The easiest way to get started, available on Windows and macOS.
*   **[Windows Portable Package](https://github.com/comfyanonymous/ComfyUI/releases):**  Get the latest commits, completely portable, for Windows.  Download, extract with [7-Zip](https://7-zip.org) and run. Make sure you put your Stable Diffusion checkpoints/models (the huge ckpt/safetensors files) in: `ComfyUI\models\checkpoints`
*   **[Manual Install](#manual-install-windows-linux):**  Supports all operating systems and GPU types (NVIDIA, AMD, Intel, Apple Silicon, Ascend).

## Examples & Tutorials

Explore the possibilities with the [example workflows](https://comfyanonymous.github.io/ComfyUI_examples/) to learn how to build different workflows.

## Release Process

ComfyUI follows a weekly release cycle.  See the [Release Process](#release-process) section for more details.

## Shortcuts

[See the original README for the full list of shortcuts.](https://github.com/comfyanonymous/ComfyUI#shortcuts)

## Installing

### Windows Portable

Download, extract with [7-Zip](https://7-zip.org) and run the portable version found at the [releases page](https://github.com/comfyanonymous/ComfyUI/releases) for Windows.

### Manual Install (Windows, Linux)

1.  **Prerequisites:**  Python 3.12 (Recommended)
2.  **Clone the repository:** `git clone [repository URL]`
3.  **Model Placement:** Put your SD checkpoints (ckpt/safetensors files) in the `models/checkpoints` directory, and your VAE in `models/vae`.
4.  **GPU-Specific Instructions:**
    *   **AMD (Linux):** Install rocm and pytorch with pip.
    *   **Intel (Windows/Linux):** Install PyTorch xpu or leverage Intel Extension for PyTorch (IPEX)
    *   **NVIDIA:** Install pytorch with pip, and the `cu129` extra index.
    *   **Apple Mac silicon:** Install pytorch nightly, and follow [manual installation](#manual-install-windows-linux) steps.
    *   **DirectML (AMD Windows):** `pip install torch-directml` (Not Recommended)
    *   **Ascend NPUs:** Install Ascend Basekit, then torch-npu.
    *   **Cambricon MLUs:** Install the Cambricon CNToolkit then PyTorch(torch_mlu).
    *   **Iluvatar Corex:** Install the Iluvatar Corex Toolkit.
5.  **Install Dependencies:** `pip install -r requirements.txt`

## Running

1.  Navigate to the ComfyUI directory in your terminal.
2.  Run `python main.py`.

## Notes

*   ComfyUI executes only the necessary parts of the graph.
*   Drag and drop generated images to load their workflows.
*   Use emphasis syntax like `(good code:1.2)` and wildcard prompts, and C-style comments.
*   Use embeddings in prompts with `embedding:filename.pt`.

## How to show high-quality previews?

Enable high-quality previews with TAESD.  Download the [taesd\_decoder.pth, taesdxl\_decoder.pth, taesd3\_decoder.pth and taef1\_decoder.pth](https://github.com/madebyollin/taesd/) and place them in the `models/vae_approx` folder, then launch ComfyUI with `--preview-method taesd`

## How to use TLS/SSL?

Use `--tls-keyfile key.pem --tls-certfile cert.pem` to enable TLS/SSL.

## Support and Community

*   [Discord](https://comfy.org/discord)
*   [Matrix space](https://app.element.io/#/room/%23comfyui_space%3Amatrix.org)
*   [https://www.comfy.org/](https://www.comfy.org/)

## Frontend Development

The new frontend is hosted in a separate repository: [ComfyUI Frontend](https://github.com/Comfy-Org/ComfyUI_frontend).
To use the most up-to-date frontend version: launch ComfyUI with `--front-end-version Comfy-Org/ComfyUI_frontend@latest`.

### Reporting Issues and Requesting Features

Report issues and feature requests related to the frontend at the [ComfyUI Frontend repository](https://github.com/Comfy-Org/ComfyUI_frontend).

## QA

*   [Which GPU should I buy for this?](https://github.com/comfyanonymous/ComfyUI/wiki/Which-GPU-should-I-buy-for-ComfyUI)