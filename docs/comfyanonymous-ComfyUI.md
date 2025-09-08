<div align="center">
  <a href="https://github.com/comfyanonymous/ComfyUI">
    <img src="https://github.com/user-attachments/assets/7ccaf2c1-9b72-41ae-9a89-5688c94b7abe" alt="ComfyUI Screenshot" width="80%">
  </a>
</div>

# ComfyUI: Unleash Your Creativity with Advanced AI Image Generation

ComfyUI is a powerful and modular visual AI engine, allowing you to design and execute intricate Stable Diffusion workflows with ease.

[![Website](https://img.shields.io/badge/ComfyOrg-4285F4?style=flat)](https://www.comfy.org/)
[![Discord](https://img.shields.io/badge/Discord-green?style=flat&logo=discord&logoColor=white&label=Discord)](https://www.comfy.org/discord)
[![Twitter](https://img.shields.io/twitter/follow/ComfyUI?style=flat)](https://x.com/ComfyUI)
[![Matrix](https://img.shields.io/badge/Matrix-000000?style=flat&logo=matrix&logoColor=white)](https://app.element.io/#/room/%23comfyui_space%3Amatrix.org)
<br>
[![GitHub Release](https://img.shields.io/github/v/release/comfyanonymous/ComfyUI?style=flat&sort=semver)](https://github.com/comfyanonymous/ComfyUI/releases)
[![GitHub Release Date](https://img.shields.io/github/release-date/comfyanonymous/ComfyUI?style=flat)](https://github.com/comfyanonymous/ComfyUI/releases)
[![GitHub Downloads](https://img.shields.io/github/downloads/comfyanonymous/ComfyUI/total?style=flat)](https://github.com/comfyanonymous/ComfyUI/releases)
[![GitHub Downloads Latest](https://img.shields.io/github/downloads/comfyanonymous/ComfyUI/latest/total?style=flat&label=downloads%40latest)](https://github.com/comfyanonymous/ComfyUI/releases)

## Key Features

*   **Node-Based Workflow:** Create complex Stable Diffusion pipelines using an intuitive graph/node/flowchart interface, eliminating the need for extensive coding.
*   **Extensive Model Support:** Compatible with a wide range of image, video, and audio models, including SD1.x, SD2.x, SDXL, Stable Cascade, Stable Video Diffusion, and more.
*   **Image & Video Editing:** Utilize specialized models for image editing and video generation, such as Omnigen 2, Flux Kontext, and Stable Video Diffusion.
*   **Asynchronous Queue System:** Optimize your workflow with an asynchronous queue for efficient processing.
*   **Smart Memory Management:** Run large models even on GPUs with limited VRAM (as low as 1GB) through intelligent offloading.
*   **Offline Functionality:** ComfyUI operates fully offline, ensuring your data remains private and reducing reliance on external services.
*   **Workflow Import/Export:** Load and save complete workflows from PNG, WebP, and FLAC files, and export as JSON files for easy sharing.
*   **Advanced Features:** Supports embeddings, LoRAs, Hypernetworks, ControlNet, T2I-Adapter, upscaling models, GLIGEN, model merging, LCM models, and latent previews.
*   **Customization:**  Configure model paths and settings using a configuration file.
*   **API Integration:**  Optional API nodes to connect with paid models from external providers via the online [Comfy API](https://docs.comfy.org/tutorials/api-nodes/overview).

## Getting Started

Choose your preferred method for getting started:

*   **[Desktop Application](https://www.comfy.org/download):** The easiest way to begin, available on Windows & macOS.
*   **[Windows Portable Package](https://github.com/comfyanonymous/ComfyUI/releases/latest/download/ComfyUI_windows_portable_nvidia.7z):** Get the latest updates and enjoy a fully portable experience on Windows.
*   **[Manual Install](#manual-install-windows-linux):** Supports all operating systems and GPU types (NVIDIA, AMD, Intel, Apple Silicon, Ascend).
*   **[comfy-cli](https://docs.comfy.org/comfy-cli/getting-started):** Install and start ComfyUI using comfy-cli.

## Examples

Explore the possibilities with the [example workflows](https://comfyanonymous.github.io/ComfyUI_examples/).

## Installation

### Windows Portable

1.  Download the [portable standalone build](https://github.com/comfyanonymous/ComfyUI/releases).
2.  Extract the contents using [7-Zip](https://7-zip.org).
3.  Place your Stable Diffusion checkpoints/models in the `ComfyUI\models\checkpoints` directory.
4.  Run the application.

    *   If you have trouble extracting it, right click the file -> properties -> unblock
5.  To share models with other UIs, see the [Config file](extra_model_paths.yaml.example).

### comfy-cli

```bash
pip install comfy-cli
comfy install
```

### Manual Install (Windows, Linux)

1.  **Prerequisites:**
    *   Python 3.12 or 3.13 is recommended.
2.  **Clone the Repository:**
    ```bash
    git clone https://github.com/comfyanonymous/ComfyUI
    cd ComfyUI
    ```
3.  **Model Placement:**
    *   Place your SD checkpoints (e.g., ckpt/safetensors files) in `models/checkpoints`.
    *   Place your VAE in `models/vae`.
4.  **GPU-Specific Instructions:**
    *   **AMD GPUs (Linux):**
        ```bash
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.4
        ```
        (Use the nightly build command for potential performance improvements.)
    *   **Intel GPUs (Windows and Linux):**
        *   **(Option 1) PyTorch xpu:**
            ```bash
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
            ```
            (Use the nightly build command for potential performance improvements.)
        *   **(Option 2) Intel Extension for PyTorch (IPEX):** Follow the [Installation](https://intel.github.io/intel-extension-for-pytorch/index.html#installation?platform=gpu) instructions.
    *   **NVIDIA:**
        ```bash
        pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu129
        ```
        (Use the nightly build command for potential performance improvements.)
5.  **Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
6.  **Run ComfyUI:**
    ```bash
    python main.py
    ```

### Other Platforms

*   **Apple Silicon (M1/M2):** Follow the [Accelerated PyTorch training on Mac](https://developer.apple.com/metal/pytorch/) guide, then follow the manual install instructions.
*   **DirectML (AMD on Windows):**  This is poorly supported.  Install `torch-directml` and run with `--directml`.  Consider alternative options for better performance.
*   **Ascend NPUs:** Follow the steps described in the [Ascend NPUs](https://github.com/comfyanonymous/ComfyUI#ascend-npus) section.
*   **Cambricon MLUs:** Follow the steps described in the [Cambricon MLUs](https://github.com/comfyanonymous/ComfyUI#cambricon-mlus) section.
*   **Iluvatar Corex:** Follow the steps described in the [Iluvatar Corex](https://github.com/comfyanonymous/ComfyUI#iluvatar-corex) section.

## Running

1.  Navigate to the ComfyUI directory in your terminal.
2.  Run the command:
    ```bash
    python main.py
    ```

### Troubleshooting

*   **AMD ROCm:** Try `HSA_OVERRIDE_GFX_VERSION=10.3.0 python main.py` (for older RDNA2) or `HSA_OVERRIDE_GFX_VERSION=11.0.0 python main.py` (for RDNA3).
*   **AMD ROCm Experimental:**  Enable experimental memory efficient attention:
    ```bash
    TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 python main.py --use-pytorch-cross-attention
    ```
    *   Try setting env variable: `PYTORCH_TUNABLEOP_ENABLED=1`

## Workflow & Interface Tips

*   Only parts of the graph with complete inputs will be executed.
*   Only changes between executions will be re-executed.
*   Drag a generated PNG to load the full workflow.
*   Use `()` for emphasis and `//` or `/* */` for comments.
*   Use curly braces `{}` for wildcard/dynamic prompts.
*   Place textual inversion concepts in the `/models/embeddings` directory.
*   Quickly search nodes with Double-Click LMB.

## High-Quality Previews

*   Use `--preview-method auto`.
*   Install [TAESD](https://github.com/madebyollin/taesd) and place the files in `models/vae_approx` then start with `--preview-method taesd`.

## TLS/SSL

1.  Generate a self-signed certificate:
    ```bash
    openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -sha256 -days 3650 -nodes -subj "/C=XX/ST=StateName/L=CityName/O=CompanyName/OU=CompanySectionName/CN=CommonNameOrHostname"
    ```
2.  Run with: `--tls-keyfile key.pem --tls-certfile cert.pem` (access via `https://...`).

## Support and Development

*   [Discord](https://comfy.org/discord): #help or #feedback channels.
*   [Matrix](https://app.element.io/#/room/%23comfyui_space%3Amatrix.org).
*   [ComfyUI Website](https://www.comfy.org/)

## Frontend Development

*   The new frontend is now hosted in a separate repository: [ComfyUI Frontend](https://github.com/Comfy-Org/ComfyUI_frontend).
*   Report frontend issues in the [ComfyUI Frontend repository](https://github.com/Comfy-Org/ComfyUI_frontend).
*   Use the latest daily frontend release: `--front-end-version Comfy-Org/ComfyUI_frontend@latest`.
*   For specific versions: `--front-end-version Comfy-Org/ComfyUI_frontend@1.2.2`.
*   Access the legacy frontend: `--front-end-version Comfy-Org/ComfyUI_legacy_frontend@latest`.

## QA

*   [GPU Recommendations](https://github.com/comfyanonymous/ComfyUI/wiki/Which-GPU-should-I-buy-for-ComfyUI)

**For the latest updates, tutorials, and community discussions, please visit the [ComfyUI GitHub Repository](https://github.com/comfyanonymous/ComfyUI).**