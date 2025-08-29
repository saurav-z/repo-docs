<div align="center">

# ComfyUI: Unleash Your AI Creativity with a Visual Workflow Interface

**ComfyUI is the ultimate visual AI engine, offering a powerful and modular node-based interface for creating and executing complex Stable Diffusion workflows.** ([See original repo](https://github.com/comfyanonymous/ComfyUI))

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

ComfyUI empowers you to design and execute advanced Stable Diffusion pipelines through an intuitive node-based interface, available on Windows, Linux, and macOS.

## Key Features:

*   **Visual Workflow Creation:** Design complex Stable Diffusion pipelines using a node/graph/flowchart interface without any coding required.
*   **Extensive Model Support:**
    *   **Image Generation:** SD1.x, SD2.x, SDXL, SDXL Turbo, Stable Cascade, SD3/SD3.5, Pixart Alpha/Sigma, AuraFlow, HunyuanDiT, Flux, Lumina Image 2.0, HiDream, and Qwen Image.
    *   **Image Editing:** Omnigen 2, Flux Kontext, HiDream E1.1, and Qwen Image Edit.
    *   **Video Generation:** Stable Video Diffusion, Mochi, LTX-Video, Hunyuan Video, Wan 2.1/2.2.
    *   **Audio Generation:** Stable Audio, ACE Step.
    *   **3D Generation:** Hunyuan3D 2.0
*   **Optimized Performance:**
    *   Asynchronous queue system for efficient processing.
    *   Optimized to re-execute only changed portions of workflows.
    *   Smart memory management allows use on low-VRAM GPUs (1GB+).
    *   Supports CPU fallback (`--cpu`).
*   **Model Compatibility:** Loads ckpt and safetensors, standalone diffusion models, VAEs, and CLIP models.
*   **Advanced Features:**
    *   Embeddings/Textual Inversion.
    *   LoRAs (regular, locon, and loha).
    *   Hypernetworks.
    *   Workflow loading/saving (PNG, WebP, FLAC, JSON).
    *   Advanced workflow capabilities like Hires fix, area composition, inpainting, ControlNet/T2I-Adapter, and upscaling.
    *   GLIGEN and Model Merging.
    *   LCM models and Loras.
    *   TAESD for high-quality previews.
    *   Offline operation; no downloads unless explicitly requested.
    *   API nodes for external paid models through the [Comfy API](https://docs.comfy.org/tutorials/api-nodes/overview).
*   **Customization:** Config file for model search paths (`extra_model_paths.yaml.example`).
*   **Extensive Examples:** Explore workflow examples on the [Examples page](https://comfyanonymous.github.io/ComfyUI_examples/).

## Getting Started:

*   **Desktop Application:** Download the easiest way to get started for Windows & macOS from [ComfyUI Download](https://www.comfy.org/download).
*   **Windows Portable Package:** Get the latest commits and completely portable version for Windows from the [releases page](https://github.com/comfyanonymous/ComfyUI/releases).
*   **Manual Install:** Supports all OS and GPU types (NVIDIA, AMD, Intel, Apple Silicon, Ascend). See installation instructions below.
*   **comfy-cli:** `pip install comfy-cli` then `comfy install` - See [comfy-cli getting started](https://docs.comfy.org/comfy-cli/getting-started)

## Installation Guides:

### Windows Portable:

1.  Download the portable version from the [releases page](https://github.com/comfyanonymous/ComfyUI/releases/latest/download/ComfyUI_windows_portable_nvidia.7z).
2.  Extract with [7-Zip](https://7-zip.org).
3.  Place your Stable Diffusion models (.ckpt/.safetensors) in `ComfyUI\models\checkpoints`.
4.  If blocked, right-click the downloaded file -> Properties -> Unblock.

**Sharing models between another UI and ComfyUI?** See the [Config file](extra_model_paths.yaml.example) to set the search paths for models. Rename this file to extra_model_paths.yaml and edit it with your favorite text editor.

### Manual Install (Windows, Linux):

1.  **Prerequisites:** Python 3.13 is recommended.
2.  **Clone the Repository:** `git clone https://github.com/comfyanonymous/ComfyUI`
3.  **Model Placement:** Place your Stable Diffusion models in `models/checkpoints` and your VAE in `models/vae`.
4.  **AMD GPUs (Linux):** Install ROCm and PyTorch:
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.4
    ```
    (Nightly build may have performance improvements):
        ```bash
        pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.4
        ```
5.  **Intel GPUs (Windows and Linux):**
    *   (Option 1) Install PyTorch xpu:
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
    ```
        (Nightly build may have performance improvements):
        ```bash
        pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/xpu
        ```
    *   (Option 2) Alternatively, leverage IPEX for improved performance via [Installation](https://intel.github.io/intel-extension-for-pytorch/index.html#installation?platform=gpu)
6.  **NVIDIA GPUs:** Install stable PyTorch:
    ```bash
    pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu129
    ```
    (Nightly build may have performance improvements):
    ```bash
    pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu129
    ```
    *   Troubleshooting: If you get the "Torch not compiled with CUDA enabled" error, uninstall and reinstall PyTorch.
7.  **Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
8.  **Other Platforms:** Instructions for Apple Silicon (M1/M2), DirectML (AMD Windows), Ascend NPUs, Cambricon MLUs, and Iluvatar Corex are available in the original README.

## Running ComfyUI:

1.  Navigate to your ComfyUI directory in the terminal.
2.  Run the command: `python main.py`

**AMD card specific tips:**
    * Try the following commands for AMD cards with issues:
        *   For 6700, 6600, and older RDNA2 cards: `HSA_OVERRIDE_GFX_VERSION=10.3.0 python main.py`
        *   For AMD 7600 and RDNA3 cards: `HSA_OVERRIDE_GFX_VERSION=11.0.0 python main.py`
    *   Enable memory efficient attention: `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 python main.py --use-pytorch-cross-attention`
    *   Try setting `PYTORCH_TUNABLEOP_ENABLED=1` (may cause initial slowdown).

## Useful Tips:

*   Only outputs of completed sections are executed.
*   Only parts of the graph that change will re-execute.
*   Drag and drop a generated image onto the webpage to load its workflow.
*   Use `()` for emphasis and `\\()` to escape parentheses.
*   Use `{day|night}` for wildcard prompts and `\\{}` to escape curly braces.
*   Place textual inversion concepts/embeddings in the `models/embeddings` directory.

## How to Show High-Quality Previews:

1.  Use `--preview-method auto` to enable previews.
2.  Install [TAESD](https://github.com/madebyollin/taesd) and place `taesd_decoder.pth`, `taesdxl_decoder.pth`, and `taesd3_decoder.pth`, and `taef1_decoder.pth` in the `models/vae_approx` folder.
3.  Restart and use `--preview-method taesd` to enable high-quality previews.

## How to Use TLS/SSL:

1.  Generate a self-signed certificate with:
    `openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -sha256 -days 3650 -nodes -subj "/C=XX/ST=StateName/L=CityName/O=CompanyName/OU=CompanySectionName/CN=CommonNameOrHostname"`
2.  Run ComfyUI with: `--tls-keyfile key.pem --tls-certfile cert.pem` to enable HTTPS access.

## Support and Development:

*   **Discord:** [https://comfy.org/discord](https://comfy.org/discord)
*   **Matrix:** [#comfyui_space:matrix.org](https://app.element.io/#/room/%23comfyui_space%3Amatrix.org)
*   **Website:** [https://www.comfy.org/](https://www.comfy.org/)

## Frontend Development:

*   **Frontend Repository:** [ComfyUI Frontend](https://github.com/Comfy-Org/ComfyUI_frontend).
*   **Issue Reporting:** Report frontend-specific issues in the frontend repository.
*   **Latest Frontend:** Use `--front-end-version Comfy-Org/ComfyUI_frontend@latest` for the latest daily release.
*   **Specific Versions:** Use `--front-end-version Comfy-Org/ComfyUI_frontend@<version>`
*   **Legacy Frontend:** Use `--front-end-version Comfy-Org/ComfyUI_legacy_frontend@latest`

## GPU Recommendations:

*   Find recommendations on the [Which GPU should I buy for ComfyUI](https://github.com/comfyanonymous/ComfyUI/wiki/Which-GPU-should-I-buy-for-ComfyUI)