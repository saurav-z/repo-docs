<div align="center">

# ComfyUI: The Most Powerful and Modular Visual AI Engine

**Create stunning AI-generated images, videos, and more with ComfyUI's intuitive node-based interface.**

[![Website](https://img.shields.io/badge/ComfyOrg-4285F4?style=flat)][website-url]
[![Discord](https://img.shields.io/badge/Discord-green?style=flat&logo=discord&logoColor=white&label=Discord)][discord-url]
[![Twitter](https://img.shields.io/twitter/follow/ComfyUI)][twitter-url]
[![Matrix](https://img.shields.io/badge/Matrix-000000?style=flat&logo=matrix&logoColor=white)][matrix-url]
<br>
[![GitHub Release](https://img.shields.io/github/release/comfyanonymous/ComfyUI?style=flat&sort=semver)][github-release-link]
[![Release Date](https://img.shields.io/github/release-date/comfyanonymous/ComfyUI?style=flat)][github-release-link]
[![Downloads](https://img.shields.io/github/downloads/comfyanonymous/ComfyUI/total?style=flat)][github-downloads-link]
[![Latest Downloads](https://img.shields.io/github/downloads/comfyanonymous/ComfyUI/latest/total?style=flat&label=downloads%40latest)][github-downloads-link]

[matrix-url]: https://app.element.io/#/room/%23comfyui_space%3Amatrix.org
[website-url]: https://www.comfy.org/
[discord-url]: https://www.comfy.org/discord
[twitter-url]: https://x.com/ComfyUI
[github-release-link]: https://github.com/comfyanonymous/ComfyUI/releases
[github-downloads-link]: https://github.com/comfyanonymous/ComfyUI/releases

![ComfyUI Screenshot](https://github.com/user-attachments/assets/7ccaf2c1-9b72-41ae-9a89-5688c94b7abe)

</div>

ComfyUI is a powerful and flexible visual AI engine, empowering you to design and execute intricate Stable Diffusion pipelines through a user-friendly, graph-based interface. It's available on Windows, Linux, and macOS, offering unparalleled control and customization for AI image and video generation. **[Visit the original repository for ComfyUI here.](https://github.com/comfyanonymous/ComfyUI)**

## Key Features

*   **Node-Based Workflow:** Visually design complex Stable Diffusion workflows using a graph/nodes/flowchart interface, eliminating the need for code.
*   **Extensive Model Support:**
    *   SD1.x, SD2.x (including [unCLIP](https://comfyanonymous.github.io/ComfyUI_examples/unclip/))
    *   SDXL, SDXL Turbo
    *   Stable Cascade
    *   SD3 and SD3.5
    *   Pixart Alpha and Sigma
    *   AuraFlow, HunyuanDiT, Flux, Lumina Image 2.0, HiDream, Qwen Image, Hunyuan Image 2.1, etc.
*   **Image Editing Models:**
    *   Omnigen 2, Flux Kontext, HiDream E1.1, Qwen Image Edit, etc.
*   **Video Generation Models:**
    *   Stable Video Diffusion, Mochi, LTX-Video, Hunyuan Video, Wan 2.1 and Wan 2.2, etc.
*   **Audio Models:**
    *   Stable Audio, ACE Step, etc.
*   **3D Model Support:**
    *   Hunyuan3D 2.0
*   **Performance & Optimization:**
    *   Asynchronous queue system for efficient processing.
    *   Optimized for memory management and can run on GPUs with as little as 1GB VRAM.
    *   CPU fallback option (`--cpu`) for users without GPUs.
*   **Model Loading:**
    *   Supports ckpt and safetensors formats.
    *   Safe loading of various model file types (ckpt, pt, pth, etc.).
    *   Embeddings/Textual inversion, LoRAs (regular, locon and loha), and Hypernetworks.
*   **Workflow Management:**
    *   Loads full workflows from generated PNG, WebP, and FLAC files.
    *   Saves/Loads workflows as JSON files.
*   **Advanced Features:**
    *   Area Composition.
    *   Inpainting with various models.
    *   ControlNet and T2I-Adapter.
    *   Upscale Models (ESRGAN, SwinIR, etc.).
    *   GLIGEN.
    *   Model Merging.
    *   LCM models and LoRAs.
    *   Latent previews with [TAESD](#how-to-show-high-quality-previews).
*   **Offline Functionality:** Core functionality operates fully offline.
*   **API Integration:** Optional API nodes for accessing paid models from external providers via the [Comfy API](https://docs.comfy.org/tutorials/api-nodes/overview).
*   **Configurable Model Paths:** Customize model search paths with the [Config file](extra_model_paths.yaml.example).

## Getting Started

Choose your preferred installation method:

*   **Desktop Application:** The easiest way to get started; available for Windows and macOS.
    *   Download: [Desktop Application](https://www.comfy.org/download)
*   **Windows Portable Package:** Get the latest commits; fully portable and for Windows.
    *   Download: [Direct link to download](https://github.com/comfyanonymous/ComfyUI/releases/latest/download/ComfyUI_windows_portable_nvidia.7z)
*   **Manual Install:** Supports all operating systems and GPU types (NVIDIA, AMD, Intel, Apple Silicon, Ascend).
    *   See the [Manual Install](#manual-install-windows-linux) instructions below.
*   **comfy-cli** (install and start ComfyUI)
    *   See the [comfy-cli](https://docs.comfy.org/comfy-cli/getting-started) instructions

## Manual Install (Windows, Linux)

1.  **Prerequisites:**
    *   Python 3.13 is recommended (3.12 is also supported if you have issues with custom node dependencies)
    *   Git (for cloning the repository)
2.  **Clone the Repository:**
    ```bash
    git clone https://github.com/comfyanonymous/ComfyUI.git
    cd ComfyUI
    ```
3.  **Model Placement:**
    *   Place your Stable Diffusion checkpoints (ckpt/safetensors files) in `models/checkpoints`.
    *   Place your VAE files in `models/vae`.
4.  **AMD GPUs (Linux only):** Install the appropriate PyTorch packages based on your ROCm version.
    *   Stable Version: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.4`
    *   Nightly Version (may have performance improvements): `pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.4`
5.  **Intel GPUs (Windows and Linux):** Choose one of the options for PyTorch installation.
    *   **Option 1 (Intel Arc GPUs):** Install PyTorch with xpu support.
        *   Stable: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu`
        *   Nightly (may have performance improvements): `pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/xpu`
    *   **Option 2 (Intel Extension for PyTorch):** Use Intel Extension for PyTorch (IPEX) for improved performance.  See [Installation](https://intel.github.io/intel-extension-for-pytorch/index.html#installation?platform=gpu) for instructions.
6.  **NVIDIA GPUs:** Install the appropriate PyTorch packages.
    *   Stable: `pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu129`
    *   Nightly (may have performance improvements): `pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu129`
7.  **Troubleshooting:** If you encounter "Torch not compiled with CUDA enabled," reinstall PyTorch:
    ```bash
    pip uninstall torch
    pip install <your_pytorch_command_from_above>
    ```
8.  **Dependencies:** Install required packages.
    ```bash
    pip install -r requirements.txt
    ```
9.  **Other Installation Guides**
    *   Apple Mac silicon [here](#apple-mac-silicon)
    *   DirectML (AMD Cards on Windows) [here](#directml-amd-cards-on-windows)
    *   Ascend NPUs [here](#ascend-npu)
    *   Cambricon MLUs [here](#cambricon-mlus)
    *   Iluvatar Corex [here](#iluvatar-corex)
10. **Running ComfyUI:**

    ```bash
    python main.py
    ```

## Running

After installation, start ComfyUI with:

```bash
python main.py
```

### AMD ROCm Tips

*   If you have issues on AMD cards, try these commands:

    *   For older RDNA2 cards: `HSA_OVERRIDE_GFX_VERSION=10.3.0 python main.py`
    *   For RDNA3 cards: `HSA_OVERRIDE_GFX_VERSION=11.0.0 python main.py`
*   Enable experimental memory-efficient attention for potential speed increases on recent PyTorch/AMD GPUs:
    ```bash
    TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 python main.py --use-pytorch-cross-attention
    ```
*   You can also try setting `PYTORCH_TUNABLEOP_ENABLED=1` to see if it improves speeds (may be slow on initial run).

## Notes

*   Only the active parts of the graph are executed, ensuring efficiency.
*   Changes only are re-executed from run to run.
*   Drag and drop generated images to load the full workflow.
*   Use `()` for emphasis (e.g., `(good code:1.2)`).
*   Use `{day|night}` for dynamic prompts.
*   Textual Inversion concepts/embeddings go in `models/embeddings`.

## How to show high-quality previews?

*   Use `--preview-method auto` to enable previews.
*   For higher-quality previews with [TAESD](https://github.com/madebyollin/taesd), download the TAESD decoder files from the linked GitHub and place them in the `models/vae_approx` folder. Then, restart ComfyUI and launch with `--preview-method taesd`.

## How to use TLS/SSL?

*   Generate a self-signed certificate and key using `openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -sha256 -days 3650 -nodes -subj "/C=XX/ST=StateName/L=CityName/O=CompanyName/OU=CompanySectionName/CN=CommonNameOrHostname"`
*   Enable TLS/SSL with `--tls-keyfile key.pem --tls-certfile cert.pem`.  Access the app via `https://...`.

## Support and Community

*   [Discord](https://comfy.org/discord): Find help and provide feedback in the #help or #feedback channels.
*   [Matrix space: #comfyui_space:matrix.org](https://app.element.io/#/room/%23comfyui_space%3Amatrix.org)
*   [Website](https://www.comfy.org/)

## Frontend Development

As of August 15, 2024, the frontend is hosted in a separate repository: [ComfyUI Frontend](https://github.com/Comfy-Org/ComfyUI_frontend).

### Reporting Issues and Requesting Features

Report frontend-related issues and feature requests in the [ComfyUI Frontend repository](https://github.com/Comfy-Org/ComfyUI_frontend).

### Using the Latest Frontend

*   To use the latest daily release:
    ```
    --front-end-version Comfy-Org/ComfyUI_frontend@latest
    ```
*   To use a specific version:
    ```
    --front-end-version Comfy-Org/ComfyUI_frontend@1.2.2
    ```

### Accessing the Legacy Frontend

Use the legacy frontend with:
```
--front-end-version Comfy-Org/ComfyUI_legacy_frontend@latest
```

## QA

### Which GPU should I buy for this?

[See this page for some recommendations](https://github.com/comfyanonymous/ComfyUI/wiki/Which-GPU-should-I-buy-for-ComfyUI)