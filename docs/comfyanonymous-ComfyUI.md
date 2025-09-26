<div align="center">

# ComfyUI: Unleash Your Creative Vision with Visual AI

**ComfyUI is a powerful and modular visual AI engine, giving you unparalleled control over Stable Diffusion and beyond.**

[![Website][website-shield]][website-url]
[![Discord][discord-shield]][discord-url]
[![Twitter][twitter-shield]][twitter-url]
[![Matrix][matrix-shield]][matrix-url]
<br>
[![GitHub Release][github-release-shield]][github-release-link]
[![Release Date][github-release-date-shield]][github-release-link]
[![Downloads][github-downloads-shield]][github-downloads-link]

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
[github-downloads-link]: https://github.com/comfyanonymous/ComfyUI/releases

![ComfyUI Screenshot](https://github.com/user-attachments/assets/7ccaf2c1-9b72-41ae-9a89-5688c94b7abe)
</div>

ComfyUI provides a node-based interface for crafting intricate AI workflows, enabling you to generate stunning visuals with Stable Diffusion and other AI models.  Visit the [original repo](https://github.com/comfyanonymous/ComfyUI) for more details.

## Key Features

*   **Visual Workflow Design:** Create complex AI pipelines using a user-friendly, graph-based interface.
*   **Wide Model Support:**  Comprehensive compatibility with a vast array of models:
    *   SD1.x, SD2.x (including unCLIP)
    *   SDXL, SDXL Turbo
    *   Stable Cascade
    *   SD3 and SD3.5
    *   Pixart Alpha and Sigma
    *   AuraFlow, HunyuanDiT, Flux, Lumina Image 2.0, HiDream, Qwen Image, Hunyuan Image 2.1
*   **Image Editing Models:** Support for advanced image editing:
    *   Omnigen 2, Flux Kontext, HiDream E1.1, Qwen Image Edit.
*   **Video Generation:** Create videos with support for multiple video models:
    *   Stable Video Diffusion, Mochi, LTX-Video, Hunyuan Video, Wan 2.1, Wan 2.2
*   **Audio Generation:**  Generate audio with supported models.
    *   Stable Audio, ACE Step
*   **3D Models:**  Support for 3D model generation.
    *   Hunyuan3D 2.0
*   **Efficiency & Optimization:**
    *   Asynchronous queue system for efficient processing.
    *   Optimized execution: Re-executes only changed parts of workflows.
    *   Smart memory management for running large models on limited VRAM.
    *   CPU fallback option (`--cpu`) for running without a GPU.
*   **Flexible Model Loading:** Load various model formats (ckpt, safetensors, etc.) and support for:
    *   Embeddings/Textual Inversion
    *   LoRAs (regular, locon and loha)
    *   Hypernetworks
*   **Workflow Management:**
    *   Load and save workflows as PNG, WebP, FLAC, and JSON files.
    *   Example workflows available.
*   **Advanced Features:**
    *   Area Composition
    *   Inpainting
    *   ControlNet and T2I-Adapter
    *   Upscale Models (ESRGAN, SwinIR, etc.)
    *   GLIGEN
    *   Model Merging
    *   LCM models and Loras
    *   Latent previews with [TAESD](https://github.com/madebyollin/taesd/)
*   **Offline Functionality:** Operates fully offline by default.
*   **API Integration (Optional):**  Use paid models through external providers via the [Comfy API](https://docs.comfy.org/tutorials/api-nodes/overview).
*   **Customizable:**  Config file (`extra_model_paths.yaml.example`) for setting model search paths.

## Getting Started

Choose your preferred method:

*   **Desktop Application:** ([Download Link](https://www.comfy.org/download)) Easiest for beginners, available on Windows & macOS.
*   **Windows Portable Package:**  Get the latest updates, fully portable, for Windows.
*   **Manual Install:**  Supports all operating systems and GPU types (NVIDIA, AMD, Intel, Apple Silicon, Ascend).

## [Examples](https://comfyanonymous.github.io/ComfyUI_examples/)

Explore what's possible with ComfyUI via the example workflows.

## Installing

### Windows Portable

1.  Download the portable standalone build from the [releases page](https://github.com/comfyanonymous/ComfyUI/releases).
2.  Extract with [7-Zip](https://7-zip.org).
3.  Place your Stable Diffusion checkpoints/models in `ComfyUI\models\checkpoints`.

**Sharing Models:**  Use the `extra_model_paths.yaml.example` file in the ComfyUI directory to specify custom model paths.

### [comfy-cli](https://docs.comfy.org/comfy-cli/getting-started)

```bash
pip install comfy-cli
comfy install
```

### Manual Install (Windows, Linux)

1.  **Clone the Repository:** `git clone this repo`
2.  **Model Placement:** Place your SD checkpoints (`.ckpt` or `.safetensors`) in the `models/checkpoints` directory and your VAEs in `models/vae`.
3.  **AMD GPU (Linux only):** Install the appropriate ROCm and PyTorch version.
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.4
    ```
    or
    ```bash
    pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.4
    ```
4.  **Intel GPU (Windows and Linux):**
    *   **(Option 1)** Install PyTorch with XPU support.
        ```bash
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
        ```
        or
        ```bash
        pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/xpu
        ```
    *   **(Option 2)** Intel Extension for PyTorch (IPEX) - See the [Installation](https://intel.github.io/intel-extension-for-pytorch/index.html#installation?platform=gpu) page for more information.
5.  **NVIDIA:** Install the stable or nightly PyTorch versions.
    ```bash
    pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu129
    ```
    or
    ```bash
    pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu129
    ```
6.  **Troubleshooting:** If you see "Torch not compiled with CUDA enabled", reinstall PyTorch as above.
7.  **Dependencies:**  From your terminal in the ComfyUI folder:
    ```bash
    pip install -r requirements.txt
    ```

#### Others:

*   **Apple Mac silicon:** Install PyTorch nightly, then follow manual install instructions.
*   **DirectML (AMD Cards on Windows):** Not recommended; use unofficial ROCm builds if possible.
    ```bash
    pip install torch-directml
    python main.py --directml
    ```
*   **Ascend NPUs:** Follow the [installation](https://ascend.github.io/docs/sources/ascend/quick_install.html) guide.
*   **Cambricon MLUs:**  Install CNToolkit and PyTorch(torch_mlu).
*   **Iluvatar Corex:**  Install the Iluvatar Corex Toolkit.

## Running

```bash
python main.py
```

### AMD ROCm Tips
```bash
HSA_OVERRIDE_GFX_VERSION=10.3.0 python main.py
```
for older RDNA2 cards
```bash
HSA_OVERRIDE_GFX_VERSION=11.0.0 python main.py
```
for AMD 7600 and maybe other RDNA3 cards
```bash
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 python main.py --use-pytorch-cross-attention
```
```bash
PYTORCH_TUNABLEOP_ENABLED=1
```

## Notes

*   Only parts of the graph with correct inputs are executed.
*   Only changed parts of the graph are re-executed.
*   Drag generated PNG files onto the webpage to load the full workflow.
*   Use parentheses `()` for emphasis (e.g., `(good code:1.2)`).
*   Use curly braces `{}` for wildcard/dynamic prompts (e.g., `{wild|card|test}`).
*   Use textual inversion concepts/embeddings from the `models/embeddings` directory.

## How to show high-quality previews?

*   Enable with `--preview-method auto`.
*   Install [TAESD](https://github.com/madebyollin/taesd) and place the decoders in the `models/vae_approx` folder for higher-quality previews.
*   Restart ComfyUI and use `--preview-method taesd`

## How to use TLS/SSL?

*   Generate a self-signed certificate: `openssl req -x509 -newkey rsa:4096 ...`
*   Enable with `--tls-keyfile key.pem --tls-certfile cert.pem`.  Access the app via `https://...`.

## Support and dev channel

*   [Discord](https://comfy.org/discord)
*   [Matrix space](https://app.element.io/#/room/%23comfyui_space%3Amatrix.org)
*   [Comfy.org](https://www.comfy.org/)

## Frontend Development

*   The new frontend is now in the separate [ComfyUI Frontend repository](https://github.com/Comfy-Org/ComfyUI_frontend).
*   Use `--front-end-version Comfy-Org/ComfyUI_frontend@latest` or specify a version for the latest frontend.
*   For the legacy frontend use `--front-end-version Comfy-Org/ComfyUI_legacy_frontend@latest`

## QA

*   GPU recommendations:  [See this page for some recommendations](https://github.com/comfyanonymous/ComfyUI/wiki/Which-GPU-should-I-buy-for-ComfyUI)