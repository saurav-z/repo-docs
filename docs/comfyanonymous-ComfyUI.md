<div align="center">

# ComfyUI: Unleash Your AI Artistry with a Visual Workflow Engine

**Create stunning AI-generated images, videos, and more with ComfyUI, the powerful and modular visual AI engine.**

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

ComfyUI is a powerful and versatile visual AI engine, ideal for both beginners and experienced users, allowing you to design and execute complex Stable Diffusion workflows through an intuitive node-based interface. 

**Key Features:**

*   **Visual Workflow Creation:** Design intricate Stable Diffusion pipelines using an easy-to-understand graph/node/flowchart interface, eliminating the need for coding.
*   **Extensive Model Support:**  Compatibility with a wide array of models, including:
    *   SD1.x, SD2.x, SDXL, SDXL Turbo, Stable Cascade, SD3/3.5, and more
    *   Image Editing Models:  Omnigen 2, Flux Kontext, HiDream E1.1, Qwen Image Edit.
    *   Video Models: Stable Video Diffusion, Mochi, LTX-Video, and more.
    *   Audio Models: Stable Audio and ACE Step.
    *   3D Models: Hunyuan3D 2.0.
*   **Efficient Processing:** Optimized for speed, only re-executing changed portions of your workflow.
*   **Smart Memory Management:**  Handles large models even on GPUs with limited VRAM, down to 1GB, through smart offloading.
*   **CPU Support:** Operates even without a GPU, using CPU processing (though slower).
*   **Checkpoint Compatibility:** Supports loading ckpt, safetensors, and other model formats.
*   **Advanced Features:**
    *   Embeddings/Textual Inversion, LoRAs, Hypernetworks, Model Merging
    *   Workflow Loading/Saving:  Load and save full workflows from PNG, WebP, and FLAC files.
    *   Nodes Interface: Create complex workflows, like Hires fix, area composition, inpainting, controlnet, and more.
    *   Upscaling:  Integrates with various upscale models (ESRGAN, SwinIR, etc.).
    *   Latent previews with TAESD.
    *   Full Offline Operation.
    *   Optional API nodes (for paid model access via Comfy API).
    *   Configuration file for model paths.
*   **Workflow Examples:** Explore pre-built workflows to jumpstart your creativity:  [Examples](https://comfyanonymous.github.io/ComfyUI_examples/).

**Getting Started:**

*   **Desktop Application:**  The easiest way to get started. Available for Windows and macOS:  [Download](https://www.comfy.org/download).
*   **Windows Portable Package:**  Get the latest commits and complete portability. Available on Windows.
*   **Manual Install:** Supports all operating systems and GPU types (NVIDIA, AMD, Intel, Apple Silicon, Ascend).

**Installation Guides:**

*   **[Windows Portable](#windows-portable)**
    *   Download, extract with [7-Zip](https://7-zip.org) and run. Make sure you put your Stable Diffusion checkpoints/models in: ComfyUI\models\checkpoints
    *   If you have trouble extracting it, right click the file -> properties -> unblock
    *   **How do I share models between another UI and ComfyUI?** See the [Config file](extra_model_paths.yaml.example) to set the search paths for models.
*   **[comfy-cli](https://docs.comfy.org/comfy-cli/getting-started)**
    ```bash
    pip install comfy-cli
    comfy install
    ```
*   **[Manual Install (Windows, Linux)](#manual-install-windows-linux)**

    1.  Clone the repository: `git clone https://github.com/comfyanonymous/ComfyUI`
    2.  Place your SD checkpoints (ckpt/safetensors) in the `models/checkpoints` directory.
    3.  Place your VAE in `models/vae`.
    4.  **GPU-Specific Installation:**
        *   **AMD (Linux):** `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.4` (Stable) or `--pre` for nightly.
        *   **Intel (Windows/Linux):**  Install PyTorch xpu: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu` or IPEX (see install [here](https://intel.github.io/intel-extension-for-pytorch/index.html#installation?platform=gpu)).
        *   **NVIDIA:** `pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu129` (Stable) or `--pre` for nightly.
        *   **Apple Silicon:** Follow the [Accelerated PyTorch training on Mac](https://developer.apple.com/metal/pytorch/) guide.
        *   **Ascend NPUs:** [Installation](https://ascend.github.io/docs/sources/pytorch/install.html#pytorch)
        *   **Cambricon MLUs:** [Installation](https://www.cambricon.com/docs/sdk_1.15.0/cntoolkit_3.7.2/cntoolkit_install_3.7.2/index.html)
        *   **Iluvatar Corex:** [Installation](https://support.iluvatar.com/#/DocumentCentre?id=1&nameCenter=2&productId=520117912052801536)
    5.  Install dependencies: `pip install -r requirements.txt`
    6.  Run ComfyUI:  `python main.py`

**Running:**

*   Execute ComfyUI using: `python main.py`
*   **AMD Card Troubleshooting:** `HSA_OVERRIDE_GFX_VERSION=10.3.0 python main.py` (RDNA2) or `HSA_OVERRIDE_GFX_VERSION=11.0.0 python main.py` (RDNA3)
*   **AMD ROCm Tips:**
    *   `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 python main.py --use-pytorch-cross-attention` (Memory efficient attention).
    *   `PYTORCH_TUNABLEOP_ENABLED=1` (May improve speed, but slower initial run).

**Useful Commands & Tips:**

*   **Drag and drop** PNG files on the webpage to load workflows.
*   Use `()` for emphasis: `(good code:1.2)` or `(bad code:0.8)`.
*   Use `{day|night}` for dynamic prompts:  `{wild|card|test}`.
*   Place textual inversion concepts/embeddings in the `models/embeddings` directory.
*   Run with `--preview-method auto` for previews.
*   To enable higher-quality previews with [TAESD](https://github.com/madebyollin/taesd), download `taesd_decoder.pth`, `taesdxl_decoder.pth`, `taesd3_decoder.pth` and `taef1_decoder.pth` into `models/vae_approx` and use `--preview-method taesd`.
*   Use `--tls-keyfile key.pem --tls-certfile cert.pem` to enable TLS/SSL.

**Shortcuts:**
(A complete list of keyboard shortcuts is available in the original README)

**Release Cycle:** ComfyUI follows a weekly release cycle.
**Frontend:** See [Frontend Development](#frontend-development) section for latest releases.

**Support and Development:**
*   [Discord](https://comfy.org/discord)
*   [Matrix space: #comfyui_space:matrix.org](https://app.element.io/#/room/%23comfyui_space%3Amatrix.org)
*   [ComfyOrg](https://www.comfy.org/)

**Contribute**: [ComfyUI on GitHub](https://github.com/comfyanonymous/ComfyUI)