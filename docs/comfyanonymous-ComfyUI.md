<div align="center">

# ComfyUI: Unleash Your AI Artistry with a Visual Workflow Engine

**ComfyUI is a powerful and modular visual AI engine, revolutionizing AI art generation through its intuitive node-based interface.**

[![Website](https://img.shields.io/badge/ComfyOrg-4285F4?style=flat)][website-url]
[![Discord](https://img.shields.io/badge/Discord-Join-green)][discord-url]
[![Twitter](https://img.shields.io/twitter/follow/ComfyUI)][twitter-url]
[![Matrix](https://img.shields.io/badge/Matrix-Chat-000000?style=flat&logo=matrix&logoColor=white)][matrix-url]
[![Release](https://img.shields.io/github/v/release/comfyanonymous/ComfyUI?style=flat&sort=semver)][github-release-link]
[![Release Date](https://img.shields.io/github/release-date/comfyanonymous/ComfyUI?style=flat)][github-release-link]
[![Downloads](https://img.shields.io/github/downloads/comfyanonymous/ComfyUI/total?style=flat)][github-downloads-link]
[![Latest Downloads](https://img.shields.io/github/downloads/comfyanonymous/ComfyUI/latest/total?style=flat&label=downloads%40latest)][github-downloads-link]

[website-url]: https://www.comfy.org/
[discord-url]: https://www.comfy.org/discord
[twitter-url]: https://x.com/ComfyUI
[matrix-url]: https://app.element.io/#/room/%23comfyui_space%3Amatrix.org
[github-release-link]: https://github.com/comfyanonymous/ComfyUI/releases
[github-downloads-link]: https://github.com/comfyanonymous/ComfyUI/releases

<img src="https://github.com/user-attachments/assets/7ccaf2c1-9b72-41ae-9a89-5688c94b7abe" alt="ComfyUI Screenshot">

</div>

ComfyUI empowers you to design and execute complex Stable Diffusion pipelines using a user-friendly graph/nodes/flowchart-based interface, offering unparalleled flexibility and control for AI image and video generation.

**Key Features:**

*   **Node-Based Workflow:** Visually create and experiment with intricate Stable Diffusion workflows without coding.
*   **Extensive Model Support:**
    *   SD1.x, SD2.x, SDXL, SDXL Turbo, Stable Cascade, SD3 and SD3.5, Pixart Alpha and Sigma, AuraFlow, HunyuanDiT, Flux, Lumina Image 2.0, HiDream, Qwen Image, Hunyuan Image 2.1 and more.
    *   Image Editing Models: Omnigen 2, Flux Kontext, HiDream E1.1, Qwen Image Edit.
    *   Video Models: Stable Video Diffusion, Mochi, LTX-Video, Hunyuan Video, Wan 2.1, Wan 2.2.
    *   Audio Models: Stable Audio, ACE Step.
    *   3D Models: Hunyuan3D 2.0
*   **Asynchronous Queue:** Manage and execute multiple workflows efficiently.
*   **Memory Optimization:** Smart memory management for running large models on GPUs with limited VRAM.
*   **CPU Support:** Run ComfyUI even without a dedicated GPU using the `--cpu` flag.
*   **Checkpoint & Model Loading:** Load ckpt, safetensors, and various other model formats.
*   **Advanced Techniques:** Support for Embeddings/Textual inversion, LoRAs, Hypernetworks, ControlNet, T2I-Adapter, Upscale Models, GLIGEN, Model Merging, LCM models and Loras, and more.
*   **Workflow Handling:** Load and save workflows as PNG, WebP, FLAC, and JSON files.
*   **Offline Functionality:** Core functionality operates fully offline.
*   **API Integration:** Optional API nodes for accessing paid models from external providers.
*   **Customization:** Utilize a [Config file](extra_model_paths.yaml.example) to set model search paths.

**Getting Started:**

*   **[Desktop Application](https://www.comfy.org/download):** Simplest way to get started, available on Windows & macOS.
*   **[Windows Portable Package](https://github.com/comfyanonymous/ComfyUI/releases):** Get the latest commits, completely portable for Windows.
*   **[Manual Install](#manual-install-windows-linux):** Supports all operating systems and GPU types (NVIDIA, AMD, Intel, Apple Silicon, Ascend).

**Explore Examples:**

Discover the power of ComfyUI with [example workflows](https://comfyanonymous.github.io/ComfyUI_examples/).

**Release Process:**

ComfyUI follows a weekly release cycle with the following repositories:

1.  **[ComfyUI Core](https://github.com/comfyanonymous/ComfyUI)** - Releases new stable versions.
2.  **[ComfyUI Desktop](https://github.com/Comfy-Org/desktop)** - Builds releases using the latest stable core version.
3.  **[ComfyUI Frontend](https://github.com/Comfy-Org/ComfyUI_frontend)** - Hosts frontend updates.

**Shortcuts**

See the [shortcuts section](https://github.com/comfyanonymous/ComfyUI#shortcuts) for keyboard shortcuts.

**Installation Instructions**

*   **Windows Portable:**
    1.  Download and extract from the [releases page](https://github.com/comfyanonymous/ComfyUI/releases).
    2.  Place model files in `ComfyUI\models\checkpoints`.
    3.  See [config file](extra_model_paths.yaml.example) to manage models.
*   **comfy-cli:**

    ```bash
    pip install comfy-cli
    comfy install
    ```
*   **Manual Install (Windows, Linux):**
    1.  Clone the repository.
    2.  Place SD checkpoints and VAE files in the appropriate `models` directories.
    3.  Install dependencies: `pip install -r requirements.txt`.
    4.  Follow specific instructions for your GPU: [AMD GPUs (Linux only)](#amd-gpus-linux-only), [Intel GPUs (Windows and Linux)](#intel-gpus-windows-and-linux), [NVIDIA](#nvidia), [Apple Mac silicon](#apple-mac-silicon), [DirectML (AMD Cards on Windows)](#directml-amd-cards-on-windows), [Ascend NPUs](#ascend-npus), [Cambricon MLUs](#cambricon-mlus), [Iluvatar Corex](#iluvatar-corex)

**Running**

```bash
python main.py
```

See [running tips](https://github.com/comfyanonymous/ComfyUI#running) for specific GPU settings.

**Notes:**

*   ComfyUI executes only parts of the graph that have all inputs and have been changed.
*   Drag and drop generated images to load their workflow.
*   Use `()` for emphasis and `{}` for dynamic prompts.
*   Place textual inversion concepts/embeddings in the `models/embeddings` directory.

**How to show high-quality previews?**

Use the command line argument: ```--preview-method auto```

To enable higher-quality previews with [TAESD](https://github.com/madebyollin/taesd), download the [taesd_decoder.pth, taesdxl_decoder.pth, taesd3_decoder.pth and taef1_decoder.pth](https://github.com/madebyollin/taesd/) and place them in the `models/vae_approx` folder. Once they're installed, restart ComfyUI and launch it with `--preview-method taesd` to enable high-quality previews.

**How to use TLS/SSL?**

Generate a self-signed certificate (not appropriate for shared/production use) and key by running the command: `openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -sha256 -days 3650 -nodes -subj "/C=XX/ST=StateName/L=CityName/O=CompanyName/OU=CompanySectionName/CN=CommonNameOrHostname"`

Use `--tls-keyfile key.pem --tls-certfile cert.pem` to enable TLS/SSL.

**Support and Community:**

*   [Discord](https://comfy.org/discord)
*   [Matrix space: #comfyui_space:matrix.org](https://app.element.io/#/room/%23comfyui_space%3Amatrix.org)
*   [https://www.comfy.org/](https://www.comfy.org/)

**Frontend Development**

The new frontend can be found [ComfyUI Frontend repository](https://github.com/Comfy-Org/ComfyUI_frontend).

To use the most up-to-date frontend version:

```
--front-end-version Comfy-Org/ComfyUI_frontend@latest
```

or for a specific version:

```
--front-end-version Comfy-Org/ComfyUI_frontend@1.2.2
```

**QA**

*   [Which GPU should I buy for this?](https://github.com/comfyanonymous/ComfyUI/wiki/Which-GPU-should-I-buy-for-ComfyUI)

**[Back to Top](#comfyui-unleash-your-ai-artistry-with-a-visual-workflow-engine)**