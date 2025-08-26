<div align="center">

# ComfyUI: Unleash Your AI Creativity with a Powerful Visual Workflow Engine

**Create stunning AI-generated images, videos, and more using ComfyUI, the most powerful and modular visual AI engine.**

[![Website][website-shield]][website-url]
[![Discord][discord-shield]][discord-url]
[![Twitter][twitter-shield]][twitter-url]
[![Matrix][matrix-shield]][matrix-url]
<br>
[![GitHub Release][github-release-shield]][github-release-link]
[![Release Date][github-release-date-shield]][github-release-link]
[![Total Downloads][github-downloads-shield]][github-downloads-link]
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
[github-downloads-shield]: https://img.shields.io/github/downloads/comfyanonymous/ComfyUI/total?style=flat
[github-downloads-latest-shield]: https://img.shields.io/github/downloads/comfyanonymous/ComfyUI/latest/total?style=flat&label=downloads%40latest
[github-downloads-link]: https://github.com/comfyanonymous/ComfyUI/releases

![ComfyUI Screenshot](https://github.com/user-attachments/assets/7ccaf2c1-9b72-41ae-9a89-5688c94b7abe)
</div>

ComfyUI empowers you to design and execute complex Stable Diffusion pipelines using a user-friendly, node-based interface, with broad compatibility across Windows, Linux, and macOS. Explore the [original repo](https://github.com/comfyanonymous/ComfyUI) for more details.

**Key Features:**

*   **Visual Workflow Design:**  Create intricate AI workflows using an intuitive, node-based interface (graphs/flowcharts) without needing to write any code.
*   **Model Support:**
    *   **Image Generation:** Supports a vast array of image generation models including SD1.x, SD2.x (unCLIP), SDXL, SDXL Turbo, Stable Cascade, SD3, Pixart Alpha/Sigma, AuraFlow, HunyuanDiT, Flux, Lumina Image 2.0, HiDream, and Qwen Image.
    *   **Image Editing:** Features models like Omnigen 2, Flux Kontext, HiDream E1.1, and Qwen Image Edit for advanced image manipulation.
    *   **Video Generation:** Includes models for video creation like Stable Video Diffusion, Mochi, LTX-Video, Hunyuan Video, Wan 2.1, and Wan 2.2.
    *   **Audio Generation:** Supports Stable Audio and ACE Step for audio-related workflows.
    *   **3D Models:** Offers support for Hunyuan3D 2.0.
*   **Optimized Performance:**
    *   **Asynchronous Queue System:** Enables efficient processing of multiple tasks.
    *   **Smart Memory Management:** Supports running large models on GPUs with limited VRAM through intelligent offloading.
    *   **CPU Mode:** Runs even without a dedicated GPU using the `--cpu` flag (albeit more slowly).
*   **Model Compatibility:**
    *   Supports loading checkpoints (ckpt, safetensors), standalone diffusion models, VAEs, and CLIP models.
    *   Supports Embeddings/Textual Inversion, LoRAs (regular, locon, and loha), and Hypernetworks.
*   **Workflow Management:**
    *   Load and save workflows using PNG, WebP, FLAC, and JSON files.
    *   Create complex workflows, including Hires fix and other advanced techniques.
*   **Advanced Features:** Area Composition, Inpainting (with regular and inpainting models), ControlNet and T2I-Adapter, Upscale Models (ESRGAN, SwinIR, etc.), GLIGEN, Model Merging, LCM models/Loras.
*   **Offline Functionality:** Works fully offline, ensuring your privacy.
*   **API Integration:** Optional API nodes for integrating with paid models from external providers via the online [Comfy API](https://docs.comfy.org/tutorials/api-nodes/overview).
*   **High-Quality Previews:** Support for [TAESD](https://github.com/madebyollin/taesd) for enhanced previews.

Get started by exploring the [example workflows](https://comfyanonymous.github.io/ComfyUI_examples/).

## Installation

### Windows Portable

1.  **Download:** [Direct link to download](https://github.com/comfyanonymous/ComfyUI/releases/latest/download/ComfyUI_windows_portable_nvidia.7z)
2.  **Extract:** Use [7-Zip](https://7-zip.org) to extract the contents.
3.  **Run:** Place your Stable Diffusion models in `ComfyUI\models\checkpoints`.
4.  **Troubleshooting:** If you have trouble extracting it, right click the file -> properties -> unblock

### Using comfy-cli
```bash
pip install comfy-cli
comfy install
```

### Manual Installation (Windows & Linux)

1.  **Clone the Repository:** `git clone <repository_url>`
2.  **Model Placement:** Place your model files in the specified directories:
    *   Stable Diffusion checkpoints (ckpt/safetensors): `models/checkpoints`
    *   VAE: `models/vae`
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Platform-Specific Instructions:

*   **AMD GPUs (Linux):** Install the appropriate drivers and PyTorch version for your system.
*   **Intel GPUs (Windows and Linux):** Follow the installation instructions for PyTorch with xpu or Intel Extension for PyTorch (IPEX).
*   **NVIDIA:** Install the appropriate PyTorch version for your system.
*   **Apple Mac silicon:** Follow the installation instructions for Accelerated PyTorch training on Mac to install pytorch nightly and follow the manual installation instructions.
*   **Ascend NPUs:**  Follow the platform-specific instructions on the [Installation](https://ascend.github.io/docs/sources/pytorch/install.html#pytorch) page.
*   **Cambricon MLUs:** Install the Cambricon CNToolkit and PyTorch(torch_mlu).
*   **Iluvatar Corex:** Install the Iluvatar Corex Toolkit.
*   **DirectML (AMD Cards on Windows):** (Not recommended) Install torch-directml and launch with: `python main.py --directml`

## Running

To launch ComfyUI, navigate to your ComfyUI directory in the terminal and run:

```bash
python main.py
```
*To enable experimental memory efficient attention on recent pytorch in ComfyUI on some AMD GPUs using this command, it should already be enabled by default on RDNA3:*
```bash
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 python main.py --use-pytorch-cross-attention
```
*You can also try setting this env variable `PYTORCH_TUNABLEOP_ENABLED=1` which might speed things up at the cost of a very slow initial run.*

## Useful Tips and Tricks

*   **Workflow Execution:**
    *   Only parts of the graph with complete inputs are executed.
    *   Only changed parts of the graph are re-executed.
*   **Workflow Loading:**  Drag and drop generated PNGs to load their complete workflows.
*   **Prompt Emphasis:** Use `(word:1.2)` or `(word:0.8)` for emphasis.
*   **Dynamic Prompts:** Use `{day|night}` for wildcard prompts.
*   **Textual Inversion:** Place embeddings in `models/embeddings`.
*   **High-Quality Previews:** Use `--preview-method taesd` with installed TAESD decoders in `models/vae_approx`
*   **TLS/SSL:** Use `--tls-keyfile key.pem --tls-certfile cert.pem` to enable secure HTTPS access.
*   **AMD ROCm Tips**: Use environment variables to run AMD cards with specific configurations.

## Support and Community

*   [Discord](https://comfy.org/discord): Get help and provide feedback.
*   [Matrix space: #comfyui_space:matrix.org](https://app.element.io/#/room/%23comfyui_space%3Amatrix.org)
*   [https://www.comfy.org/](https://www.comfy.org/)

## Frontend Development

As of August 15, 2024, the frontend is in a separate repository: [ComfyUI Frontend](https://github.com/Comfy-Org/ComfyUI_frontend).

### Reporting Issues and Requesting Features

For any bugs, issues, or feature requests related to the frontend, please use the [ComfyUI Frontend repository](https://github.com/Comfy-Org/ComfyUI_frontend).

### Using the Latest Frontend

1.  **Latest Daily Release:**
    ```
    --front-end-version Comfy-Org/ComfyUI_frontend@latest
    ```

2.  **Specific Version:**
    ```
    --front-end-version Comfy-Org/ComfyUI_frontend@1.2.2
    ```

### Accessing the Legacy Frontend
```
--front-end-version Comfy-Org/ComfyUI_legacy_frontend@latest
```

## FAQ

### Which GPU should I buy for this?

[See this page for some recommendations](https://github.com/comfyanonymous/ComfyUI/wiki/Which-GPU-should-I-buy-for-ComfyUI)