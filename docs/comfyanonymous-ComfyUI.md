<div align="center">

# ComfyUI: Unleash the Power of Visual AI

**ComfyUI is a revolutionary node-based interface for crafting complex and stunning AI image, video, and audio workflows, offering unparalleled customization and control.**

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

ComfyUI empowers users to create and execute sophisticated Stable Diffusion pipelines through an intuitive graph/node/flowchart interface. Available on Windows, Linux, and macOS. For the latest updates and more information, visit the [original ComfyUI repository](https://github.com/comfyanonymous/ComfyUI).

## Key Features

*   **Node-Based Workflow:** Design and execute complex Stable Diffusion workflows using a visual, node-based interface, eliminating the need for coding.
*   **Extensive Model Support:** Compatible with a vast range of image, image editing, video, and audio models, including:
    *   SD1.x, SD2.x, SDXL, SDXL Turbo, Stable Cascade, SD3/SD3.5, and more.
    *   Image Editing models: Omnigen 2, Flux Kontext, HiDream E1.1, Qwen Image Edit, and more.
    *   Video models: Stable Video Diffusion, Mochi, LTX-Video, and more.
    *   Audio models: Stable Audio, ACE Step, and more.
    *   3D Models: Hunyuan3D 2.0
*   **Asynchronous Queue System:** Efficiently manage and process multiple tasks.
*   **Optimized Performance:** Benefit from optimizations like only re-executing changed parts of the workflow and smart memory management.
*   **Broad Hardware Compatibility:** Works with and without a GPU, supporting NVIDIA, AMD, Intel, and Apple Silicon.
*   **Model Loading:** Load ckpt and safetensors, embeddings, LoRAs, and hypernetworks.
*   **Workflow Management:** Load, save, and share workflows as JSON files or from generated image files.
*   **Advanced Techniques:** Utilize features like Hires fix, inpainting, ControlNet, upscaling models, and more.
*   **Offline Functionality:** Works fully offline.
*   **API Integration:** Optional API nodes to leverage paid models from external providers via the online [Comfy API](https://docs.comfy.org/tutorials/api-nodes/overview).
*   **Customization:** Customize with a [Config file](extra_model_paths.yaml.example) to set model search paths.

## Get Started

Choose the best option for your setup:

*   **Desktop Application:** [Download](https://www.comfy.org/download) for Windows & macOS - The easiest way to get started.
*   **Windows Portable Package:** Get the latest commits, and fully portable. - [Releases](https://github.com/comfyanonymous/ComfyUI/releases).
*   **Manual Install:** Supports all operating systems and GPU types - follow the instructions below.

## [Examples](https://comfyanonymous.github.io/ComfyUI_examples/)

Explore the possibilities with ComfyUI through comprehensive [example workflows](https://comfyanonymous.github.io/ComfyUI_examples/).

## Installation

### Windows Portable

Download, extract, and run from the [releases page](https://github.com/comfyanonymous/ComfyUI/releases).
*   [Direct link to download](https://github.com/comfyanonymous/ComfyUI/releases/latest/download/ComfyUI_windows_portable_nvidia.7z)
*   Place your Stable Diffusion checkpoints/models in: `ComfyUI\models\checkpoints`.

### [comfy-cli](https://docs.comfy.org/comfy-cli/getting-started)

Install and start ComfyUI using comfy-cli:
```bash
pip install comfy-cli
comfy install
```

### Manual Install (Windows, Linux)

1.  **Prerequisites:**
    *   Python 3.13 is recommended. If you encounter issues, try 3.12.
    *   Git is required.
2.  **Clone the Repository:**
    ```bash
    git clone https://github.com/comfyanonymous/ComfyUI.git
    cd ComfyUI
    ```
3.  **Model Placement:**
    *   Place your SD checkpoints (ckpt/safetensors) in `models/checkpoints`.
    *   Place your VAE in `models/vae`.
4.  **GPU-Specific Instructions:**
    *   **AMD (Linux):** Install `rocm` and `pytorch` (stable):
        ```bash
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.4
        ```
        Or install `pytorch` nightly:
        ```bash
        pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.4
        ```
    *   **Intel (Windows and Linux):**
        *   **(Option 1) PyTorch xpu:**
            ```bash
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
            ```
             Or install `pytorch` nightly:
            ```bash
            pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/xpu
            ```
        *   **(Option 2) Intel Extension for PyTorch (IPEX):** Visit [Installation](https://intel.github.io/intel-extension-for-pytorch/index.html#installation?platform=gpu).
    *   **NVIDIA:** Install PyTorch (stable):
        ```bash
        pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu129
        ```
        Or install `pytorch` nightly:
        ```bash
        pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu129
        ```
    *   **Troubleshooting:** If you encounter a CUDA-related error, reinstall PyTorch.
5.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
6.  **Others:**
    *   **Apple Mac silicon:** Follow instructions in the original readme.
    *   **DirectML (AMD on Windows):**
        ```bash
        pip install torch-directml
        python main.py --directml
        ```
    *   **Ascend NPUs:** Follow instructions in the original readme.
    *   **Cambricon MLUs:** Follow instructions in the original readme.
    *   **Iluvatar Corex:** Follow instructions in the original readme.

## Running

```bash
python main.py
```

### AMD Card Tips
```bash
#For 6700, 6600 and maybe other RDNA2 or older:
HSA_OVERRIDE_GFX_VERSION=10.3.0 python main.py
#For AMD 7600 and maybe other RDNA3 cards:
HSA_OVERRIDE_GFX_VERSION=11.0.0 python main.py
#For memory efficient attention on AMD GPUs (test):
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 python main.py --use-pytorch-cross-attention
#Try setting env variable (test)
PYTORCH_TUNABLEOP_ENABLED=1
```

## Notes
*   Only parts of the graph that have an output with all the correct inputs will be executed.
*   Only parts of the graph that change from each execution to the next will be executed, if you submit the same graph twice only the first will be executed. If you change the last part of the graph only the part you changed and the part that depends on it will be executed.
*   Dragging a generated png on the webpage or loading one will give you the full workflow including seeds that were used to create it.
*   You can use () to change emphasis of a word or phrase like: (good code:1.2) or (bad code:0.8). The default emphasis for () is 1.1. To use () characters in your actual prompt escape them like \\( or \\).
*   You can use {day|night}, for wildcard/dynamic prompts. With this syntax "{wild|card|test}" will be randomly replaced by either "wild", "card" or "test" by the frontend every time you queue the prompt. To use {} characters in your actual prompt escape them like: \\{ or \\}.
*   Dynamic prompts also support C-style comments, like `// comment` or `/* comment */`.
*   To use a textual inversion concepts/embeddings in a text prompt put them in the models/embeddings directory and use them in the CLIPTextEncode node like this (you can omit the .pt extension):

```embedding:embedding_filename.pt```

## How to show high-quality previews?

Enable previews using `--preview-method auto`.

*   Install TAESD: Place `taesd_decoder.pth`, `taesdxl_decoder.pth`, `taesd3_decoder.pth`, and `taef1_decoder.pth` from [TAESD](https://github.com/madebyollin/taesd/) in the `models/vae_approx` folder.
*   Restart ComfyUI and launch with `--preview-method taesd` for high-quality previews.

## How to use TLS/SSL?

Generate a self-signed certificate:
`openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -sha256 -days 3650 -nodes -subj "/C=XX/ST=StateName/L=CityName/O=CompanyName/OU=CompanySectionName/CN=CommonNameOrHostname"`

Enable TLS/SSL: `--tls-keyfile key.pem --tls-certfile cert.pem` (access via `https://...`).

## Support and Dev Channels

*   [Discord](https://comfy.org/discord)
*   [Matrix space: #comfyui_space:matrix.org](https://app.element.io/#/room/%23comfyui_space%3Amatrix.org)

## Frontend Development

The new frontend is hosted in the separate [ComfyUI Frontend repository](https://github.com/Comfy-Org/ComfyUI_frontend).

*   **Reporting Issues:** Use the [ComfyUI Frontend repository](https://github.com/Comfy-Org/ComfyUI_frontend) for frontend-related issues.
*   **Latest Frontend:** Launch ComfyUI with `--front-end-version Comfy-Org/ComfyUI_frontend@latest` for the latest daily release or `--front-end-version Comfy-Org/ComfyUI_frontend@<version>` for a specific version.
*   **Legacy Frontend:** Use `--front-end-version Comfy-Org/ComfyUI_legacy_frontend@latest`.

## QA

### Which GPU should I buy for this?

[See this page for some recommendations](https://github.com/comfyanonymous/ComfyUI/wiki/Which-GPU-should-I-buy-for-ComfyUI)