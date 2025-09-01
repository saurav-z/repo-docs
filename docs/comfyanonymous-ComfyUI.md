<div align="center">

# ComfyUI: Unleash Your AI Creativity with a Visual Workflow Engine

**Design and execute advanced Stable Diffusion pipelines with the most powerful and modular visual AI engine.** ([Original Repo](https://github.com/comfyanonymous/ComfyUI))

[![Website][website-shield]][website-url]
[![Discord][discord-shield]][discord-url]
[![Twitter][twitter-shield]][twitter-url]
[![Matrix][matrix-shield]][matrix-url]
<br>
[![GitHub Release][github-release-shield]][github-release-link]
[![Release Date][github-release-date-shield]][github-release-link]
[![Downloads][github-downloads-shield]][github-downloads-link]
[![Latest Download][github-downloads-latest-shield]][github-downloads-link]

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

## Key Features

*   **Visual Workflow Interface:** Design intricate Stable Diffusion pipelines using a node-based, flowchart interface, eliminating the need for coding.
*   **Broad Model Support:** Compatible with a wide range of image, image editing, video, and audio models, including:
    *   **Image Models:** SD1.x, SD2.x, SDXL, SDXL Turbo, Stable Cascade, SD3/SD3.5, Pixart, AuraFlow, HunyuanDiT, Flux, Lumina Image 2.0, HiDream, Qwen Image.
    *   **Image Editing Models:** Omnigen 2, Flux Kontext, HiDream E1.1, Qwen Image Edit.
    *   **Video Models:** Stable Video Diffusion, Mochi, LTX-Video, Hunyuan Video, Wan 2.1/2.2.
    *   **Audio Models:** Stable Audio, ACE Step.
    *   **3D Models:** Hunyuan3D 2.0.
*   **Efficient Processing:** Leverages an asynchronous queue system and smart optimizations, executing only necessary parts of the workflow.
*   **Resource Optimization:** Includes smart memory management and supports running on low-VRAM GPUs (as low as 1GB) with intelligent offloading and CPU fallback.
*   **Model Compatibility:** Loads ckpt, safetensors, and other model file formats.
*   **Advanced Features:**
    *   Embeddings/Textual Inversion
    *   LoRAs (regular, locon, loha)
    *   Hypernetworks
    *   Workflow loading from PNG, WebP, and FLAC files
    *   Workflow saving/loading as JSON
    *   Hires fix and advanced workflows
    *   Area Composition and Inpainting
    *   ControlNet and T2I-Adapter
    *   Upscale Models (ESRGAN, SwinIR, etc.)
    *   GLIGEN
    *   Model Merging
    *   LCM models and LoRAs
    *   Latent previews with TAESD
    *   Optional API nodes for external model providers.
    *   Configuration file for model search paths.
*   **Offline Functionality:** Works fully offline; no automatic downloads unless desired.

## Getting Started

ComfyUI offers several installation methods:

*   **[Desktop Application](https://www.comfy.org/download)**: Easiest to get started with, available on Windows & macOS.
*   **[Windows Portable Package](#installing)**: Get the latest commits in a completely portable package. Available on Windows.
*   **[Manual Install](#manual-install-windows-linux)**: Supports all operating systems, GPU types (NVIDIA, AMD, Intel, Apple Silicon, Ascend).
*   **[comfy-cli](https://docs.comfy.org/comfy-cli/getting-started)**: Install and start ComfyUI using the command line.

## Examples

Explore what's possible with ComfyUI by checking out the [example workflows](https://comfyanonymous.github.io/ComfyUI_examples/).

##  Release Process

*   **Core:** Releases a new stable version (e.g., v0.7.0)
*   **Desktop:** Builds a new release using the latest stable core version
*   **Frontend:** Weekly frontend updates are merged into the core repository

## Shortcuts

| Keybind                            | Explanation                                                                                                        |
|------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| `Ctrl` + `Enter`                      | Queue up current graph for generation                                                                              |
| `Ctrl` + `Shift` + `Enter`              | Queue up current graph as first for generation                                                                     |
| `Ctrl` + `Alt` + `Enter`                | Cancel current generation                                                                                          |
| `Ctrl` + `Z`/`Ctrl` + `Y`                 | Undo/Redo                                                                                                          |
| `Ctrl` + `S`                          | Save workflow                                                                                                      |
| `Ctrl` + `O`                          | Load workflow                                                                                                      |
| `Ctrl` + `A`                          | Select all nodes                                                                                                   |
| `Alt `+ `C`                           | Collapse/uncollapse selected nodes                                                                                 |
| `Ctrl` + `M`                          | Mute/unmute selected nodes                                                                                         |
| `Ctrl` + `B`                           | Bypass selected nodes (acts like the node was removed from the graph and the wires reconnected through)            |
| `Delete`/`Backspace`                   | Delete selected nodes                                                                                              |
| `Ctrl` + `Backspace`                   | Delete the current graph                                                                                           |
| `Space`                              | Move the canvas around when held and moving the cursor                                                             |
| `Ctrl`/`Shift` + `Click`                 | Add clicked node to selection                                                                                      |
| `Ctrl` + `C`/`Ctrl` + `V`                  | Copy and paste selected nodes (without maintaining connections to outputs of unselected nodes)                     |
| `Ctrl` + `C`/`Ctrl` + `Shift` + `V`          | Copy and paste selected nodes (maintaining connections from outputs of unselected nodes to inputs of pasted nodes) |
| `Shift` + `Drag`                       | Move multiple selected nodes at the same time                                                                      |
| `Ctrl` + `D`                           | Load default graph                                                                                                 |
| `Alt` + `+`                          | Canvas Zoom in                                                                                                     |
| `Alt` + `-`                          | Canvas Zoom out                                                                                                    |
| `Ctrl` + `Shift` + LMB + Vertical drag | Canvas Zoom in/out                                                                                                 |
| `P`                                  | Pin/Unpin selected nodes                                                                                           |
| `Ctrl` + `G`                           | Group selected nodes                                                                                               |
| `Q`                                 | Toggle visibility of the queue                                                                                     |
| `H`                                  | Toggle visibility of history                                                                                       |
| `R`                                  | Refresh graph                                                                                                      |
| `F`                                  | Show/Hide menu                                                                                                      |
| `.`                                  | Fit view to selection (Whole graph when nothing is selected)                                                        |
| Double-Click LMB                   | Open node quick search palette                                                                                     |
| `Shift` + Drag                       | Move multiple wires at once                                                                                        |
| `Ctrl` + `Alt` + LMB                   | Disconnect all wires from clicked slot                                                                             |

`Ctrl` can also be replaced with `Cmd` instead for macOS users

## Installation

### Windows Portable

*   Download, extract, and run from the [releases page](https://github.com/comfyanonymous/ComfyUI/releases).
*   Place Stable Diffusion checkpoints in `ComfyUI\models\checkpoints`.
*   [Config file](extra_model_paths.yaml.example) for setting model search paths in the standalone windows build, located in the ComfyUI directory. Rename this file to `extra_model_paths.yaml`

### comfy-cli

```bash
pip install comfy-cli
comfy install
```

### Manual Installation (Windows, Linux)

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/comfyanonymous/ComfyUI
    cd ComfyUI
    ```
2.  **Place Models:** Put your SD checkpoints (ckpt/safetensors) in `models/checkpoints` and VAEs in `models/vae`.
3.  **GPU-Specific Setup:**
    *   **AMD (Linux):** Install ROCm and PyTorch.
        ```bash
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.4
        ```
    *   **Intel (Windows/Linux):** Install PyTorch with xpu support.
        ```bash
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
        ```
    *   **NVIDIA:** Install stable PyTorch.
        ```bash
        pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu129
        ```
4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
5.  **Run ComfyUI:**
    ```bash
    python main.py
    ```

### Support for Specific Hardware
*   **Apple Mac Silicon:** Install PyTorch nightly and follow the manual installation steps.
*   **DirectML (AMD Cards on Windows):** Install `torch-directml` and run with `--directml`. Not recommended for production.
*   **Ascend NPUs:** Follow the installation steps outlined in the documentation for torch-npu.
*   **Cambricon MLUs:** Follow the installation steps outlined in the documentation for torch-mlu.
*   **Iluvatar Corex:** Follow the installation steps outlined in the documentation for Iluvatar Extension for PyTorch.

## Running

```bash
python main.py
```

### Tips for AMD and ROCm:

*   For older AMD GPUs: `HSA_OVERRIDE_GFX_VERSION=10.3.0 python main.py`
*   For newer RDNA3 cards: `HSA_OVERRIDE_GFX_VERSION=11.0.0 python main.py`
*   Enable experimental memory efficient attention (on recent PyTorch, might not be default): `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 python main.py --use-pytorch-cross-attention`
*   Try `PYTORCH_TUNABLEOP_ENABLED=1` for potential speedups.

## Notes

*   Only executed parts of the graph that have outputs with correct inputs.
*   Only executed parts of the graph that have changed since last execution.
*   Dragging a generated PNG loads the full workflow.
*   Use parentheses for emphasis (e.g., (good code:1.2)).
*   Use curly braces for dynamic prompts (e.g., {wild|card|test}).

## How to Show High-Quality Previews?

*   Enable with `--preview-method auto`.
*   Install `taesd_decoder.pth` and place it in `models/vae_approx`.
*   Restart and run with `--preview-method taesd`.

## How to use TLS/SSL?

1.  Generate a self-signed certificate.
2.  Run `openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -sha256 -days 3650 -nodes -subj "/C=XX/ST=StateName/L=CityName/O=CompanyName/OU=CompanySectionName/CN=CommonNameOrHostname"`
3.  Use `--tls-keyfile key.pem --tls-certfile cert.pem` to enable.

## Support

*   [Discord](https://comfy.org/discord): #help or #feedback channels.
*   [Matrix space: #comfyui_space:matrix.org](https://app.element.io/#/room/%23comfyui_space%3Amatrix.org)
*   [https://www.comfy.org/](https://www.comfy.org/)

## Frontend Development

*   The frontend is now in a separate repository: [ComfyUI Frontend](https://github.com/Comfy-Org/ComfyUI_frontend).
*   For frontend issues, use the [ComfyUI Frontend repository](https://github.com/Comfy-Org/ComfyUI_frontend).
*   Use `--front-end-version` to specify frontend version. For example, `--front-end-version Comfy-Org/ComfyUI_frontend@latest` or `--front-end-version Comfy-Org/ComfyUI_frontend@1.2.2`.
*   Access the legacy frontend with `--front-end-version Comfy-Org/ComfyUI_legacy_frontend@latest`.

## QA

*   [Which GPU should I buy for ComfyUI?](https://github.com/comfyanonymous/ComfyUI/wiki/Which-GPU-should-I-buy-for-ComfyUI)