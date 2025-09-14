<div align="center">

# ComfyUI: Unleash Advanced AI Image Generation with a Visual Workflow

**Create stunning AI-generated images, videos, and more with ComfyUI, the powerful and modular visual AI engine.**

[![Website](https://img.shields.io/badge/ComfyOrg-4285F4?style=flat)](https://www.comfy.org/)
[![Discord](https://img.shields.io/badge/Discord-green?style=flat&logo=discord&logoColor=white&label=Discord&suffix=%20total)](https://www.comfy.org/discord)
[![Twitter](https://img.shields.io/twitter/follow/ComfyUI?style=flat)](https://x.com/ComfyUI)
[![Matrix](https://img.shields.io/badge/Matrix-000000?style=flat&logo=matrix&logoColor=white)](https://app.element.io/#/room/%23comfyui_space%3Amatrix.org)
<br>
[![GitHub Release](https://img.shields.io/github/v/release/comfyanonymous/ComfyUI?style=flat&sort=semver)](https://github.com/comfyanonymous/ComfyUI/releases)
[![Release Date](https://img.shields.io/github/release-date/comfyanonymous/ComfyUI?style=flat)](https://github.com/comfyanonymous/ComfyUI/releases)
[![Downloads](https://img.shields.io/github/downloads/comfyanonymous/ComfyUI/total?style=flat)](https://github.com/comfyanonymous/ComfyUI/releases)
[![Latest Downloads](https://img.shields.io/github/downloads/comfyanonymous/ComfyUI/latest/total?style=flat&label=downloads%40latest)](https://github.com/comfyanonymous/ComfyUI/releases)

![ComfyUI Screenshot](https://github.com/user-attachments/assets/7ccaf2c1-9b72-41ae-9a89-5688c94b7abe)

</div>

ComfyUI provides a node-based interface for creating and executing complex Stable Diffusion pipelines, empowering users with unparalleled control over their AI image generation workflows. **Check out the original repository for more details: [ComfyUI](https://github.com/comfyanonymous/ComfyUI)**

## Key Features

*   **Visual Workflow Interface:** Design and execute complex Stable Diffusion workflows using a drag-and-drop, node-based interface without any coding.
*   **Extensive Model Support:**
    *   Supports a wide range of image generation models, including:
        *   SD1.x, SD2.x, unCLIP
        *   SDXL, SDXL Turbo
        *   Stable Cascade, SD3 and SD3.5, Pixart Alpha and Sigma, AuraFlow, HunyuanDiT, Flux, Lumina Image 2.0, HiDream, Qwen Image
    *   Offers support for various image editing models: Omnigen 2, Flux Kontext, HiDream E1.1, Qwen Image Edit
    *   Includes comprehensive video model support: Stable Video Diffusion, Mochi, LTX-Video, Hunyuan Video, Wan 2.1, Wan 2.2
    *   Integrates audio models: Stable Audio, ACE Step
    *   Supports 3D models: Hunyuan3D 2.0
*   **Advanced Optimization:**
    *   Asynchronous queue system for efficient processing.
    *   Optimized execution, re-executing only changed parts of the workflow.
    *   Smart memory management for running large models on GPUs with limited VRAM.
    *   CPU fallback mode (`--cpu`) for users without GPUs.
*   **Model Compatibility:**
    *   Loads ckpt, safetensors, and other model file types.
    *   Supports embeddings/textual inversions, LoRAs (regular, locon, and loha), and hypernetworks.
*   **Workflow Management:**
    *   Loads full workflows from generated PNG, WebP, and FLAC files, preserving seeds.
    *   Saves and loads workflows as JSON files.
    *   Create complex workflows using nodes, such as for [Hires fix](https://comfyanonymous.github.io/ComfyUI_examples/2_pass_txt2img/) or advanced use cases.
*   **Powerful Capabilities:** Includes features such as:
    *   Area Composition
    *   Inpainting (with regular and inpainting models)
    *   ControlNet and T2I-Adapter
    *   Upscale Models (ESRGAN, SwinIR, etc.)
    *   GLIGEN
    *   Model Merging
    *   LCM models and Loras
    *   Latent previews with [TAESD](#how-to-show-high-quality-previews)
*   **Offline Functionality:** Operates fully offline.
*   **API Integration:** Optional API nodes for paid models via the [Comfy API](https://docs.comfy.org/tutorials/api-nodes/overview).
*   **Customization:** Uses a [Config file](extra_model_paths.yaml.example) to set the search paths for models.

## Getting Started

Choose your preferred installation method:

*   **Desktop Application:** ([Download](https://www.comfy.org/download)) Easiest way to get started, available on Windows & macOS.
*   **Windows Portable Package:** Get the latest commits, completely portable. Available on Windows.
*   **Manual Install:** Supports all operating systems and GPU types (NVIDIA, AMD, Intel, Apple Silicon, Ascend).

## Examples
Explore example workflows on the [examples page](https://comfyanonymous.github.io/ComfyUI_examples/) to see the possibilities.

## Release Process

ComfyUI follows a weekly release cycle.

1.  **ComfyUI Core**: Releases new stable versions
2.  **ComfyUI Desktop**: Builds using the latest stable core version
3.  **ComfyUI Frontend**: Weekly frontend updates are merged into the core repository

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

## Installing

### Windows Portable

A portable standalone build is available on the [releases page](https://github.com/comfyanonymous/ComfyUI/releases).

*   **[Direct Link to Download](https://github.com/comfyanonymous/ComfyUI/releases/latest/download/ComfyUI_windows_portable_nvidia.7z)**
*   Extract with [7-Zip](https://7-zip.org).
*   Place Stable Diffusion checkpoints/models in `ComfyUI\models\checkpoints`.
*   If extraction issues, right-click the file -> properties -> unblock

#### Share Models

Configure model search paths in `ComfyUI/extra_model_paths.yaml`. Rename and edit the file.

### [comfy-cli](https://docs.comfy.org/comfy-cli/getting-started)

```bash
pip install comfy-cli
comfy install
```

### Manual Install (Windows, Linux)

*   Python 3.13 is well supported. Consider 3.12 if facing custom node dependency issues.
*   Clone the repository.
*   Place SD checkpoints (ckpt/safetensors) in `models/checkpoints`.
*   Place VAEs in `models/vae`.

#### GPU-Specific Installation

Follow the instructions for your specific GPU:

*   **AMD GPUs (Linux):**
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.4
    ```
    or, for the nightly build:
    ```bash
    pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.4
    ```
*   **Intel GPUs (Windows and Linux):**
    *   Option 1: Install PyTorch xpu
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
    ```
    or, for the nightly build:
    ```bash
    pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/xpu
    ```
    *   Option 2: Leverage Intel Extension for PyTorch (IPEX). Visit [Installation](https://intel.github.io/intel-extension-for-pytorch/index.html#installation?platform=gpu)
*   **NVIDIA:**
    ```bash
    pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu129
    ```
    or, for the nightly build:
    ```bash
    pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu129
    ```
    *   **Troubleshooting:** If you get "Torch not compiled with CUDA enabled" error, uninstall and reinstall PyTorch.
*   **Apple Mac Silicon:**
    1.  Install PyTorch nightly (see [Accelerated PyTorch training on Mac](https://developer.apple.com/metal/pytorch/)).
    2.  Follow the [manual installation](#manual-install-windows-linux) instructions.
    3.  Install [dependencies](#dependencies).
    4.  Launch with `python main.py`.
*   **DirectML (AMD Cards on Windows):** (This is poorly supported.)
    ```bash
    pip install torch-directml
    python main.py --directml
    ```
*   **Ascend NPUs:** Follow the installation instructions on the [Ascend documentation](https://ascend.github.io/docs/sources/ascend/quick_install.html) and [torch-npu installation](https://ascend.github.io/docs/sources/pytorch/install.html#pytorch).
*   **Cambricon MLUs:** Follow the [Cambricon CNToolkit installation](https://www.cambricon.com/docs/sdk_1.15.0/cntoolkit_3.7.2/cntoolkit_install_3.7.2/index.html) and [PyTorch(torch_mlu) installation](https://www.cambricon.com/docs/sdk_1.15.0/cambricon_pytorch_1.17.0/user_guide_1.9/index.html).
*   **Iluvatar Corex:** Follow the [Iluvatar Corex Toolkit installation](https://support.iluvatar.com/#/DocumentCentre?id=1&nameCenter=2&productId=520117912052801536).

#### Dependencies

Install dependencies inside the ComfyUI folder:

```bash
pip install -r requirements.txt
```

## Running

```bash
python main.py
```

### GPU Troubleshooting

*   **AMD (Non-ROCm):**  Try:
    ```bash
    HSA_OVERRIDE_GFX_VERSION=10.3.0 python main.py  # for older RDNA2 or similar
    HSA_OVERRIDE_GFX_VERSION=11.0.0 python main.py  # for RDNA3
    ```
*   **AMD ROCm Tips:**
    ```bash
    TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 python main.py --use-pytorch-cross-attention
    PYTORCH_TUNABLEOP_ENABLED=1 python main.py
    ```

## Notes

*   Only graph sections with all correct inputs are executed.
*   Only changes between executions are processed.
*   Drag-and-drop generated PNGs to load the full workflow.
*   Use `()` for emphasis (e.g., `(good code:1.2)`).
*   Use `{}` for dynamic prompts (e.g., `{wild|card|test}`).
*   Use embeddings in text prompts like this: `embedding:embedding_filename.pt`.

## How to show high-quality previews?

Use `--preview-method auto` to enable previews.
*   Install [TAESD](https://github.com/madebyollin/taesd): Download `taesd_decoder.pth`, `taesdxl_decoder.pth`, `taesd3_decoder.pth` and `taef1_decoder.pth` from the repository and place them in `models/vae_approx`.
*   Restart ComfyUI and run with `--preview-method taesd`.

## How to use TLS/SSL?

*   Generate a self-signed certificate:
    ```bash
    openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -sha256 -days 3650 -nodes -subj "/C=XX/ST=StateName/L=CityName/O=CompanyName/OU=CompanySectionName/CN=CommonNameOrHostname"
    ```
*   Run with `--tls-keyfile key.pem --tls-certfile cert.pem` for HTTPS access.

## Support and Dev Channels

*   [Discord](https://comfy.org/discord) - #help and #feedback channels
*   [Matrix space: #comfyui_space:matrix.org](https://app.element.io/#/room/%23comfyui_space%3Amatrix.org)
*   [Comfy.org](https://www.comfy.org/)

## Frontend Development

The frontend is now in a separate repository: [ComfyUI Frontend](https://github.com/Comfy-Org/ComfyUI_frontend).

### Reporting Issues and Requesting Features

For frontend issues, use the [ComfyUI Frontend repository](https://github.com/Comfy-Org/ComfyUI_frontend).

### Using the Latest Frontend

1.  **Stable:** The default frontend updates fortnightly.
2.  **Latest Daily:** Run ComfyUI with `--front-end-version Comfy-Org/ComfyUI_frontend@latest`
3.  **Specific Version:** Replace `latest` with the version number (e.g., `--front-end-version Comfy-Org/ComfyUI_frontend@1.2.2`).

### Accessing the Legacy Frontend

Use `--front-end-version Comfy-Org/ComfyUI_legacy_frontend@latest`.

## QA

### Which GPU should I buy for this?

[See this page for some recommendations](https://github.com/comfyanonymous/ComfyUI/wiki/Which-GPU-should-I-buy-for-ComfyUI)