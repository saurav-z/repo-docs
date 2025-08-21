<div align="center">

# ComfyUI: Unleash the Power of Visual AI with Node-Based Workflows

**Create stunning images, videos, and more with ComfyUI, the most powerful and modular visual AI engine!**

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

</div>

ComfyUI is a powerful and flexible visual AI engine that allows you to build and execute complex Stable Diffusion pipelines using an intuitive node-based interface.  Available for Windows, Linux, and macOS. For the original repository, [see here](https://github.com/comfyanonymous/ComfyUI).

## Key Features

*   **Node-Based Workflow:** Design and experiment with complex Stable Diffusion pipelines without coding.
*   **Model Support:** Extensive support for a wide range of image, video, and audio models.
    *   **Image Models:** SD1.x, SD2.x, SDXL, SDXL Turbo, Stable Cascade, SD3 and SD3.5, Pixart Alpha and Sigma, AuraFlow, HunyuanDiT, Flux, Lumina Image 2.0, HiDream, Cosmos Predict2, Qwen Image.
    *   **Image Editing Models:** Omnigen 2, Flux Kontext, HiDream E1.1, Qwen Image Edit.
    *   **Video Models:** Stable Video Diffusion, Mochi, LTX-Video, Hunyuan Video, Nvidia Cosmos, Cosmos Predict2, Wan 2.1, Wan 2.2.
    *   **Audio Models:** Stable Audio, ACE Step.
    *   **3D Models:** Hunyuan3D 2.0
*   **Asynchronous Queue:** Efficiently manage and execute multiple generations.
*   **Optimized Performance:** Includes smart memory management and only re-executes changed parts of workflows.
*   **Hardware Compatibility:** Works with various GPUs (NVIDIA, AMD, Intel), and even without a GPU using `--cpu`.
*   **Model Loading:** Supports ckpt, safetensors, and other file types.
*   **Advanced Features:** Embeddings, LoRAs, Hypernetworks, ControlNet, Upscaling, Model Merging, LCM models and Loras, Area Composition, Inpainting and more.
*   **Workflow Handling:** Load and save workflows via PNG, WebP, FLAC, and JSON files.
*   **Extensible:**  Optional API nodes to use paid models from external providers.
*   **Offline Capable:** Fully functional offline, minimizing external dependencies.

## Getting Started

Choose your preferred installation method:

*   **Desktop Application:**  Easiest way to get started. Available on Windows & macOS. [Download](https://www.comfy.org/download)
*   **Windows Portable Package:** Get the latest commits, fully portable. Available on Windows.
*   **Manual Install:** Supports all operating systems and GPU types.

## Examples

Explore the possibilities with the extensive [example workflows](https://comfyanonymous.github.io/ComfyUI_examples/).

## Release Process

ComfyUI has a weekly release cycle with three interconnected repositories:

1.  **[ComfyUI Core](https://github.com/comfyanonymous/ComfyUI)**: Stable releases.
2.  **[ComfyUI Desktop](https://github.com/Comfy-Org/desktop)**: Builds desktop releases using the latest stable core.
3.  **[ComfyUI Frontend](https://github.com/Comfy-Org/ComfyUI_frontend)**: Hosts the latest frontend updates.

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

*Download, extract with [7-Zip](https://7-zip.org) and run*. Place Stable Diffusion checkpoints/models in: `ComfyUI\models\checkpoints`.

### [Direct link to download](https://github.com/comfyanonymous/ComfyUI/releases/latest/download/ComfyUI_windows_portable_nvidia.7z)

### [comfy-cli](https://docs.comfy.org/comfy-cli/getting-started)

```bash
pip install comfy-cli
comfy install
```

### Manual Install (Windows, Linux)

1.  **Clone the Repository:** `git clone [repo]`
2.  **Place Models:** Put SD checkpoints (ckpt/safetensors) in `models/checkpoints`, VAEs in `models/vae`.
3.  **Install Dependencies:** `pip install -r requirements.txt`

**Specific GPU Instructions:**

*   **AMD (Linux):** Install ROCm and PyTorch with: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.4`
*   **Intel (Windows/Linux):** Install PyTorch XPU or Intel Extension for PyTorch.
*   **NVIDIA:** Install PyTorch with: `pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu129`
*   **Apple Silicon:** Follow Apple Developer Guide instructions.
*   **DirectML (AMD on Windows):** `pip install torch-directml` (experimental).
*   **Ascend NPUs, Cambricon MLUs, Iluvatar Corex:** Follow the specific installation instructions provided.

## Running

```bash
python main.py
```

**GPU Troubleshooting:**

*   For AMD cards, try `HSA_OVERRIDE_GFX_VERSION=... python main.py`.
*   Enable experimental memory-efficient attention: `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 python main.py --use-pytorch-cross-attention`.

## Notes

*   ComfyUI executes only parts of the graph with outputs and correct inputs.
*   Only changed parts of the graph are re-executed.
*   Drag-and-drop generated images to load the full workflow.
*   Use parentheses `()` for emphasis in prompts.
*   Use curly braces `{}` for dynamic prompts.
*   Embeddings go in the `models/embeddings` directory.

## High-Quality Previews

Enable previews with `--preview-method auto`.  For higher-quality previews, download the TAESD decoders and launch with `--preview-method taesd`.

## TLS/SSL

Generate a certificate and key: `openssl req -x509 -newkey rsa:4096 ...`. Enable TLS/SSL with `--tls-keyfile key.pem --tls-certfile cert.pem`.

## Support & Community

*   [Discord](https://comfy.org/discord):  Get help in the #help or #feedback channels.
*   [Matrix Space](https://app.element.io/#/room/%23comfyui_space%3Amatrix.org)
*   [ComfyUI Website](https://www.comfy.org/)

## Frontend Development

The frontend is now in a separate repository: [ComfyUI Frontend](https://github.com/Comfy-Org/ComfyUI_frontend).

*   For frontend issues, use the ComfyUI Frontend repository.
*   Use `--front-end-version` in the command line to control the frontend version.  
    *   Example:  `--front-end-version Comfy-Org/ComfyUI_frontend@latest`

## QA

### Which GPU should I buy for this?

[See this page for some recommendations](https://github.com/comfyanonymous/ComfyUI/wiki/Which-GPU-should-I-buy-for-ComfyUI)