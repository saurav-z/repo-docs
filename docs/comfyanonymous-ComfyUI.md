<div align="center">

# ComfyUI: Unleash Your Creativity with Visual AI (Stable Diffusion)

**Create stunning AI-generated images and videos with ComfyUI, the most powerful and modular visual AI engine built for Stable Diffusion and beyond.**

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
<!-- Workaround to display total user from https://github.com/badges/shields/issues/4500#issuecomment-2060079995 -->
[discord-shield]: https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fdiscord.com%2Fapi%2Finvites%2Fcomfyorg%3Fwith_counts%3Dtrue&query=%24.approximate_member_count&logo=discord&logoColor=white&label=Discord&color=green&suffix=%20total
[discord-url]: https://www.comfy.org/discord
[twitter-shield]: https://img.shields.io/twitter/follow/ComfyUI
[twitter-url]: https://x.com/ComfyUI

[github-release-shield]: https://img.shields.io/github/v/release/comfyanonymous/ComfyUI?style=flat&sort=semver
[github-release-link]: https://github.com/comfyanonymous/ComfyUI/releases
[github-release-date-shield]: https://img.shields.io/github/release-date/comfyanonymous/ComfyUI?style=flat
[github-downloads-shield]: https://img.shields/github/downloads/comfyanonymous/ComfyUI/total?style=flat
[github-downloads-latest-shield]: https://img.shields.io/github/downloads/comfyanonymous/ComfyUI/latest/total?style=flat&label=downloads%40latest
[github-downloads-link]: https://github.com/comfyanonymous/ComfyUI/releases

![ComfyUI Screenshot](https://github.com/user-attachments/assets/7ccaf2c1-9b72-41ae-9a89-5688c94b7abe)
</div>

ComfyUI offers a node-based interface, enabling users to create complex Stable Diffusion workflows without coding.  Available on Windows, Linux, and macOS.  **[Get started with ComfyUI now](https://github.com/comfyanonymous/ComfyUI) and experience the future of AI image generation!**

## Key Features

*   **Node-Based Workflow:** Design and execute intricate Stable Diffusion pipelines using a visual, graph-based interface.
*   **Extensive Model Support:**  Compatible with a wide array of image, video, and audio models, including:
    *   **Image Models:** SD1.x, SD2.x, SDXL, SDXL Turbo, Stable Cascade, SD3/3.5, Pixart, AuraFlow, HunyuanDiT, Flux, Lumina 2.0, HiDream, and Qwen Image.
    *   **Image Editing Models:** Omnigen 2, Flux Kontext, HiDream E1.1, and Qwen Image Edit.
    *   **Video Models:** Stable Video Diffusion, Mochi, LTX-Video, Hunyuan Video, Wan 2.1 & 2.2.
    *   **Audio Models:** Stable Audio and ACE Step.
    *   **3D Models:** Hunyuan3D 2.0
*   **Flexible Execution:**  Utilizes an asynchronous queue system and optimizes execution by re-running only changed parts of the workflow.
*   **Smart Memory Management:** Automatically runs large models on GPUs with as little as 1GB of VRAM through smart offloading.
*   **Cross-Platform Compatibility:**  Works on Windows, Linux, and macOS, and can even run on the CPU with the `--cpu` flag.
*   **Model Loading:** Supports ckpt, safetensors, and various other file types.
*   **Advanced Features:** Includes support for embeddings, LoRAs, Hypernetworks, workflow loading/saving, nodes for Hires fix, inpainting, ControlNet, and more.
*   **Optional API Integration:** Connect with paid models from external providers via the Comfy API.
*   **Offline Functionality:** Core functionality works completely offline.
*   **High-Quality Previews:** Optional integration with TAESD for enhanced preview quality.

## Getting Started

ComfyUI offers several installation methods:

*   **Desktop Application:**  The easiest way to get started, available for Windows & macOS.  [Download](https://www.comfy.org/download)
*   **Windows Portable Package:**  Get the latest commits with a completely portable solution for Windows. [Learn More](#installing)
*   **Manual Install:**  Supports all operating systems and GPU types (NVIDIA, AMD, Intel, Apple Silicon, Ascend). [Learn More](#manual-install-windows-linux)

## [Examples](https://comfyanonymous.github.io/ComfyUI_examples/)

Explore the possibilities with a collection of example workflows.

## Release Process

ComfyUI follows a weekly release cycle. Learn more about the release cycle, and the three interconnected repositories that comprise the project.

1.  **ComfyUI Core:**  [https://github.com/comfyanonymous/ComfyUI](https://github.com/comfyanonymous/ComfyUI)
2.  **ComfyUI Desktop:**  [https://github.com/Comfy-Org/desktop](https://github.com/Comfy-Org/desktop)
3.  **ComfyUI Frontend:** [https://github.com/Comfy-Org/ComfyUI_frontend](https://github.com/Comfy-Org/ComfyUI_frontend)

## Useful Shortcuts

| Keybind                       | Explanation                                                                                                    |
| ----------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `Ctrl` + `Enter`              | Queue up current graph for generation                                                                          |
| `Ctrl` + `Shift` + `Enter`      | Queue up current graph as first for generation                                                                 |
| `Ctrl` + `Alt` + `Enter`        | Cancel current generation                                                                                      |
| `Ctrl` + `Z`/`Ctrl` + `Y`       | Undo/Redo                                                                                                      |
| `Ctrl` + `S`                  | Save workflow                                                                                                  |
| `Ctrl` + `O`                  | Load workflow                                                                                                  |
| `Ctrl` + `A`                  | Select all nodes                                                                                               |
| `Alt `+ `C`                   | Collapse/uncollapse selected nodes                                                                             |
| `Ctrl` + `M`                  | Mute/unmute selected nodes                                                                                     |
| `Ctrl` + `B`                   | Bypass selected nodes (acts like the node was removed from the graph and the wires reconnected through)        |
| `Delete`/`Backspace`           | Delete selected nodes                                                                                          |
| `Ctrl` + `Backspace`           | Delete the current graph                                                                                       |
| `Space`                       | Move the canvas around when held and moving the cursor                                                         |
| `Ctrl`/`Shift` + `Click`       | Add clicked node to selection                                                                                  |
| `Ctrl` + `C`/`Ctrl` + `V`      | Copy and paste selected nodes (without maintaining connections to outputs of unselected nodes)                 |
| `Ctrl` + `C`/`Ctrl` + `Shift` + `V` | Copy and paste selected nodes (maintaining connections from outputs of unselected nodes to inputs of pasted nodes) |
| `Shift` + `Drag`               | Move multiple selected nodes at the same time                                                                  |
| `Ctrl` + `D`                   | Load default graph                                                                                             |
| `Alt` + `+`                  | Canvas Zoom in                                                                                                 |
| `Alt` + `-`                  | Canvas Zoom out                                                                                                |
| `Ctrl` + `Shift` + LMB + Vertical drag | Canvas Zoom in/out                                                                                      |
| `P`                           | Pin/Unpin selected nodes                                                                                       |
| `Ctrl` + `G`                   | Group selected nodes                                                                                           |
| `Q`                          | Toggle visibility of the queue                                                                                   |
| `H`                           | Toggle visibility of history                                                                                     |
| `R`                           | Refresh graph                                                                                                  |
| `F`                           | Show/Hide menu                                                                                                  |
| `.`                           | Fit view to selection (Whole graph when nothing is selected)                                                    |
| Double-Click LMB             | Open node quick search palette                                                                                 |
| `Shift` + Drag               | Move multiple wires at once                                                                                      |
| `Ctrl` + `Alt` + LMB           | Disconnect all wires from clicked slot                                                                         |

## Installing

### Windows Portable

*   Download, extract and run from the [releases page](https://github.com/comfyanonymous/ComfyUI/releases) on Windows.
*   Place your Stable Diffusion checkpoints (the huge ckpt/safetensors files) in: `ComfyUI\models\checkpoints`
*   If you have trouble extracting it, right click the file -> properties -> unblock
*   See the [Config file](extra_model_paths.yaml.example) to set the search paths for models.  Rename this file to `extra_model_paths.yaml` and edit it with your favorite text editor.

### \[comfy-cli](https://docs.comfy.org/comfy-cli/getting-started)

Install and start ComfyUI using comfy-cli:

```bash
pip install comfy-cli
comfy install
```

### Manual Install (Windows, Linux)

Python 3.13 is very well supported. If you have trouble with some custom node dependencies you can try 3.12

1.  Git clone this repository.
2.  Place your SD checkpoints in `models/checkpoints`
3.  Place your VAE in `models/vae`
4.  Install dependencies (see platform-specific instructions below)

### AMD GPUs (Linux only)

Install rocm and pytorch:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.4
```
OR:

```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.4
```

### Intel GPUs (Windows and Linux)

(Option 1) Intel Arc GPU users can install native PyTorch with torch.xpu support using pip. More information can be found [here](https://pytorch.org/docs/main/notes/get_start_xpu.html)

1.  To install PyTorch xpu, use the following command:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
```
OR:

```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/xpu
```

(Option 2) Alternatively, Intel GPUs supported by Intel Extension for PyTorch (IPEX) can leverage IPEX for improved performance.

1.  visit [Installation](https://intel.github.io/intel-extension-for-pytorch/index.html#installation?platform=gpu) for more information.

### NVIDIA

```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu129
```

OR:

```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu129
```

#### Troubleshooting

If you get the "Torch not compiled with CUDA enabled" error, uninstall and reinstall PyTorch:

```bash
pip uninstall torch
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu129
```

### Others:

*   **Apple Mac silicon:**  Follow these instructions for [Accelerated PyTorch training on Mac](https://developer.apple.com/metal/pytorch/) and then the [ComfyUI manual installation](#manual-install-windows-linux) instructions.
*   **DirectML (AMD Cards on Windows):** Not recommended.
*   **Ascend NPUs:** Follow the instructions on the [installation](https://ascend.github.io/docs/sources/ascend/quick_install.html) page.
*   **Cambricon MLUs:** Follow the instructions on the [Installation](https://www.cambricon.com/docs/sdk_1.15.0/cntoolkit_3.7.2/cntoolkit_install_3.7.2/index.html) and [Installation](https://www.cambricon.com/docs/sdk_1.15.0/cambricon_pytorch_1.17.0/user_guide_1.9/index.html) pages.
*   **Iluvatar Corex:** Follow the instructions on the [Installation](https://support.iluvatar.com/#/DocumentCentre?id=1&nameCenter=2&productId=520117912052801536) page.

### Dependencies

Install dependencies from the command line:

```bash
pip install -r requirements.txt
```

### Running

Run ComfyUI:

```bash
python main.py
```

### AMD ROCm Tips

Enable experimental memory efficient attention:

```bash
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 python main.py --use-pytorch-cross-attention
```

## Notes

*   Only parts of the graph with all inputs will be executed.
*   Only parts of the graph that change between executions will be re-executed.
*   Drag a generated PNG onto the webpage to load the full workflow.
*   Use parentheses `()` for emphasis (e.g., `(good code:1.2)`), and escape with `\\(`, `\\)`.
*   Use curly braces `{}` for wildcard prompts (e.g., `{wild|card|test}`), and escape with `\\{`, `\\}`.
*   Use textual inversion concepts in `models/embeddings` directory.

## How to show high-quality previews?

Use ```--preview-method auto``` to enable previews.  Download [TAESD](https://github.com/madebyollin/taesd/) decoders to the `models/vae_approx` folder and start with `--preview-method taesd` for higher quality.

## How to use TLS/SSL?

Generate a self-signed certificate and key (not for production use):

```bash
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -sha256 -days 3650 -nodes -subj "/C=XX/ST=StateName/L=CityName/O=CompanyName/OU=CompanySectionName/CN=CommonNameOrHostname"
```

Enable TLS/SSL:

```bash
--tls-keyfile key.pem --tls-certfile cert.pem
```

## Support and Dev Channels

*   [Discord](https://comfy.org/discord)
*   [Matrix space: #comfyui_space:matrix.org](https://app.element.io/#/room/%23comfyui_space%3Amatrix.org)
*   [https://www.comfy.org/](https://www.comfy.org/)

## Frontend Development

The frontend is now hosted in a separate repository: [ComfyUI Frontend](https://github.com/Comfy-Org/ComfyUI_frontend).

*   **Reporting Issues and Requesting Features:** Use the [ComfyUI Frontend repository](https://github.com/Comfy-Org/ComfyUI_frontend) for frontend-related issues.
*   **Using the Latest Frontend:**
    *   For the latest daily release: `--front-end-version Comfy-Org/ComfyUI_frontend@latest`
    *   For a specific version: `--front-end-version Comfy-Org/ComfyUI_frontend@1.2.2`
*   **Accessing the Legacy Frontend:** `--front-end-version Comfy-Org/ComfyUI_legacy_frontend@latest`

## QA

*   [Which GPU should I buy for this?](https://github.com/comfyanonymous/ComfyUI/wiki/Which-GPU-should-I-buy-for-ComfyUI)