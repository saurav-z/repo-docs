<div align="center">
  <img src="https://github.com/user-attachments/assets/7ccaf2c1-9b72-41ae-9a89-5688c94b7abe" alt="ComfyUI Screenshot" width="70%">
</div>

# ComfyUI: Unleash Your AI Creativity with a Powerful Visual Workflow Engine

**ComfyUI is a cutting-edge visual AI engine that empowers you to build and execute advanced stable diffusion pipelines through an intuitive node-based interface.** [Get Started with ComfyUI](https://github.com/comfyanonymous/ComfyUI)

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
[github-downloads-shield]: https://img.shields.io/github/downloads/comfyanonymous/ComfyUI/total?style=flat
[github-downloads-latest-shield]: https://img.shields.io/github/downloads/comfyanonymous/ComfyUI/latest/total?style=flat&label=downloads%40latest
[github-downloads-link]: https://github.com/comfyanonymous/ComfyUI/releases

## Key Features

*   **Visual Workflow Interface:** Design and execute complex Stable Diffusion workflows effortlessly using a graph/node-based interface.
*   **Extensive Model Support:** Compatible with a wide range of image, video, and audio models, including SD1.x, SD2.x, SDXL, Stable Cascade, SD3, and many more. Explore the [Examples](https://comfyanonymous.github.io/ComfyUI_examples/)
*   **Image Editing Models:** Utilize powerful image editing models such as Omnigen 2, Flux Kontext, and Qwen Image Edit.
*   **Video & Audio Generation:** Generate stunning videos with Stable Video Diffusion, Mochi, and more, as well as audio with Stable Audio and ACE Step.
*   **3D Model Integration:** Support for 3D models like Hunyuan3D 2.0.
*   **Asynchronous Queue & Optimization:** Benefit from an asynchronous queue system and optimizations that only re-execute changed parts of your workflow.
*   **Smart Memory Management:** Run large models even on GPUs with limited VRAM through smart offloading.
*   **CPU Support:** Works even without a GPU using the `--cpu` flag.
*   **Checkpoint & LoRA Support:** Load ckpt, safetensors, embeddings, LoRAs, and Hypernetworks.
*   **Workflow Management:** Load and save workflows as JSON files and from PNG, WebP, and FLAC files.
*   **Advanced Features:** Includes area composition, inpainting, ControlNet, T2I-Adapter, and upscaling models.
*   **Offline Functionality:** Operates fully offline, ensuring privacy and control.
*   **Modular Design:** Easily extend functionality with custom nodes and extensions.
*   **API Nodes:** Utilize optional API nodes to integrate paid models from external providers.
*   **Flexible Configuration:** Configure search paths for models using the `extra_model_paths.yaml` file.

## Getting Started

Choose your preferred installation method:

*   **[Desktop Application](https://www.comfy.org/download):** The easiest way to get started, available for Windows & macOS.
*   **[Windows Portable Package](#installing):** Get the latest commits, completely portable, for Windows.
*   **[Manual Install](#manual-install-windows-linux):** Supports all operating systems and GPU types (NVIDIA, AMD, Intel, Apple Silicon, Ascend).
*   **[comfy-cli](https://docs.comfy.org/comfy-cli/getting-started)**: Use this simple utility to install ComfyUI.

## Shortcuts

*   `Ctrl + Enter`: Queue graph for generation
*   `Ctrl + Shift + Enter`: Queue graph as first for generation
*   `Ctrl + Alt + Enter`: Cancel generation
*   `Ctrl + Z`/`Ctrl + Y`: Undo/Redo
*   `Ctrl + S`: Save workflow
*   `Ctrl + O`: Load workflow
*   `Ctrl + A`: Select all nodes
*   `Alt + C`: Collapse/uncollapse selected nodes
*   `Ctrl + M`: Mute/unmute selected nodes
*   `Ctrl + B`: Bypass selected nodes
*   `Delete`/`Backspace`: Delete selected nodes
*   `Ctrl + Backspace`: Delete the current graph
*   `Space`: Move canvas
*   `Ctrl`/`Shift` + `Click`: Add to selection
*   `Ctrl + C`/`Ctrl + V`: Copy/paste nodes (without connections)
*   `Ctrl + C`/`Ctrl + Shift + V`: Copy/paste nodes (with connections)
*   `Shift + Drag`: Move multiple selected nodes
*   `Ctrl + D`: Load default graph
*   `Alt + +`: Canvas zoom in
*   `Alt + -`: Canvas zoom out
*   `Ctrl + Shift + LMB + Vertical drag`: Canvas Zoom in/out
*   `P`: Pin/Unpin selected nodes
*   `Ctrl + G`: Group selected nodes
*   `Q`: Toggle queue visibility
*   `H`: Toggle history visibility
*   `R`: Refresh graph
*   `F`: Show/Hide menu
*   `.`: Fit view to selection
*   Double-Click LMB: Open node quick search palette
*   `Shift + Drag`: Move multiple wires at once
*   `Ctrl + Alt + LMB`: Disconnect wires from slot
*   `Ctrl` can also be replaced with `Cmd` on macOS.

## Installing

### Windows Portable

*   Download, extract with [7-Zip](https://7-zip.org), and run.
*   Place your Stable Diffusion checkpoints/models in: `ComfyUI\models\checkpoints`
*   Rename the `extra_model_paths.yaml.example` file to `extra_model_paths.yaml` to specify model search paths.

### [comfy-cli](https://docs.comfy.org/comfy-cli/getting-started)

```bash
pip install comfy-cli
comfy install
```

### Manual Install (Windows, Linux)

*   Clone the repository.
*   Place checkpoints/models in `models/checkpoints` and VAEs in `models/vae`.
*   Install dependencies: `pip install -r requirements.txt`

#### AMD GPUs (Linux only)

Install pytorch for AMD:

```pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.4```

or for the nightly version:

```pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.4```

#### Intel GPUs (Windows and Linux)

(Option 1) Install Pytorch xpu:
```pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu```

or for the nightly version:
```pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/xpu```

(Option 2) Use Intel Extension for PyTorch (IPEX) for improved performance. [Installation](https://intel.github.io/intel-extension-for-pytorch/index.html#installation?platform=gpu)

#### NVIDIA

Install pytorch for NVIDIA:

```pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu129```

or for the nightly version:

```pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu129```

*   **Troubleshooting:** If you encounter "Torch not compiled with CUDA enabled," reinstall with the above command.

#### Others

*   **Apple Mac silicon:** Install pytorch nightly and follow the manual installation instructions.
*   **DirectML (AMD Cards on Windows):**  `pip install torch-directml` then `python main.py --directml` (Not recommended).
*   **Ascend NPUs:** Follow the installation instructions for your platform.
*   **Cambricon MLUs:** Follow the installation instructions for your platform.
*   **Iluvatar Corex:** Follow the installation instructions for your platform.

## Running

```bash
python main.py
```

*   **AMD Cards (Troubleshooting):** `HSA_OVERRIDE_GFX_VERSION=10.3.0 python main.py` (for older RDNA2) or `HSA_OVERRIDE_GFX_VERSION=11.0.0 python main.py` (for RDNA3).
*   **AMD ROCm Tips:**  Experiment with `--use-pytorch-cross-attention` and `PYTORCH_TUNABLEOP_ENABLED=1`.

## Notes

*   Only executed parts of the graph that have all correct inputs and those that change from execution to execution.
*   Drag-and-drop PNGs to load workflows with seeds.
*   Use `()` for emphasis and `{}` for dynamic prompts.
*   Place textual inversion concepts/embeddings in the `models/embeddings` directory.

## High-Quality Previews

Enable high-quality previews: `--preview-method auto`.  For TAESD previews, download `taesd_decoder.pth`, `taesdxl_decoder.pth`, `taesd3_decoder.pth`, and `taef1_decoder.pth` from the [TAESD](https://github.com/madebyollin/taesd/)  repo, put them in `models/vae_approx`, and launch with `--preview-method taesd`.

## TLS/SSL

Use `--tls-keyfile key.pem --tls-certfile cert.pem` to enable HTTPS.
Generate a self-signed certificate: `openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -sha256 -days 3650 -nodes -subj "/C=XX/ST=StateName/L=CityName/O=CompanyName/OU=CompanySectionName/CN=CommonNameOrHostname"`

## Support

*   [Discord](https://comfy.org/discord): #help or #feedback channels.
*   [Matrix](https://app.element.io/#/room/%23comfyui_space%3Amatrix.org)

## Frontend Development

The frontend is now in a separate repository: [ComfyUI Frontend](https://github.com/Comfy-Org/ComfyUI_frontend).

*   For the latest daily release: `--front-end-version Comfy-Org/ComfyUI_frontend@latest`
*   For a specific version: `--front-end-version Comfy-Org/ComfyUI_frontend@1.2.2`
*   Legacy frontend: `--front-end-version Comfy-Org/ComfyUI_legacy_frontend@latest`

## QA

*   [Which GPU should I buy for this?](https://github.com/comfyanonymous/ComfyUI/wiki/Which-GPU-should-I-buy-for-ComfyUI)

**[Visit the ComfyUI GitHub Repository](https://github.com/comfyanonymous/ComfyUI) for more information and to get started today!**