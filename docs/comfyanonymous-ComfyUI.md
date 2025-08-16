<div align="center">

# ComfyUI: Unleash the Power of Visual AI

**ComfyUI is a cutting-edge, node-based AI engine, empowering you to create stunning visuals through flexible and modular workflows.**  Discover unparalleled control and customization for your AI image generation, editing, and more!

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

![ComfyUI Screenshot](https://github.com/user-attachments/assets/7ccaf2c1-9b72-41ae-9a89-5688c94b7abe)
</div>

**Get started with ComfyUI by visiting the [original repository](https://github.com/comfyanonymous/ComfyUI)!**

## Key Features

*   **Node-Based Workflow:** Design and execute complex Stable Diffusion pipelines visually with a graph/nodes/flowchart interface, eliminating the need for code.
*   **Extensive Model Support:**  Compatible with a wide range of image, video, and audio models, including:
    *   SD1.x, SD2.x (and unCLIP)
    *   SDXL, SDXL Turbo
    *   Stable Cascade
    *   SD3 and SD3.5
    *   Pixart Alpha and Sigma
    *   AuraFlow, HunyuanDiT, Flux, Lumina Image 2.0, HiDream, Cosmos Predict2, Qwen Image.
    *   Image Editing models: Omnigen 2, Flux Kontext, HiDream E1.1
    *   Video models: Stable Video Diffusion, Mochi, LTX-Video, Hunyuan Video, Nvidia Cosmos and Cosmos Predict2, Wan 2.1, Wan 2.2
    *   Audio models: Stable Audio, ACE Step
    *   3D Models: Hunyuan3D 2.0
*   **Asynchronous Queue & Optimizations:** Benefit from an asynchronous queue system and optimized execution that only re-processes changed parts of your workflow.
*   **Smart Memory Management:** Run large models on GPUs with as little as 1GB of VRAM with smart offloading.
*   **Broad Hardware Compatibility:**  Works with and without a GPU (using `--cpu`), supporting NVIDIA, AMD, Intel, Apple Silicon, and Ascend hardware.
*   **Checkpoint and File Support:** Load various file types, including ckpt, safetensors, and more, for models, VAEs, and CLIP models.
*   **Advanced Features:**  Explore functionalities like embeddings/textual inversion, LoRAs, Hypernetworks, workflow loading/saving, and node-based complex workflow creation.
*   **ControlNet & Upscaling:** Integrates ControlNet, T2I-Adapter, and supports a wide range of upscale models (ESRGAN, SwinIR, etc.).
*   **Preview Capabilities:**  Utilize latent previews with TAESD for high-quality results.
*   **Offline Functionality:**  Operates fully offline, with the option to use paid models through the Comfy API.
*   **Configurable:**  Customize model paths with the `extra_model_paths.yaml.example` configuration file.
*   **Comprehensive Examples:** Explore diverse workflows at the [Examples page](https://comfyanonymous.github.io/ComfyUI_examples/).

## Installation

Choose the installation method that best suits your needs:

*   **Desktop Application:**  The easiest way to get started, available for Windows and macOS. ([Download](https://www.comfy.org/download))
*   **Windows Portable Package:** Get the latest commits with a fully portable package. ([Releases](https://github.com/comfyanonymous/ComfyUI/releases))
*   **comfy-cli:** Install and start ComfyUI using `comfy-cli`:
    ```bash
    pip install comfy-cli
    comfy install
    ```
*   **Manual Install (Windows, Linux, macOS):**  Supports all operating systems and GPU types, requiring you to manually configure dependencies.

    *   **Prerequisites:** Python 3.12 (recommended) or 3.13, Git.
    *   **Steps:**
        1.  Clone the repository: `git clone https://github.com/comfyanonymous/ComfyUI`
        2.  Place your SD checkpoints (e.g., ckpt/safetensors) in `models/checkpoints`.
        3.  Place your VAEs in `models/vae`.
        4.  **GPU-Specific Instructions:**
            *   **AMD (Linux):**  Install ROCm and PyTorch.
                ```bash
                pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.4
                ```
                or
                ```bash
                pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.4
                ```
            *   **Intel (Windows and Linux):**
                *   (Option 1) Install PyTorch xpu
                ```bash
                pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
                ```
                or
                ```bash
                pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/xpu
                ```
                *   (Option 2) Intel Extension for PyTorch (IPEX) [Installation](https://intel.github.io/intel-extension-for-pytorch/index.html#installation?platform=gpu)
            *   **NVIDIA:**
                ```bash
                pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu129
                ```
                or
                ```bash
                pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu129
                ```
            *   **Apple Mac silicon:** Follow the [Accelerated PyTorch training on Mac](https://developer.apple.com/metal/pytorch/) guide to install PyTorch nightly.
            *   **Ascend NPUs:** [Installation](https://ascend.github.io/docs/sources/pytorch/install.html#pytorch)
            *   **Cambricon MLUs:** [Installation](https://www.cambricon.com/docs/sdk_1.15.0/cntoolkit_3.7.2/cntoolkit_install_3.7.2/index.html)
            *   **Iluvatar Corex:** [Installation](https://support.iluvatar.com/#/DocumentCentre?id=1&nameCenter=2&productId=520117912052801536)
        5.  Install dependencies: `pip install -r requirements.txt`
        6.  Run ComfyUI: `python main.py`

## Running

Execute ComfyUI using the command:

```bash
python main.py
```

*   **AMD ROCm Tips:**  For improved performance with some AMD GPUs, experiment with the following:

    ```bash
    HSA_OVERRIDE_GFX_VERSION=10.3.0 python main.py  (for some RDNA2 or older)
    HSA_OVERRIDE_GFX_VERSION=11.0.0 python main.py  (for some RDNA3)
    TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 python main.py --use-pytorch-cross-attention
    PYTORCH_TUNABLEOP_ENABLED=1 python main.py
    ```

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

## Notes

*   Only connected graph parts with all input requirements met are executed.
*   Only changed parts of a graph are re-executed on subsequent runs.
*   Dragging and dropping a generated PNG or loading a workflow loads the complete workflow with seeds.
*   Use `()` to adjust emphasis: `(good code:1.2)` or `(bad code:0.8)`.
*   Use `{day|night}` for wildcard prompts.
*   Use `// comment` or `/* comment */` for dynamic prompts comments
*   Use textual inversion concepts/embeddings in a text prompt in the models/embeddings directory like this (you can omit the .pt extension): `embedding:embedding_filename.pt`

## High-Quality Previews

*   Enable preview previews with `--preview-method auto`.
*   For higher quality previews, download `taesd_decoder.pth, taesdxl_decoder.pth, taesd3_decoder.pth and taef1_decoder.pth` from [TAESD](https://github.com/madebyollin/taesd/) and place them in the `models/vae_approx` folder.
*   Restart ComfyUI and run it with `--preview-method taesd`.

## TLS/SSL

Generate a self-signed certificate with:
```bash
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -sha256 -days 3650 -nodes -subj "/C=XX/ST=StateName/L=CityName/O=CompanyName/OU=CompanySectionName/CN=CommonNameOrHostname"
```
Enable TLS/SSL with: `--tls-keyfile key.pem --tls-certfile cert.pem`

## Support and Development

*   **Discord:**  Join the #help or #feedback channels on [Discord](https://comfy.org/discord).
*   **Matrix:**  Connect with the community on [Matrix](https://app.element.io/#/room/%23comfyui_space%3Amatrix.org).
*   **Website:**  Visit [https://www.comfy.org/](https://www.comfy.org/) for additional resources.

## Frontend Development

ComfyUI now has a separate frontend repository for frontend development: [ComfyUI Frontend](https://github.com/Comfy-Org/ComfyUI_frontend).

*   Use `--front-end-version Comfy-Org/ComfyUI_frontend@latest` for the latest daily release.
*   Use `--front-end-version Comfy-Org/ComfyUI_frontend@<version>` for a specific version.
*   Use `--front-end-version Comfy-Org/ComfyUI_legacy_frontend@latest` for the legacy frontend.

## QA

*   **GPU Recommendations:**  Consult the [GPU Recommendations](https://github.com/comfyanonymous/ComfyUI/wiki/Which-GPU-should-I-buy-for-ComfyUI) page.