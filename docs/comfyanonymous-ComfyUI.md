<div align="center">

# ComfyUI: Unleash Your AI Creativity with a Powerful Visual Workflow Engine

**ComfyUI is the most powerful and modular visual AI engine and application, empowering you to create intricate and innovative image and video workflows.**

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

## Key Features:

*   **Node-Based Workflow:** Design complex Stable Diffusion pipelines using a visual, node-based interface, eliminating the need for extensive coding.
*   **Broad Model Support:**  Works with a wide range of image, video, and audio models including:
    *   **Image Models:** SD1.x, SD2.x, SDXL, SDXL Turbo, Stable Cascade, SD3/3.5, Pixart Alpha/Sigma, AuraFlow, HunyuanDiT, Flux, Lumina Image 2.0, HiDream, and Qwen Image.
    *   **Image Editing Models:** Omnigen 2, Flux Kontext, HiDream E1.1, Qwen Image Edit.
    *   **Video Models:** Stable Video Diffusion, Mochi, LTX-Video, Hunyuan Video, Wan 2.1/2.2.
    *   **Audio Models:** Stable Audio, ACE Step.
    *   **3D Models:** Hunyuan3D 2.0.
*   **Asynchronous Processing:** Utilizes an asynchronous queue system for efficient workflow execution.
*   **Optimized Performance:** Includes optimizations such as only re-executing changed parts of workflows and smart memory management, allowing it to run on GPUs with as little as 1GB VRAM.
*   **Broad Hardware Support:** Works even without a GPU using the `--cpu` flag (though slower).
*   **Checkpoint and File Support:** Loads ckpt, safetensors, and other model formats, including VAEs and CLIP models.
*   **Workflow Flexibility:** Supports embeddings/textual inversions, LoRAs, hypernetworks, and loading full workflows from PNG, WebP, and FLAC files.
*   **Extensive Workflow Examples:** Explore pre-built workflows for complex tasks like Hires fix, inpainting, ControlNet, and more. Find them at the [Examples Page](https://comfyanonymous.github.io/ComfyUI_examples/).
*   **Integration with External APIs:** Optional API nodes to utilize paid models through the online [Comfy API](https://docs.comfy.org/tutorials/api-nodes/overview).

## Getting Started

Choose your preferred installation method:

*   **Desktop Application:** The easiest way to get started, available for Windows and macOS.  [Download](https://www.comfy.org/download)
*   **Windows Portable Package:**  Get the latest commits in a completely portable package (Windows only).
*   **Manual Install:** Supports all operating systems and GPU types (NVIDIA, AMD, Intel, Apple Silicon, Ascend).  See the installation instructions below.
*   **comfy-cli:** Installation and start using comfy-cli:
    ```bash
    pip install comfy-cli
    comfy install
    ```

Explore pre-built workflows to get you started on the [Examples Page](https://comfyanonymous.github.io/ComfyUI_examples/).

### Manual Install (Windows, Linux)

#### Prerequisites:
*   Python 3.12/3.13
*   Git

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/comfyanonymous/ComfyUI.git
    cd ComfyUI
    ```

2.  **Model Placement:**
    *   Place your Stable Diffusion checkpoints (ckpt/safetensors) in the `models/checkpoints` directory.
    *   Place your VAEs in the `models/vae` directory.

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **GPU-Specific Installation (if applicable):**
    *   **AMD GPUs (Linux Only):**
        ```bash
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.4
        # or, for nightly builds:
        pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.4
        ```
    *   **Intel GPUs (Windows/Linux):**
        *   **Option 1 (Intel Arc):**
            ```bash
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
            # or, for nightly builds:
            pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/xpu
            ```
        *   **Option 2 (IPEX):** Follow the instructions [here](https://intel.github.io/intel-extension-for-pytorch/index.html#installation?platform=gpu).
    *   **NVIDIA GPUs:**
        ```bash
        pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu129
        # or, for nightly builds:
        pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu129
        ```
    *   **DirectML (AMD on Windows):** (Not Recommended)
        ```bash
        pip install torch-directml
        ```
        Run with: `python main.py --directml`

    *   **Ascend NPUs:** Follow the instructions [here](https://ascend.github.io/docs/sources/pytorch/install.html#pytorch).
    *   **Cambricon MLUs:** Follow the instructions [here](https://www.cambricon.com/docs/sdk_1.15.0/cambricon_pytorch_1.17.0/user_guide_1.9/index.html).
    *   **Iluvatar Corex:** Follow the instructions [here](https://support.iluvatar.com/#/DocumentCentre?id=1&nameCenter=2&productId=520117912052801536).
    *   **Apple Silicon:** Follow the instructions [here](https://developer.apple.com/metal/pytorch/), then the standard manual install instructions.

5.  **Run ComfyUI:**
    ```bash
    python main.py
    ```
    *   For AMD cards, try:
        *   `HSA_OVERRIDE_GFX_VERSION=10.3.0 python main.py` (for 6700, 6600, etc.)
        *   `HSA_OVERRIDE_GFX_VERSION=11.0.0 python main.py` (for 7600, etc.)
    *   For AMD ROCm experimental memory efficient attention:
        ```bash
        TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 python main.py --use-pytorch-cross-attention
        ```

## Useful shortcuts

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

## Workflow Examples

Explore example workflows on the [Examples page](https://comfyanonymous.github.io/ComfyUI_examples/).

## Release Process

[See the release process](https://github.com/comfyanonymous/ComfyUI/releases)

## Tips and Tricks

*   **High-Quality Previews:** Enable higher-quality previews with [TAESD](https://github.com/madebyollin/taesd) by placing the `taesd_decoder.pth` files in the `models/vae_approx` folder and launching ComfyUI with `--preview-method taesd`.
*   **TLS/SSL:** Use `--tls-keyfile key.pem --tls-certfile cert.pem` to enable TLS/SSL. Generate self-signed certificates with `openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -sha256 -days 3650 -nodes -subj "/C=XX/ST=StateName/L=CityName/O=CompanyName/OU=CompanySectionName/CN=CommonNameOrHostname"`
*   **Model Paths:** Configure custom model search paths using the `extra_model_paths.yaml.example` file.  Rename the file to `extra_model_paths.yaml` and modify to your needs.

## Support and Community

*   **Discord:** Get help and provide feedback in the `#help` or `#feedback` channels on the [ComfyUI Discord server](https://www.comfy.org/discord).
*   **Matrix:** Join the [Matrix space: #comfyui_space:matrix.org](https://app.element.io/#/room/%23comfyui_space%3Amatrix.org).
*   **Website:**  Visit [https://www.comfy.org/](https://www.comfy.org/) for more information.

## Frontend Development

The frontend is now in a separate repository: [ComfyUI Frontend](https://github.com/Comfy-Org/ComfyUI_frontend).

*   **Reporting Issues:** Report frontend issues in the [ComfyUI Frontend repository](https://github.com/Comfy-Org/ComfyUI_frontend).
*   **Latest Frontend:** Use `--front-end-version Comfy-Org/ComfyUI_frontend@latest` to use the latest daily release or specify a version like `--front-end-version Comfy-Org/ComfyUI_frontend@1.2.2`.
*   **Legacy Frontend:** Use `--front-end-version Comfy-Org/ComfyUI_legacy_frontend@latest` to access the legacy frontend.

## QA

*   [Which GPU should I buy for this?](https://github.com/comfyanonymous/ComfyUI/wiki/Which-GPU-should-I-buy-for-ComfyUI)

**[Explore the power of ComfyUI and start creating today!](https://github.com/comfyanonymous/ComfyUI)**