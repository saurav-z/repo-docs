<div align="center">
  <img src="https://github.com/user-attachments/assets/7ccaf2c1-9b72-41ae-9a89-5688c94b7abe" alt="ComfyUI Screenshot" width="80%">
</div>

# ComfyUI: Unleash Your Creativity with Visual AI 
**Design and execute advanced Stable Diffusion workflows with a user-friendly, node-based interface.**  
[Original Repository](https://github.com/comfyanonymous/ComfyUI)

[![Website][website-shield]][website-url]
[![Discord][discord-shield]][discord-url]
[![Twitter][twitter-shield]][twitter-url]
[![Matrix][matrix-shield]][matrix-url]
<br>
[![GitHub Release][github-release-shield]][github-release-link]
[![Release Date][github-release-date-shield]][github-release-link]
[![Downloads][github-downloads-shield]][github-downloads-link]
[![Downloads Latest][github-downloads-latest-shield]][github-downloads-link]

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

*   **Node-Based Workflow:** Design complex Stable Diffusion pipelines visually using a flexible graph interface, eliminating the need for coding.
*   **Broad Model Support:** Compatible with a wide array of image, video, and audio models, including SD1.x, SD2.x, SDXL, Stable Cascade, Stable Video Diffusion, Stable Audio and many more (full list below).
*   **Advanced Features:** Includes an asynchronous queue system, memory optimization for low-VRAM GPUs, support for checkpoints, LoRAs, Hypernetworks, ControlNet, and more.
*   **Flexible Input & Output:** Load and save workflows as JSON files, and load from generated PNG, WebP, and FLAC files.
*   **Offline Functionality:** Core functionality works entirely offline; optional API nodes integrate with external services.
*   **Community & Support:**  Engage with the ComfyUI community via Discord and Matrix.

## Getting Started

Choose the installation method that suits your needs:

*   **Desktop Application:** The easiest way to get started, available for Windows and macOS.  [Download](https://www.comfy.org/download)
*   **Windows Portable Package:** Get the latest commits and enjoy complete portability on Windows. [Link to Portable Release](https://github.com/comfyanonymous/ComfyUI/releases/latest/download/ComfyUI_windows_portable_nvidia.7z)
*   **Manual Install:** Supports all operating systems and GPU types (NVIDIA, AMD, Intel, Apple Silicon, Ascend).  See below for detailed instructions.

## Examples

Explore the possibilities of ComfyUI with the many [example workflows](https://comfyanonymous.github.io/ComfyUI_examples/).

## Supported Models

*   **Image Models:** SD1.x, SD2.x, SDXL, SDXL Turbo, Stable Cascade, SD3 and SD3.5, Pixart Alpha and Sigma, AuraFlow, HunyuanDiT, Flux, Lumina Image 2.0, HiDream, Qwen Image.
*   **Image Editing Models:** Omnigen 2, Flux Kontext, HiDream E1.1, Qwen Image Edit.
*   **Video Models:** Stable Video Diffusion, Mochi, LTX-Video, Hunyuan Video, Wan 2.1, Wan 2.2.
*   **Audio Models:** Stable Audio, ACE Step.
*   **3D Models:** Hunyuan3D 2.0.

## Manual Install (Windows, Linux)

### Prerequisites

*   Python 3.10, 3.11, 3.12, or 3.13 (3.13 is recommended)
*   Git (for cloning the repository)

### Steps

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/comfyanonymous/ComfyUI.git
    cd ComfyUI
    ```

2.  **Place Models:**
    *   Put your Stable Diffusion checkpoints (ckpt/safetensors files) in the `models/checkpoints` directory.
    *   Place your VAEs in the `models/vae` directory.

3.  **GPU-Specific Instructions**

    *   **NVIDIA:**
        ```bash
        pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu129
        ```
    *   **AMD (Linux Only):**
        ```bash
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.4
        ```
        Alternatively, for nightly builds:
        ```bash
        pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.4
        ```
    *   **Intel:**

        (Option 1) Install PyTorch xpu:
        ```bash
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
        ```
        (Option 2) Use Intel Extension for PyTorch (IPEX). Visit the [Installation](https://intel.github.io/intel-extension-for-pytorch/index.html#installation?platform=gpu) page.
    *   **DirectML (AMD on Windows):**  
        (Note: Not recommended and poorly supported)
        ```bash
        pip install torch-directml
        ```
    *   **Ascend NPUs:** Follow the installation instructions for torch-npu. ([Installation](https://ascend.github.io/docs/sources/pytorch/install.html#pytorch))
    *   **Cambricon MLUs:**  Follow the instructions for torch_mlu. ([Installation](https://www.cambricon.com/docs/sdk_1.15.0/cntoolkit_3.7.2/cntoolkit_install_3.7.2/index.html))
    *   **Iluvatar Corex:** Follow the instructions for Iluvatar Extension for PyTorch. ([Installation](https://support.iluvatar.com/#/DocumentCentre?id=1&nameCenter=2&productId=520117912052801536))

4.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

5.  **Run ComfyUI:**

    ```bash
    python main.py
    ```

    *   For AMD cards not officially supported by ROCm:
        *   For 6700, 6600, and older RDNA2: `HSA_OVERRIDE_GFX_VERSION=10.3.0 python main.py`
        *   For AMD 7600 and newer RDNA3: `HSA_OVERRIDE_GFX_VERSION=11.0.0 python main.py`
    *   To enable experimental memory efficient attention on recent pytorch on some AMD GPUs:  `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 python main.py --use-pytorch-cross-attention`

## Keybinds

*   Ctrl + Enter: Queue generation
*   Ctrl + Shift + Enter: Queue as first
*   Ctrl + Alt + Enter: Cancel generation
*   Ctrl + Z / Ctrl + Y: Undo/Redo
*   Ctrl + S: Save workflow
*   Ctrl + O: Load workflow
*   Ctrl + A: Select all
*   Alt + C: Collapse/Uncollapse
*   Ctrl + M: Mute/Unmute
*   Ctrl + B: Bypass
*   Delete/Backspace: Delete nodes
*   Ctrl + Backspace: Delete graph
*   Space: Pan canvas
*   Ctrl/Shift + Click: Add to selection
*   Ctrl + C / Ctrl + V: Copy/Paste
*   Ctrl + C / Ctrl + Shift + V: Copy/Paste with connections
*   Shift + Drag: Move multiple nodes
*   Ctrl + D: Load default graph
*   Alt + + / Alt + -: Zoom
*   Ctrl + Shift + LMB + Vertical drag: Zoom
*   P: Pin/Unpin
*   Ctrl + G: Group
*   Q: Toggle queue
*   H: Toggle history
*   R: Refresh graph
*   F: Show/Hide menu
*   .: Fit view to selection
*   Double-click: Open quick search
*   Shift + Drag: Move multiple wires
*   Ctrl + Alt + LMB: Disconnect wires

(Replace Ctrl with Cmd on macOS)

## How to show high-quality previews?

Use `--preview-method auto` to enable previews. To enable higher-quality previews with [TAESD](https://github.com/madebyollin/taesd), download the [taesd_decoder.pth, taesdxl_decoder.pth, taesd3_decoder.pth and taef1_decoder.pth](https://github.com/madebyollin/taesd/) and place them in the `models/vae_approx` folder. Once they're installed, restart ComfyUI and launch it with `--preview-method taesd` to enable high-quality previews.

## Support and Development

*   **Discord:** Join the ComfyUI community for help and feedback:  [Discord](https://comfy.org/discord)
*   **Matrix:** [Matrix Space](https://app.element.io/#/room/%23comfyui_space%3Amatrix.org)

## Frontend Development

As of August 15, 2024, the new frontend is hosted in a separate repository: [ComfyUI Frontend](https://github.com/Comfy-Org/ComfyUI_frontend).

### Reporting Issues and Requesting Features

For any frontend-related bugs, issues, or feature requests, please use the [ComfyUI Frontend repository](https://github.com/Comfy-Org/ComfyUI_frontend).

### Using the Latest Frontend

To use the latest frontend:

1.  Launch ComfyUI with:
    ```
    --front-end-version Comfy-Org/ComfyUI_frontend@latest
    ```

2.  For a specific version:
    ```
    --front-end-version Comfy-Org/ComfyUI_frontend@1.2.2
    ```

### Accessing the Legacy Frontend

If needed, use:
```
--front-end-version Comfy-Org/ComfyUI_legacy_frontend@latest