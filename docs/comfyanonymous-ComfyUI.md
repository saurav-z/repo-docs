# ComfyUI: The Powerful and Modular Visual AI Engine

**Unleash your creativity with ComfyUI, the node-based interface that empowers you to build complex AI workflows for image, video, and audio generation.**  [Visit the original repository](https://github.com/comfyanonymous/ComfyUI)

[![Website][website-shield]][website-url]
[![Discord][discord-shield]][discord-url]
[![Twitter][twitter-shield]][twitter-url]
[![Matrix][matrix-shield]][matrix-url]
<br>
[![GitHub Release][github-release-shield]][github-release-link]
[![GitHub Release Date][github-release-date-shield]][github-release-link]
[![GitHub Downloads][github-downloads-shield]][github-downloads-link]
[![GitHub Downloads Latest][github-downloads-latest-shield]][github-downloads-link]

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

ComfyUI provides a powerful and flexible visual interface for creating and executing sophisticated AI pipelines. This innovative approach leverages a node-based system (graph/flowchart) to design and experiment with your workflows, making it accessible to both beginners and experienced users. Compatible with Windows, Linux, and macOS.

## Key Features

*   **Node-Based Workflow:** Design complex Stable Diffusion workflows visually without coding.
*   **Extensive Model Support:**
    *   SD1.x, SD2.x, SDXL, SDXL Turbo, Stable Cascade, SD3 and SD3.5
    *   Pixart Alpha and Sigma, AuraFlow, HunyuanDiT, Flux, Lumina Image 2.0, HiDream, Cosmos Predict2, Qwen Image
    *   Image Editing Models: Omnigen 2, Flux Kontext, HiDream E1.1, Qwen Image Edit
    *   Video Models: Stable Video Diffusion, Mochi, LTX-Video, Hunyuan Video, Nvidia Cosmos and Cosmos Predict2, Wan 2.1, Wan 2.2
    *   Audio Models: Stable Audio, ACE Step
    *   3D Models: Hunyuan3D 2.0
*   **Optimizations:**
    *   Asynchronous queue system for efficient processing.
    *   Only executes changed parts of the workflow.
    *   Smart memory management for low VRAM GPUs.
*   **Model Compatibility:** Supports ckpt, safetensors, all-in-one checkpoints, VAEs, CLIP models, embeddings/textual inversions, LoRAs (regular, locon, loha), and Hypernetworks.
*   **Workflow Management:** Load and save workflows in PNG, WebP, FLAC, and JSON formats.
*   **Advanced Features:**
    *   Area Composition, Inpainting, ControlNet and T2I-Adapter, Upscale Models, GLIGEN, Model Merging, LCM models and Loras.
    *   Latent previews with TAESD.
*   **Offline Functionality:** Operates fully offline; no unnecessary downloads.
*   **API Integration:** Optional API nodes to connect to paid models through the [Comfy API](https://docs.comfy.org/tutorials/api-nodes/overview).
*   **Configuration:** Configurable model search paths using `extra_model_paths.yaml.example`.
*   **Example Workflows:** Explore ready-to-use workflows on the [Examples page](https://comfyanonymous.github.io/ComfyUI_examples/).

## Get Started

ComfyUI offers multiple installation options:

*   **[Desktop Application](https://www.comfy.org/download)**: The easiest way to get started on Windows and macOS.
*   **[Windows Portable Package](#installing)**: Get the latest commits, fully portable, and ready to use on Windows.
*   **[Manual Install](#manual-install-windows-linux)**: Supports all operating systems and GPU types (NVIDIA, AMD, Intel, Apple Silicon, Ascend).
*   **comfy-cli:** install and start ComfyUI easily, check [comfy-cli](https://docs.comfy.org/comfy-cli/getting-started)

## Release Process

ComfyUI follows a weekly release cycle.

1.  **[ComfyUI Core](https://github.com/comfyanonymous/ComfyUI)**: Releases stable versions.
2.  **[ComfyUI Desktop](https://github.com/Comfy-Org/desktop)**: Builds releases using the latest core version.
3.  **[ComfyUI Frontend](https://github.com/Comfy-Org/ComfyUI_frontend)**: Frontend development and merging with the core.

## Quick Start Shortcuts

| Keybind                            | Function                                                                                        |
| ---------------------------------- | ----------------------------------------------------------------------------------------------- |
| `Ctrl` + `Enter`                      | Queue current graph for generation                                                                |
| `Ctrl` + `Shift` + `Enter`              | Queue current graph as first for generation                                                        |
| `Ctrl` + `Alt` + `Enter`                | Cancel current generation                                                                           |
| `Ctrl` + `Z`/`Ctrl` + `Y`                 | Undo/Redo                                                                                           |
| `Ctrl` + `S`                          | Save workflow                                                                                       |
| `Ctrl` + `O`                          | Load workflow                                                                                       |
| `Ctrl` + `A`                          | Select all nodes                                                                                    |
| `Alt `+ `C`                           | Collapse/uncollapse selected nodes                                                                   |
| `Ctrl` + `M`                          | Mute/unmute selected nodes                                                                            |
| `Ctrl` + `B`                           | Bypass selected nodes                                                                              |
| `Delete`/`Backspace`                   | Delete selected nodes                                                                               |
| `Ctrl` + `Backspace`                   | Delete the current graph                                                                            |
| `Space`                              | Move the canvas around                                                                             |
| `Ctrl`/`Shift` + `Click`                 | Add clicked node to selection                                                                       |
| `Ctrl` + `C`/`Ctrl` + `V`                  | Copy and paste selected nodes                                                                      |
| `Ctrl` + `C`/`Ctrl` + `Shift` + `V`          | Copy and paste selected nodes with connections                                                  |
| `Shift` + `Drag`                       | Move multiple selected nodes                                                                       |
| `Ctrl` + `D`                           | Load default graph                                                                                 |
| `Alt` + `+`                          | Canvas Zoom in                                                                                     |
| `Alt` + `-`                          | Canvas Zoom out                                                                                    |
| `Ctrl` + `Shift` + LMB + Vertical drag | Canvas Zoom in/out                                                                                 |
| `P`                                  | Pin/Unpin selected nodes                                                                          |
| `Ctrl` + `G`                           | Group selected nodes                                                                               |
| `Q`                                 | Toggle visibility of the queue                                                                       |
| `H`                                  | Toggle visibility of history                                                                         |
| `R`                                  | Refresh graph                                                                                        |
| `F`                                  | Show/Hide menu                                                                                        |
| `.`                                  | Fit view to selection (Whole graph when nothing is selected)                                           |
| Double-Click LMB                   | Open node quick search palette                                                                      |
| `Shift` + Drag                       | Move multiple wires at once                                                                       |
| `Ctrl` + `Alt` + LMB                   | Disconnect all wires from clicked slot                                                               |

## Installing

### Windows Portable

1.  Download the standalone build from the [releases page](https://github.com/comfyanonymous/ComfyUI/releases).
2.  Extract the archive using [7-Zip](https://7-zip.org).
3.  Place your Stable Diffusion checkpoints in `ComfyUI\models\checkpoints`.
4.  If you have trouble extracting it, right click the file -> properties -> unblock
5.  To use models from other applications:
    *  See the [Config file](extra_model_paths.yaml.example) to set the search paths for models. In the standalone windows build you can find this file in the ComfyUI directory. Rename this file to extra_model_paths.yaml and edit it with your favorite text editor.

### comfy-cli
```bash
pip install comfy-cli
comfy install
```
### Manual Install (Windows, Linux)

1.  Clone the repository:

    ```bash
    git clone https://github.com/comfyanonymous/ComfyUI.git
    cd ComfyUI
    ```
2.  Place your SD checkpoints in `models/checkpoints` and VAEs in `models/vae`.
3.  **Install Dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

4.  **GPU-Specific Instructions:**

    *   **AMD (Linux):** Install ROCm and PyTorch.
        ```bash
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.4
        ```
        or
        ```bash
        pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.4
        ```
    *   **Intel (Windows and Linux):**
        *   Option 1: Install PyTorch xpu.
            ```bash
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
            ```
            or
            ```bash
            pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/xpu
            ```
        *   Option 2: Use Intel Extension for PyTorch (IPEX). See [Installation](https://intel.github.io/intel-extension-for-pytorch/index.html#installation?platform=gpu).
    *   **NVIDIA:**
        ```bash
        pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu129
        ```
        or
        ```bash
        pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu129
        ```
    *   **Troubleshooting:** If you encounter the "Torch not compiled with CUDA enabled" error, uninstall and reinstall PyTorch.

### Others:

#### Apple Mac silicon
1. Install pytorch nightly. For instructions, read the [Accelerated PyTorch training on Mac](https://developer.apple.com/metal/pytorch/) Apple Developer guide (make sure to install the latest pytorch nightly).
2. Follow the [ComfyUI manual installation](#manual-install-windows-linux) instructions for Windows and Linux.
3. Install the ComfyUI [dependencies](#dependencies). If you have another Stable Diffusion UI [you might be able to reuse the dependencies](#i-already-have-another-ui-for-stable-diffusion-installed-do-i-really-have-to-install-all-of-these-dependencies).
4. Launch ComfyUI by running `python main.py`

> **Note**: Remember to add your models, VAE, LoRAs etc. to the corresponding Comfy folders, as discussed in [ComfyUI manual installation](#manual-install-windows-linux).

#### DirectML (AMD Cards on Windows)
```bash
pip install torch-directml
python main.py --directml
```

#### Ascend NPUs
Follow instructions from [installation](https://ascend.github.io/docs/sources/ascend/quick_install.html) page.

#### Cambricon MLUs
Follow instructions from [Installation](https://www.cambricon.com/docs/sdk_1.15.0/cntoolkit_3.7.2/cntoolkit_install_3.7.2/index.html)

#### Iluvatar Corex
Follow instructions from [Installation](https://support.iluvatar.com/#/DocumentCentre?id=1&nameCenter=2&productId=520117912052801536)

## Running

```bash
python main.py
```

### For AMD cards not officially supported by ROCm
For 6700, 6600 and maybe other RDNA2 or older:
```bash
HSA_OVERRIDE_GFX_VERSION=10.3.0 python main.py
```
For AMD 7600 and maybe other RDNA3 cards:
```bash
HSA_OVERRIDE_GFX_VERSION=11.0.0 python main.py
```

### AMD ROCm Tips
Enable memory efficient attention (experimental):
```bash
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 python main.py --use-pytorch-cross-attention
```
You can also try setting this env variable `PYTORCH_TUNABLEOP_ENABLED=1` which might speed things up at the cost of a very slow initial run.

## Additional Notes

*   Only parts of the graph that have an output with all the correct inputs will be executed.
*   Only changed parts of the graph are executed between runs.
*   Drag and drop generated images to load the full workflow.
*   Use `()` for emphasis:  `(good code:1.2)` or `(bad code:0.8)`.
*   Use `// comment` or `/* comment */` for comments.
*   Textual Inversion/Embeddings: Place in `models/embeddings` and use them in the CLIPTextEncode node:  `embedding:embedding_filename.pt`.

## How to show high-quality previews?

Enable previews with `--preview-method auto`.  Install TAESD decoders (`taesd_decoder.pth`, `taesdxl_decoder.pth`, `taesd3_decoder.pth and taef1_decoder.pth`) from [TAESD](https://github.com/madebyollin/taesd/) into `models/vae_approx`.  Then, restart ComfyUI and run with `--preview-method taesd`.

## How to use TLS/SSL?

Generate a self-signed certificate: `openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -sha256 -days 3650 -nodes -subj "/C=XX/ST=StateName/L=CityName/O=CompanyName/OU=CompanySectionName/CN=CommonNameOrHostname"`

Enable TLS/SSL: `--tls-keyfile key.pem --tls-certfile cert.pem`. The app will be accessible via `https://...`.

## Support and Dev Channel

*   [Discord](https://comfy.org/discord): #help or #feedback channels.
*   [Matrix space: #comfyui_space:matrix.org](https://app.element.io/#/room/%23comfyui_space%3Amatrix.org)
*   [https://www.comfy.org/](https://www.comfy.org/)

## Frontend Development

The new frontend is hosted in a separate repository: [ComfyUI Frontend](https://github.com/Comfy-Org/ComfyUI_frontend).

### Reporting Issues and Requesting Features

For frontend issues, use the [ComfyUI Frontend repository](https://github.com/Comfy-Org/ComfyUI_frontend).

### Using the Latest Frontend

*   Launch ComfyUI with `--front-end-version Comfy-Org/ComfyUI_frontend@latest` for the latest daily release.
*   Use `--front-end-version Comfy-Org/ComfyUI_frontend@<version>` for specific versions.

### Accessing the Legacy Frontend

Use `--front-end-version Comfy-Org/ComfyUI_legacy_frontend@latest` to use the legacy frontend.

## QA
### Which GPU should I buy for this?

[See this page for some recommendations](https://github.com/comfyanonymous/ComfyUI/wiki/Which-GPU-should-I-buy-for-ComfyUI)