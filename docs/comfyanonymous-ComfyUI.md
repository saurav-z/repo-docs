<div align="center">

# ComfyUI: Unleash Your Creativity with Visual AI

**ComfyUI is a powerful and modular visual AI engine, providing an intuitive node-based interface for creating and executing intricate Stable Diffusion workflows.** Explore the endless possibilities of AI image generation with this open-source powerhouse.  Get started today by visiting the [ComfyUI GitHub Repository](https://github.com/comfyanonymous/ComfyUI).

[![Website][website-shield]][website-url]
[![Dynamic JSON Badge][discord-shield]][discord-url]
[![Twitter][twitter-shield]][twitter-url]
[![Matrix][matrix-shield]][matrix-url]
<br>
[![][github-release-shield]][github-release-link]
[![][github-release-date-shield]][github-release-link]
[![][github-downloads-shield]][github-downloads-link]
[![][github-downloads-latest-shield]][github-downloads-link]

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

*   **Node-Based Interface:** Design and experiment with complex Stable Diffusion workflows using an intuitive, visual interface without coding.
*   **Wide Model Support:**  Compatible with a broad range of image, video, and audio models, including SD1.x, SD2.x, SDXL, Stable Cascade, and more.
*   **Flexible Workflow Design:**  Utilize an asynchronous queue system and smart memory management for efficient execution, even on GPUs with limited VRAM.
*   **Advanced Features:**  Includes support for ControlNet, T2I-Adapter, upscaling models, model merging, and more, empowering you to refine your creations.
*   **Offline Functionality:**  Operate fully offline with core components, offering enhanced privacy and control.
*   **Customization:** Load a variety of model types (ckpt, safetensors), and use LORAs, Hypernetworks, embeddings, and more.
*   **Workflow Sharing:** Import and export complete workflows via PNG, WebP, and JSON files, enabling easy collaboration and reuse.

## Getting Started

Choose the installation method that best suits your needs:

*   **[Desktop Application](https://www.comfy.org/download):** Easiest way to get started, available for Windows and macOS.
*   **[Windows Portable Package](#installing):**  Get the latest commits with a completely portable version, available on Windows.
*   **[Manual Install](#manual-install-windows-linux):** Supports all operating systems and GPU types (NVIDIA, AMD, Intel, Apple Silicon, Ascend).

## Examples

Explore what's possible with ComfyUI by browsing the [example workflows](https://comfyanonymous.github.io/ComfyUI_examples/).

## Release Process

ComfyUI follows a weekly release cycle, typically targeting Fridays. Three interconnected repositories contribute to the project's development:

1.  **[ComfyUI Core](https://github.com/comfyanonymous/ComfyUI):**  Releases stable versions (e.g., v0.7.0), serving as the foundation for the desktop release.
2.  **[ComfyUI Desktop](https://github.com/Comfy-Org/desktop):**  Builds new releases based on the latest stable core version.
3.  **[ComfyUI Frontend](https://github.com/Comfy-Org/ComfyUI_frontend):** Hosts the user interface.  Weekly frontend updates are merged into the core repository, with features frozen for the upcoming release.

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

# Installing

## Windows Portable

Download and extract, put your checkpoints/models in the models/checkpoints folder, and run.

### [Direct link to download](https://github.com/comfyanonymous/ComfyUI/releases/latest/download/ComfyUI_windows_portable_nvidia.7z)

If you have trouble extracting it, right click the file -> properties -> unblock

#### How do I share models between another UI and ComfyUI?

See the [Config file](extra_model_paths.yaml.example) to set the search paths for models. In the standalone windows build you can find this file in the ComfyUI directory. Rename this file to extra_model_paths.yaml and edit it with your favorite text editor.

## [comfy-cli](https://docs.comfy.org/comfy-cli/getting-started)

```bash
pip install comfy-cli
comfy install
```

## Manual Install (Windows, Linux)

python 3.13 is supported but using 3.12 is recommended because some custom nodes and their dependencies might not support it yet.

Git clone this repo.

Put your SD checkpoints (the huge ckpt/safetensors files) in: models/checkpoints

Put your VAE in: models/vae


### AMD GPUs (Linux only)
AMD users can install rocm and pytorch with pip if you don't have it already installed, this is the command to install the stable version:

```pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.4```

This is the command to install the nightly with ROCm 6.4 which might have some performance improvements:

```pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.4```

### Intel GPUs (Windows and Linux)

(Option 1) Intel Arc GPU users can install native PyTorch with torch.xpu support using pip. More information can be found [here](https://pytorch.org/docs/main/notes/get_start_xpu.html)

1. To install PyTorch xpu, use the following command:

```pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu```

This is the command to install the Pytorch xpu nightly which might have some performance improvements:

```pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/xpu```

(Option 2) Alternatively, Intel GPUs supported by Intel Extension for PyTorch (IPEX) can leverage IPEX for improved performance.

1. visit [Installation](https://intel.github.io/intel-extension-for-pytorch/index.html#installation?platform=gpu) for more information.

### NVIDIA

Nvidia users should install stable pytorch using this command:

```pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu129```

This is the command to install pytorch nightly instead which might have performance improvements.

```pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu129```

#### Troubleshooting

If you get the "Torch not compiled with CUDA enabled" error, uninstall torch with:

```pip uninstall torch```

And install it again with the command above.

### Dependencies

```pip install -r requirements.txt```

### Others:

#### Apple Mac silicon

1. Install pytorch nightly. For instructions, read the [Accelerated PyTorch training on Mac](https://developer.apple.com/metal/pytorch/) Apple Developer guide (make sure to install the latest pytorch nightly).
1. Follow the [ComfyUI manual installation](#manual-install-windows-linux) instructions for Windows and Linux.
1. Install the ComfyUI [dependencies](#dependencies). If you have another Stable Diffusion UI [you might be able to reuse the dependencies](#i-already-have-another-ui-for-stable-diffusion-installed-do-i-really-have-to-install-all-of-these-dependencies).
1. Launch ComfyUI by running `python main.py`

> **Note**: Remember to add your models, VAE, LoRAs etc. to the corresponding Comfy folders, as discussed in [ComfyUI manual installation](#manual-install-windows-linux).

#### DirectML (AMD Cards on Windows)

```pip install torch-directml``` Then you can launch ComfyUI with: ```python main.py --directml```

#### Ascend NPUs

Follow the [installation](https://ascend.github.io/docs/sources/ascend/quick_install.html) page instructions for your platform, then follow the [ComfyUI manual installation](#manual-install-windows-linux) guide.

#### Cambricon MLUs

1. Install the Cambricon CNToolkit by adhering to the platform-specific instructions on the [Installation](https://www.cambricon.com/docs/sdk_1.15.0/cntoolkit_3.7.2/cntoolkit_install_3.7.2/index.html)
2. Next, install the PyTorch(torch_mlu) following the instructions on the [Installation](https://www.cambricon.com/docs/sdk_1.15.0/cambricon_pytorch_1.17.0/user_guide_1.9/index.html)
3. Launch ComfyUI by running `python main.py`

#### Iluvatar Corex

1. Install the Iluvatar Corex Toolkit by adhering to the platform-specific instructions on the [Installation](https://support.iluvatar.com/#/DocumentCentre?id=1&nameCenter=2&productId=520117912052801536)
2. Launch ComfyUI by running `python main.py`

# Running

```python main.py```

### For AMD cards not officially supported by ROCm

```HSA_OVERRIDE_GFX_VERSION=10.3.0 python main.py```

or for RDNA3 cards:
```HSA_OVERRIDE_GFX_VERSION=11.0.0 python main.py```

### AMD ROCm Tips

```TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 python main.py --use-pytorch-cross-attention```

```PYTORCH_TUNABLEOP_ENABLED=1 python main.py```

# Notes

Only parts of the graph with correct inputs will be executed.

Only parts of the graph that change from each execution to the next will be executed.

Dragging a generated png on the webpage or loading one will give you the full workflow including seeds that were used to create it.

Use () for emphasis and \\( \\) for escaping these characters.
Use {} for dynamic prompts and \\{ \\} for escaping these characters.

Use embedding:embedding_filename.pt in CLIPTextEncode node.

## How to show high-quality previews?

```--preview-method auto```

Download [taesd_decoder.pth, taesdxl_decoder.pth, taesd3_decoder.pth and taef1_decoder.pth](https://github.com/madebyollin/taesd/) to `models/vae_approx` and launch with `--preview-method taesd` for high-quality previews.

## How to use TLS/SSL?

Generate a self-signed certificate and key: `openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -sha256 -days 3650 -nodes -subj "/C=XX/ST=StateName/L=CityName/O=CompanyName/OU=CompanySectionName/CN=CommonNameOrHostname"`

Use `--tls-keyfile key.pem --tls-certfile cert.pem` to enable TLS/SSL.

> Note: Windows users can use [alexisrolland/docker-openssl](https://github.com/alexisrolland/docker-openssl) or one of the [3rd party binary distributions](https://wiki.openssl.org/index.php/Binaries) to run the command example above.
<br/><br/>If you use a container, note that the volume mount `-v` can be a relative path so `... -v ".\:/openssl-certs" ...` would create the key & cert files in the current directory of your command prompt or powershell terminal.

## Support and dev channel

[Discord](https://comfy.org/discord)

[Matrix space: #comfyui_space:matrix.org](https://app.element.io/#/room/%23comfyui_space%3Amatrix.org)

See also: [https://www.comfy.org/](https://www.comfy.org/)

## Frontend Development

[ComfyUI Frontend](https://github.com/Comfy-Org/ComfyUI_frontend) now hosts the frontend.

### Reporting Issues and Requesting Features

Use the [ComfyUI Frontend repository](https://github.com/Comfy-Org/ComfyUI_frontend) for frontend-specific bugs.

### Using the Latest Frontend

Use `--front-end-version Comfy-Org/ComfyUI_frontend@latest` for the latest release.
Use `--front-end-version Comfy-Org/ComfyUI_frontend@1.2.2` for a specific version.

### Accessing the Legacy Frontend

Use `--front-end-version Comfy-Org/ComfyUI_legacy_frontend@latest`

# QA

### Which GPU should I buy for this?

[See this page for some recommendations](https://github.com/comfyanonymous/ComfyUI/wiki/Which-GPU-should-I-buy-for-ComfyUI)