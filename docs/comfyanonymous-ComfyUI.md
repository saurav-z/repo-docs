<div align="center">
  <h1>ComfyUI: Unleash the Power of Visual AI with a Node-Based Workflow</h1>
  <p>Design and execute complex Stable Diffusion pipelines visually with ComfyUI, offering unparalleled flexibility and control.</p>

  [![Website](https://img.shields.io/badge/ComfyOrg-4285F4?style=flat)][website-url]
  [![Discord](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fdiscord.com%2Fapi%2Finvites%2Fcomfyorg%3Fwith_counts%3Dtrue&query=%24.approximate_member_count&logo=discord&logoColor=white&label=Discord&color=green&suffix=%20total)][discord-url]
  [![Twitter](https://img.shields.io/twitter/follow/ComfyUI)][twitter-url]
  [![Matrix](https://img.shields.io/badge/Matrix-000000?style=flat&logo=matrix&logoColor=white)][matrix-url]
  <br>
  [![GitHub Release](https://img.shields.io/github/v/release/comfyanonymous/ComfyUI?style=flat&sort=semver)][github-release-link]
  [![Release Date](https://img.shields.io/github/release-date/comfyanonymous/ComfyUI?style=flat)][github-release-link]
  [![Downloads](https://img.shields.io/github/downloads/comfyanonymous/ComfyUI/total?style=flat)][github-downloads-link]
  [![Latest Downloads](https://img.shields.io/github/downloads/comfyanonymous/ComfyUI/latest/total?style=flat&label=downloads%40latest)][github-downloads-link]

  [website-url]: https://www.comfy.org/
  [discord-url]: https://www.comfy.org/discord
  [twitter-url]: https://x.com/ComfyUI
  [matrix-url]: https://app.element.io/#/room/%23comfyui_space%3Amatrix.org
  [github-release-link]: https://github.com/comfyanonymous/ComfyUI/releases
  [github-downloads-link]: https://github.com/comfyanonymous/ComfyUI/releases
  <br>
  <img src="https://github.com/user-attachments/assets/7ccaf2c1-9b72-41ae-9a89-5688c94b7abe" alt="ComfyUI Screenshot">
</div>

ComfyUI provides a powerful, modular, and user-friendly interface for creating and running advanced AI workflows, especially for Stable Diffusion.  Available on Windows, Linux, and macOS.

## Key Features

*   ✅ **Node-Based Workflow:** Design complex Stable Diffusion pipelines visually using an intuitive graph/nodes/flowchart interface, eliminating the need for extensive coding.
*   ✅ **Model Support:** Extensive support for a wide range of image, video, and audio models, including SD1.x, SD2.x, SDXL, Stable Cascade, and more.
*   ✅ **Optimization:** Efficient execution with asynchronous queueing, smart memory management (including support for low VRAM), and only re-executing changed parts of a workflow.
*   ✅ **Flexible Input:** Supports ckpt, safetensors, embeddings, LoRAs, Hypernetworks, and loading workflows from PNG, WebP, and FLAC files.
*   ✅ **Advanced Features:** Includes support for area composition, inpainting, ControlNet, T2I-Adapter, upscaling, GLIGEN, model merging, LCM models, and latent previews.
*   ✅ **Offline Operation:** Works fully offline, ensuring privacy and control over your data.
*   ✅ **API Integration:** Optional API nodes allow for integration with paid models from external providers.
*   ✅ **Customization:** Uses a [config file](extra_model_paths.yaml.example) for setting model search paths.

Explore the comprehensive capabilities through example workflows on the [Examples page](https://comfyanonymous.github.io/ComfyUI_examples/).

## Getting Started

Choose the installation method that best suits your needs:

*   **[Desktop Application](https://www.comfy.org/download)**: The easiest way to get started, available on Windows & macOS.
*   **[Windows Portable Package](#installing)**: Get the latest commits with a completely portable package (Windows only).
*   **[Manual Install](#manual-install-windows-linux)**:  Supports all operating systems and GPU types (NVIDIA, AMD, Intel, Apple Silicon, Ascend).

## Release Process

ComfyUI is released weekly, primarily on Fridays.  It involves three interconnected repositories:

1.  **[ComfyUI Core](https://github.com/comfyanonymous/ComfyUI)**:  Releases a new stable version.
2.  **[ComfyUI Desktop](https://github.com/Comfy-Org/desktop)**: Builds a new release using the latest stable core version.
3.  **[ComfyUI Frontend](https://github.com/Comfy-Org/ComfyUI_frontend)**:  Weekly frontend updates are merged into the core repository.

## Useful Shortcuts

| Keybind                        | Explanation                                                                                                                                                             |
| ------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Ctrl` + `Enter`                 | Queue the current graph for generation.                                                                                                                               |
| `Ctrl` + `Shift` + `Enter`         | Queue the current graph as the first in the generation queue.                                                                                                       |
| `Ctrl` + `Alt` + `Enter`           | Cancel the current generation.                                                                                                                                       |
| `Ctrl` + `Z`/`Ctrl` + `Y`          | Undo/Redo.                                                                                                                                                           |
| `Ctrl` + `S`                     | Save the workflow.                                                                                                                                                     |
| `Ctrl` + `O`                     | Load a workflow.                                                                                                                                                       |
| `Ctrl` + `A`                     | Select all nodes.                                                                                                                                                      |
| `Alt` + `C`                      | Collapse/uncollapse selected nodes.                                                                                                                                    |
| `Ctrl` + `M`                     | Mute/unmute selected nodes.                                                                                                                                            |
| `Ctrl` + `B`                     | Bypass selected nodes (acts like the node was removed from the graph and the wires reconnected through).                                                              |
| `Delete`/`Backspace`             | Delete selected nodes.                                                                                                                                                 |
| `Ctrl` + `Backspace`             | Delete the current graph.                                                                                                                                              |
| `Space`                        | Move the canvas around (hold and drag with the cursor).                                                                                                                   |
| `Ctrl`/`Shift` + `Click`         | Add the clicked node to the selection.                                                                                                                                |
| `Ctrl` + `C`/`Ctrl` + `V`        | Copy and paste selected nodes (without maintaining connections to outputs of unselected nodes).                                                                        |
| `Ctrl` + `C`/`Ctrl` + `Shift` + `V` | Copy and paste selected nodes (maintaining connections from outputs of unselected nodes to inputs of pasted nodes).                                                   |
| `Shift` + `Drag`                 | Move multiple selected nodes simultaneously.                                                                                                                           |
| `Ctrl` + `D`                     | Load the default graph.                                                                                                                                                |
| `Alt` + `+`                    | Zoom in on the canvas.                                                                                                                                                   |
| `Alt` + `-`                    | Zoom out on the canvas.                                                                                                                                                  |
| `Ctrl` + `Shift` + LMB + Vertical drag | Zoom in/out on the canvas.                                                                                                                                       |
| `P`                            | Pin/Unpin selected nodes.                                                                                                                                                |
| `Ctrl` + `G`                     | Group selected nodes.                                                                                                                                                  |
| `Q`                            | Toggle the visibility of the queue.                                                                                                                                    |
| `H`                            | Toggle the visibility of the history.                                                                                                                                  |
| `R`                            | Refresh the graph.                                                                                                                                                       |
| `F`                            | Show/Hide the menu.                                                                                                                                                      |
| `.`                            | Fit view to selection (or the whole graph if nothing is selected).                                                                                                     |
| Double-Click LMB               | Open the node quick search palette.                                                                                                                                    |
| `Shift` + Drag                 | Move multiple wires simultaneously.                                                                                                                                     |
| `Ctrl` + `Alt` + LMB           | Disconnect all wires from the clicked slot.                                                                                                                               |

*Note: `Ctrl` can be replaced with `Cmd` for macOS users.*

## Installing

### Windows Portable

1.  Download the standalone portable build for Windows from the [releases page](https://github.com/comfyanonymous/ComfyUI/releases).
2.  Extract the downloaded `.7z` file using [7-Zip](https://7-zip.org).
3.  Place your Stable Diffusion checkpoints (ckpt/safetensors) in the `ComfyUI\models\checkpoints` directory.
4.  If you encounter extraction issues, unblock the file by right-clicking it, selecting "Properties," and then checking the "Unblock" box.
5.  To share models with other UIs, use the [Config file](extra_model_paths.yaml.example) in the ComfyUI directory (rename it to `extra_model_paths.yaml`).

### [comfy-cli](https://docs.comfy.org/comfy-cli/getting-started)

Install and start ComfyUI with the `comfy-cli` tool:

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
2.  Place your Stable Diffusion checkpoints (ckpt/safetensors) in the `models/checkpoints` directory and VAEs in `models/vae`.
3.  **AMD GPUs (Linux Only):** Install PyTorch with ROCm (if not already installed):

    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.4
    ```
    (or the nightly build)

4.  **Intel GPUs (Windows and Linux):**

    *   **(Option 1) Intel Arc GPUs:** Install PyTorch with XPU support:

        ```bash
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
        ```
        (or the nightly build)
    *   **(Option 2) Intel Extension for PyTorch (IPEX):** Visit [Installation](https://intel.github.io/intel-extension-for-pytorch/index.html#installation?platform=gpu) for instructions.

5.  **NVIDIA GPUs:** Install PyTorch with CUDA:

    ```bash
    pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu129
    ```
    (or the nightly build)

6.  **Dependencies:** Install required packages:

    ```bash
    pip install -r requirements.txt
    ```
7.  **Apple Mac Silicon:** Follow the instructions under "Others" in the original README.
8.  **Other Hardware Platforms:** Follow the instructions in the original README for DirectML, Ascend NPUs, Cambricon MLUs, and Iluvatar Corex.

## Running

Start ComfyUI from the command line:

```bash
python main.py
```

### AMD ROCm Tips

*   For potential speed improvements, enable experimental memory-efficient attention:

    ```bash
    TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 python main.py --use-pytorch-cross-attention
    ```

*   Try setting the environment variable `PYTORCH_TUNABLEOP_ENABLED=1`.

## Notes

*   Only parts of the graph with all correct inputs will be executed.
*   Only parts of the graph that change from each execution to the next will be executed, if you submit the same graph twice only the first will be executed. If you change the last part of the graph only the part you changed and the part that depends on it will be executed.
*   Dragging a generated png on the webpage or loading one will give you the full workflow including seeds that were used to create it.
*   Use `()` for emphasis (e.g., `(good code:1.2)`). Escape with `\\( or \\)`.
*   Use `{day|night}` for dynamic prompts. Escape with `\\{ or \\}`.
*   Use textual inversion concepts/embeddings in the `models/embeddings` directory and call them in the CLIPTextEncode node (e.g., `embedding:embedding_filename.pt`).

## High-Quality Previews

Enable high-quality previews with [TAESD](https://github.com/madebyollin/taesd):

1.  Download `taesd_decoder.pth`, `taesdxl_decoder.pth`, `taesd3_decoder.pth`, and `taef1_decoder.pth` from [the TAESD repository](https://github.com/madebyollin/taesd/).
2.  Place them in the `models/vae_approx` folder.
3.  Restart ComfyUI and run with `--preview-method taesd`.

## Using TLS/SSL

Generate a self-signed certificate (not for production use):
`openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -sha256 -days 3650 -nodes -subj "/C=XX/ST=StateName/L=CityName/O=CompanyName/OU=CompanySectionName/CN=CommonNameOrHostname"`

Enable TLS/SSL: `--tls-keyfile key.pem --tls-certfile cert.pem`  (access via `https://...`).

## Support and Development

*   **Discord:** Visit the `#help` or `#feedback` channels on the [Discord](https://comfy.org/discord).
*   **Matrix:** Join the [Matrix space](https://app.element.io/#/room/%23comfyui_space%3Amatrix.org).
*   **Website:** [https://www.comfy.org/](https://www.comfy.org/)

## Frontend Development

The frontend is now in a separate repository: [ComfyUI Frontend](https://github.com/Comfy-Org/ComfyUI_frontend).  The compiled JS (from TS/Vue) is under the `web/` directory.

*   **Issue Reporting & Feature Requests:**  Submit frontend-related issues and feature requests in the [ComfyUI Frontend repository](https://github.com/Comfy-Org/ComfyUI_frontend).

*   **Using the Latest Frontend:**
    *   For the latest daily release:  `--front-end-version Comfy-Org/ComfyUI_frontend@latest`
    *   For a specific version: `--front-end-version Comfy-Org/ComfyUI_frontend@1.2.2`

*   **Legacy Frontend:** Use `--front-end-version Comfy-Org/ComfyUI_legacy_frontend@latest`.

## QA

*   Check the [Which GPU should I buy for ComfyUI](https://github.com/comfyanonymous/ComfyUI/wiki/Which-GPU-should-I-buy-for-ComfyUI) page.

---

**[Back to ComfyUI Repository](https://github.com/comfyanonymous/ComfyUI)**