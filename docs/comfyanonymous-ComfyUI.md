<div align="center">
  <img src="https://github.com/user-attachments/assets/7ccaf2c1-9b72-41ae-9a89-5688c94b7abe" alt="ComfyUI Screenshot" width="600">
</div>

# ComfyUI: Unleash the Power of Visual AI with a Node-Based Workflow

ComfyUI is a powerful and modular visual AI engine, giving you unparalleled control over your Stable Diffusion pipelines through an intuitive node-based interface.  

[![Website][website-shield]][website-url]
[![Discord][discord-shield]][discord-url]
[![Twitter][twitter-shield]][twitter-url]
[![Matrix][matrix-shield]][matrix-url]
<br>
[![GitHub Release][github-release-shield]][github-release-link]
[![GitHub Release Date][github-release-date-shield]][github-release-link]
[![GitHub Downloads][github-downloads-shield]][github-downloads-link]
[![Latest Downloads][github-downloads-latest-shield]][github-downloads-link]

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
[github-downloads-shield]: https://img.shields/github/downloads/comfyanonymous/ComfyUI/total?style=flat
[github-downloads-latest-shield]: https://img.shields.io/github/downloads/comfyanonymous/ComfyUI/latest/total?style=flat&label=downloads%40latest
[github-downloads-link]: https://github.com/comfyanonymous/ComfyUI/releases

## Key Features

*   **Node-Based Workflow:** Design complex Stable Diffusion pipelines visually with an intuitive graph/node/flowchart interface, eliminating the need for coding.
*   **Extensive Model Support:** Compatible with a wide range of image, video, audio, and 3D models, including:
    *   SD1.x, SD2.x, SDXL, SDXL Turbo, Stable Cascade, SD3 and SD3.5, Pixart Alpha and Sigma, AuraFlow, HunyuanDiT, Flux, Lumina Image 2.0, HiDream, Qwen Image
    *   Omnigen 2, Flux Kontext, HiDream E1.1, Qwen Image Edit
    *   Stable Video Diffusion, Mochi, LTX-Video, Hunyuan Video, Wan 2.1, Wan 2.2
    *   Stable Audio, ACE Step
    *   Hunyuan3D 2.0
*   **Optimization & Efficiency:** Asynchronous queue system, only re-executes changed parts of the workflow, smart memory management for low-VRAM GPUs, and CPU fallback option.
*   **Model Compatibility:** Loads ckpt, safetensors, and other model formats, supports embeddings, LoRAs, hypernetworks, and more.
*   **Workflow Flexibility:** Load and save workflows as JSON, PNG, WebP and FLAC files; includes features like Hires fix, inpainting, ControlNet/T2I-Adapter, and more.
*   **Advanced Features:** Area Composition, Model Merging, LCM models/Loras, latent previews (TAESD), works fully offline, optional API nodes.
*   **Customization:**  Config file (`extra_model_paths.yaml.example`) to set search paths for models.
*   **Offline Functionality:** Core functionality operates entirely offline.
*   **API Nodes:** Optionally use paid models from external providers via the online [Comfy API](https://docs.comfy.org/tutorials/api-nodes/overview).

## Getting Started

Choose your preferred installation method:

*   **[Desktop Application](https://www.comfy.org/download):** Easiest setup for Windows & macOS.
*   **[Windows Portable Package](https://github.com/comfyanonymous/ComfyUI/releases/latest/download/ComfyUI_windows_portable_nvidia.7z):**  Portable, standalone build for Windows.
*   **[Manual Install](#manual-install-windows-linux):**  Supports all operating systems (Windows, Linux, macOS) and GPU types (NVIDIA, AMD, Intel, Apple Silicon, Ascend, Cambricon, Iluvatar) including CPU-only mode.  Requires Python and dependencies.

## Resources

*   **[Examples](https://comfyanonymous.github.io/ComfyUI_examples/):** Explore example workflows.
*   **[Wiki](https://github.com/comfyanonymous/ComfyUI/wiki):** Extensive documentation, tips, and troubleshooting guides.
*   **[Comfy API](https://docs.comfy.org/tutorials/api-nodes/overview):** Documentation for optional API nodes.
*   **[Support and dev channel](#support-and-dev-channel):** Discord, Matrix.
*   **[Which GPU should I buy for this?](https://github.com/comfyanonymous/ComfyUI/wiki/Which-GPU-should-I-buy-for-ComfyUI)**: GPU recommendations.

## Installation Guides

### Windows Portable

[See above for direct link to download](https://github.com/comfyanonymous/ComfyUI/releases/latest/download/ComfyUI_windows_portable_nvidia.7z)
1.  Download, extract with [7-Zip](https://7-zip.org), and run.
2.  Place checkpoints/models in `ComfyUI\models\checkpoints`.
3.  Right-click the file -> properties -> unblock if you have trouble extracting.

### [comfy-cli](https://docs.comfy.org/comfy-cli/getting-started)

Install and start ComfyUI using comfy-cli:

```bash
pip install comfy-cli
comfy install
```

### Manual Install (Windows, Linux, macOS)

1.  **Prerequisites:**
    *   Python 3.13 (or 3.12 if you encounter issues with custom nodes).
    *   Git (for cloning the repository).
2.  **Clone the repository:**
    ```bash
    git clone https://github.com/comfyanonymous/ComfyUI.git
    cd ComfyUI
    ```
3.  **Model Placement:** Place your SD checkpoints (the large `.ckpt` / `.safetensors` files) in the `models/checkpoints` directory and your VAE in `models/vae`.
4.  **Platform-Specific Setup**

    *   **AMD GPUs (Linux):** Install ROCm and PyTorch (if not already installed):
        ```bash
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.4
        ```
        Or for nightly:
        ```bash
        pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.4
        ```

    *   **Intel GPUs (Windows and Linux):**
        *   **(Option 1) PyTorch xpu:**
            ```bash
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
            ```
            Or for nightly:
            ```bash
            pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/xpu
            ```
        *   **(Option 2) Intel Extension for PyTorch (IPEX):** Visit [Installation](https://intel.github.io/intel-extension-for-pytorch/index.html#installation?platform=gpu) for details.

    *   **NVIDIA GPUs:**
        ```bash
        pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu129
        ```
        Or for nightly:
        ```bash
        pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu129
        ```
5.  **Install Dependencies:** Open a terminal in the `ComfyUI` directory and run:
    ```bash
    pip install -r requirements.txt
    ```
6.  **Run ComfyUI:**
    ```bash
    python main.py
    ```

### Running Instructions

*   Navigate to the ComfyUI directory in your terminal.
*   Run the application using the command:

    ```bash
    python main.py
    ```
    *   For AMD cards not officially supported by ROCm, try:
        *   For 6700, 6600 and maybe other RDNA2 or older: `HSA_OVERRIDE_GFX_VERSION=10.3.0 python main.py`
        *   For AMD 7600 and maybe other RDNA3 cards: `HSA_OVERRIDE_GFX_VERSION=11.0.0 python main.py`
    *   AMD ROCm Tips:
        *   Enable experimental memory efficient attention (on recent PyTorch): `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 python main.py --use-pytorch-cross-attention`
        *   Try setting the environment variable: `PYTORCH_TUNABLEOP_ENABLED=1`

## Advanced Installation (Specific Hardware)

*   **Apple Mac silicon:**
    1.  Install PyTorch nightly (see [Accelerated PyTorch training on Mac](https://developer.apple.com/metal/pytorch/)).
    2.  Follow the [ComfyUI manual installation](#manual-install-windows-linux) steps.
    3.  Install the [dependencies](#dependencies).
    4.  Launch using `python main.py`.

*   **DirectML (AMD Cards on Windows):**
    *   (Note: Not recommended due to limited support).
    ```bash
    pip install torch-directml
    python main.py --directml
    ```

*   **Ascend NPUs:** Follow instructions at [Installation](https://ascend.github.io/docs/sources/pytorch/install.html#pytorch).

*   **Cambricon MLUs:** Follow instructions at [Installation](https://www.cambricon.com/docs/sdk_1.15.0/cntoolkit_3.7.2/cntoolkit_install_3.7.2/index.html).

*   **Iluvatar Corex:** Follow instructions at [Installation](https://support.iluvatar.com/#/DocumentCentre?id=1&nameCenter=2&productId=520117912052801536).

## Useful Tips

*   **Graph Execution:** Only connected parts of the graph with inputs will be executed.  Only parts of the graph that change between executions will be re-executed.  Submitting the same graph twice will only run the first time.
*   **Workflow Loading:** Drag generated PNGs onto the webpage to load full workflows (including seeds).
*   **Emphasis:** Use `(good code:1.2)` or `(bad code:0.8)` to change emphasis.
*   **Dynamic Prompts:** Use `{day|night}` for wildcard/dynamic prompts. Use `\\{` or `\\}` to escape.
*   **Embeddings:** Place textual inversions/embeddings in `models/embeddings` and use them in CLIPTextEncode nodes (e.g., `embedding:embedding_filename.pt`).

## High-Quality Previews

*   Use `--preview-method auto` to enable previews.
*   For high-quality previews with [TAESD](https://github.com/madebyollin/taesd), download `taesd_decoder.pth, taesdxl_decoder.pth, taesd3_decoder.pth and taef1_decoder.pth` and place them in the `models/vae_approx` folder. Restart ComfyUI and use `--preview-method taesd`.

## Using TLS/SSL

*   Generate a self-signed certificate (for testing only):
    ```bash
    openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -sha256 -days 3650 -nodes -subj "/C=XX/ST=StateName/L=CityName/O=CompanyName/OU=CompanySectionName/CN=CommonNameOrHostname"
    ```
*   Run with: `--tls-keyfile key.pem --tls-certfile cert.pem` to access via `https://...`.

## Support and dev channel

*   [Discord](https://comfy.org/discord):  #help or #feedback channels.
*   [Matrix space: #comfyui_space:matrix.org](https://app.element.io/#/room/%23comfyui_space%3Amatrix.org).

## Frontend Development

*   The new frontend is now hosted in a separate repository: [ComfyUI Frontend](https://github.com/Comfy-Org/ComfyUI_frontend).
*   For frontend-related issues, use the [ComfyUI Frontend repository](https://github.com/Comfy-Org/ComfyUI_frontend).
*   To use the latest daily frontend release, run `python main.py --front-end-version Comfy-Org/ComfyUI_frontend@latest`.
*   For a specific frontend version, use `python main.py --front-end-version Comfy-Org/ComfyUI_frontend@<version>`.
*   For the legacy frontend: `python main.py --front-end-version Comfy-Org/ComfyUI_legacy_frontend@latest`.

## Keybinds and Shortcuts

| Keybind                            | Description                                                                                                        |
|------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| `Ctrl` + `Enter`                      | Queue graph for generation                                                                                       |
| `Ctrl` + `Shift` + `Enter`              | Queue graph as first for generation                                                                              |
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
| `Space`                              | Move the canvas around                                                                                            |
| `Ctrl`/`Shift` + `Click`                 | Add clicked node to selection                                                                                      |
| `Ctrl` + `C`/`Ctrl` + `V`                  | Copy and paste selected nodes (without connections)                     |
| `Ctrl` + `C`/`Ctrl` + `Shift` + `V`          | Copy and paste selected nodes (with connections) |
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
| `Ctrl` can also be replaced with `Cmd` for macOS users |

## Contributing

Contributions are welcome! Please see the [contributing guidelines](CONTRIBUTING.md) for more information.

[**Visit the ComfyUI GitHub Repository**](https://github.com/comfyanonymous/ComfyUI)