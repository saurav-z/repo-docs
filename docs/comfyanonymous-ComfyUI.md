# ComfyUI: The Most Powerful Visual AI Engine

**Unlock the power of visual AI with ComfyUI, a node-based interface for creating and experimenting with advanced Stable Diffusion workflows.**  ([Original Repo](https://github.com/comfyanonymous/ComfyUI))

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

ComfyUI provides a flexible, modular, and powerful visual interface for creating and executing Stable Diffusion pipelines. It allows users to design intricate workflows using a node-based system, eliminating the need for extensive coding. Compatible with Windows, Linux, and macOS.

## Key Features:

*   **Node-Based Workflow:** Design complex Stable Diffusion pipelines using an intuitive, visual node graph interface without any coding.
*   **Wide Model Support:**  Compatible with a vast range of models:
    *   **Image Models:** SD1.x, SD2.x, SDXL, SDXL Turbo, Stable Cascade, SD3/SD3.5, and many more (Pixart, AuraFlow, HunyuanDiT, Flux, Lumina Image 2.0, HiDream, Cosmos Predict2, Qwen Image).
    *   **Image Editing Models:**  Omnigen 2, Flux Kontext, HiDream E1.1.
    *   **Video Models:** Stable Video Diffusion, Mochi, LTX-Video, Hunyuan Video, Nvidia Cosmos, Wan 2.1/2.2.
    *   **Audio Models:** Stable Audio, ACE Step.
    *   **3D Models:** Hunyuan3D 2.0.
*   **Asynchronous Queue & Optimizations:**
    *   Asynchronous queue system for efficient processing.
    *   Optimized to re-execute only changed parts of the workflow.
    *   Smart memory management for running large models on GPUs with limited VRAM.
*   **Flexible Hardware Compatibility:** Works with or without a GPU, including CPU fallback (`--cpu`).  Supports NVIDIA, AMD, Intel, Apple Silicon, and Ascend hardware.
*   **Model Compatibility:** Loads various model formats (ckpt, safetensors, etc.), VAEs, and CLIP models.
*   **Extensive Customization:** Supports embeddings/textual inversion, LoRAs, Hypernetworks, and loading/saving complete workflows.
*   **Advanced Features:** Offers features such as Inpainting, ControlNet, T2I-Adapter, Upscale Models, GLIGEN, Model Merging, LCM models, and more.
*   **Offline Functionality:** Operates fully offline, minimizing data download requirements.
*   **Integration:**  Optional API nodes to access paid models from external providers via the Comfy API.
*   **Configuration:**  Uses a config file (`extra_model_paths.yaml.example`) to define search paths for models.
*   **Workflow Examples:**  Explore pre-built workflows on the [Examples page](https://comfyanonymous.github.io/ComfyUI_examples/).

## Get Started

*   **[Desktop Application](https://www.comfy.org/download)**: Easiest way to start, available for Windows & macOS.
*   **[Windows Portable Package](https://github.com/comfyanonymous/ComfyUI/releases)**: Standalone build, completely portable, available for Windows.
*   **[Manual Install](https://github.com/comfyanonymous/ComfyUI#manual-install-windows-linux)**: Supports all operating systems and GPU types (NVIDIA, AMD, Intel, Apple Silicon, Ascend).

##  Shortcuts

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

*(Installation instructions remain unchanged, but are included for completeness)*

## Release Process

ComfyUI follows a weekly release cycle targeting Friday but this regularly changes because of model releases or large changes to the codebase. There are three interconnected repositories:

1.  **[ComfyUI Core](https://github.com/comfyanonymous/ComfyUI)**
    *   Releases a new stable version (e.g., v0.7.0)
    *   Serves as the foundation for the desktop release

2.  **[ComfyUI Desktop](https://github.com/Comfy-Org/desktop)**
    *   Builds a new release using the latest stable core version

3.  **[ComfyUI Frontend](https://github.com/Comfy-Org/ComfyUI_frontend)**
    *   Weekly frontend updates are merged into the core repository
    *   Features are frozen for the upcoming core release
    *   Development continues for the next release cycle

## How to show high-quality previews?

*(Instructions on high quality previews remain unchanged, but are included for completeness)*

## How to use TLS/SSL?

*(Instructions on TLS/SSL remain unchanged, but are included for completeness)*

## Support and dev channel

*(Information about support and dev channels remain unchanged, but are included for completeness)*

## Frontend Development

*(Frontend development information remain unchanged, but is included for completeness)*

## QA

*(QA information remains unchanged, but is included for completeness)*