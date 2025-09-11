<div align="center">
  <h1>ComfyUI: The Visual AI Engine for Cutting-Edge Image Generation</h1>
  <p>Unlock unparalleled control and flexibility in your AI art creation with ComfyUI, a powerful and modular visual interface for Stable Diffusion and beyond. <a href="https://github.com/comfyanonymous/ComfyUI">Explore the original repository here</a>.</p>

  [![Website](https://img.shields.io/badge/ComfyOrg-4285F4?style=flat)][website-url]
  [![Discord](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fdiscord.com%2Fapi%2Finvites%2Fcomfyorg%3Fwith_counts%3Dtrue&query=%24.approximate_member_count&logo=discord&logoColor=white&label=Discord&color=green&suffix=%20total)][discord-url]
  [![Twitter](https://img.shields.io/twitter/follow/ComfyUI)][twitter-url]
  [![Matrix](https://img.shields.io/badge/Matrix-000000?style=flat&logo=matrix&logoColor=white)][matrix-url]
  <br>
  [![GitHub Release](https://img.shields.io/github/v/release/comfyanonymous/ComfyUI?style=flat&sort=semver)][github-release-link]
  [![Release Date](https://img.shields.io/github/release-date/comfyanonymous/ComfyUI?style=flat)][github-release-link]
  [![Downloads](https://img.shields.io/github/downloads/comfyanonymous/ComfyUI/total?style=flat)][github-downloads-link]
  [![Downloads Latest](https://img.shields.io/github/downloads/comfyanonymous/ComfyUI/latest/total?style=flat&label=downloads%40latest)][github-downloads-link]

  [website-url]: https://www.comfy.org/
  [discord-url]: https://www.comfy.org/discord
  [twitter-url]: https://x.com/ComfyUI
  [matrix-url]: https://app.element.io/#/room/%23comfyui_space%3Amatrix.org
  [github-release-link]: https://github.com/comfyanonymous/ComfyUI/releases
  [github-downloads-link]: https://github.com/comfyanonymous/ComfyUI/releases

  <img src="https://github.com/user-attachments/assets/7ccaf2c1-9b72-41ae-9a89-5688c94b7abe" alt="ComfyUI Screenshot" width="80%">
</div>

## Key Features of ComfyUI

*   **Node-Based Workflow:** Visually design and execute complex Stable Diffusion workflows using an intuitive node-based interface.
*   **Wide Model Support:**
    *   **Image Generation:** SD1.x, SD2.x, SDXL, SDXL Turbo, Stable Cascade, SD3/SD3.5, and more.
    *   **Image Editing:** Comprehensive support for editing models like Omnigen 2, Flux Kontext, and others.
    *   **Video Generation:** Stable Video Diffusion, Mochi, LTX-Video, and more.
    *   **Audio Generation:** Stable Audio, ACE Step, and more.
    *   **3D Generation:** Hunyuan3D 2.0
*   **Optimized Performance:** Asynchronous queue, only re-executes changes, smart memory management for low-VRAM GPUs, CPU fallback.
*   **Model Compatibility:** Load ckpt, safetensors, embeddings, LoRAs, Hypernetworks, and more.
*   **Workflow Management:** Load/save workflows as JSON or PNG files, supports advanced features like Hires fix, inpainting, ControlNet, and upscaling.
*   **Offline Functionality:** Works fully offline, with optional API nodes for integrating paid models.
*   **Preview Quality:** Options for high-quality previews with TAESD integration.
*   **Configurability:**  Utilize a config file to define search paths for models.

## Getting Started

Choose your preferred installation method:

*   **Desktop Application:** Easiest to start, available on Windows & macOS.  Download at [Desktop Application](https://www.comfy.org/download)
*   **Windows Portable Package:** Latest commits, fully portable.
*   **Manual Install:** Supports all OS and GPU types (NVIDIA, AMD, Intel, Apple Silicon, Ascend).

## Additional Resources

*   **Examples:** Explore example workflows to see ComfyUI in action:  [Examples](https://comfyanonymous.github.io/ComfyUI_examples/)
*   **Shortcuts:** Use the following keyboard shortcuts for efficient workflow.
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
Follow the instructions for installing ComfyUI based on your operating system and setup:
*   **Windows Portable** Follow instructions at [Windows Portable](#windows-portable)
*   **comfy-cli** Follow instructions at [comfy-cli](#comfy-cli)
*   **Manual Install (Windows, Linux)** Follow instructions at [Manual Install (Windows, Linux)](#manual-install-windows-linux)

## Release Process
ComfyUI follows a weekly release cycle targeting Friday. There are three interconnected repositories:
1.  **[ComfyUI Core](https://github.com/comfyanonymous/ComfyUI)** Releases a new stable version (e.g., v0.7.0)
2.  **[ComfyUI Desktop](https://github.com/Comfy-Org/desktop)** Builds a new release using the latest stable core version
3.  **[ComfyUI Frontend](https://github.com/Comfy-Org/ComfyUI_frontend)** Weekly frontend updates are merged into the core repository.

## Frontend Development

*   The new frontend is hosted in a separate repository: [ComfyUI Frontend](https://github.com/Comfy-Org/ComfyUI_frontend).
*   For frontend-specific bugs, issues, or feature requests, please use the [ComfyUI Frontend repository](https://github.com/Comfy-Org/ComfyUI_frontend).
*   To use the latest daily release, launch ComfyUI with the following command line argument:
    ```
    --front-end-version Comfy-Org/ComfyUI_frontend@latest
    ```
*   To use a specific version, replace `latest` with the desired version number:
    ```
    --front-end-version Comfy-Org/ComfyUI_frontend@1.2.2
    ```
*   If you need to use the legacy frontend, you can access it using the following command line argument:
    ```
    --front-end-version Comfy-Org/ComfyUI_legacy_frontend@latest
    ```

## Troubleshooting and Support

*   **How to show high-quality previews?** Use ```--preview-method auto``` or install TAESD and use  ```--preview-method taesd```
*   **How to use TLS/SSL?** Use `--tls-keyfile key.pem --tls-certfile cert.pem`
*   **Support and dev channel**:  [Discord](https://comfy.org/discord), [Matrix space: #comfyui_space:matrix.org](https://app.element.io/#/room/%23comfyui_space%3Amatrix.org)

## QA

*   **Which GPU should I buy for this?** [See this page for some recommendations](https://github.com/comfyanonymous/ComfyUI/wiki/Which-GPU-should-I-buy-for-ComfyUI)