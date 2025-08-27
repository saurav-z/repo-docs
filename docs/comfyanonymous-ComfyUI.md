# ComfyUI: Unleash Your Creativity with Visual AI

**ComfyUI is the leading visual AI engine, empowering users to build and execute complex Stable Diffusion workflows with its intuitive node-based interface.** Explore endless possibilities in image, video, and audio generation without writing a single line of code. [View the original repository.](https://github.com/comfyanonymous/ComfyUI)

## Key Features:

*   **Node-Based Workflow:** Design intricate Stable Diffusion pipelines using a visual, graph-based interface for unparalleled control and flexibility.
*   **Extensive Model Support:** Compatible with a wide range of models, including SD1.x, SD2.x, SDXL, Stable Cascade, SD3/SD3.5, and many more.
*   **Image Editing and Video Generation:** Utilize image editing models like Omnigen 2 and video generation models such as Stable Video Diffusion for advanced creative projects.
*   **Audio Generation:** Generate audio using Stable Audio and ACE Step.
*   **3D Model Support:** Explore 3D model capabilities with Hunyuan3D 2.0.
*   **Asynchronous Queue System:** Efficiently manage and execute multiple tasks concurrently.
*   **Optimized Performance:** Benefit from smart memory management, asynchronous queues, and efficient execution of modified workflows.
*   **Wide Hardware Compatibility:** Supports various hardware including NVIDIA, AMD, Intel, Apple Silicon, and Ascend.
*   **Checkpoint & File Support:** Load and utilize .ckpt, .safetensors, .pt, and other model file formats.
*   **Workflow Management:** Load, save, and share workflows as JSON files, including support for seeds.
*   **Advanced Features:** Integrate features like Inpainting, ControlNet, Upscale Models, GLIGEN, Model Merging, LCM, and more.
*   **Offline Capabilities:** Operates fully offline, ensuring privacy and control over your data.
*   **Optional API Integration:** Access paid models from external providers via the Comfy API.
*   **Workflow Examples:** Get started quickly with pre-built workflows.

## Installation

Choose your preferred installation method:

*   **Desktop Application:** Easiest for beginners; available for Windows & macOS. [Download](https://www.comfy.org/download)
*   **Windows Portable Package:** Latest commits, fully portable, Windows only.
    *   [Direct Download](https://github.com/comfyanonymous/ComfyUI/releases/latest/download/ComfyUI_windows_portable_nvidia.7z)
    *   Place checkpoints/models in `ComfyUI\models\checkpoints`
    *   Configure model paths using `extra_model_paths.yaml` in the ComfyUI directory.
*   **comfy-cli:** Quick command-line installation.

    ```bash
    pip install comfy-cli
    comfy install
    ```

*   **Manual Install (Windows, Linux, macOS):** Supports all OS & GPU types.

    1.  Clone the repository.
    2.  Place SD checkpoints in `models/checkpoints` and VAEs in `models/vae`.
    3.  Follow hardware-specific instructions for AMD, Intel, NVIDIA, Apple Silicon, Ascend, Cambricon MLUs, and Iluvatar Corex GPUs.
    4.  Install dependencies: `pip install -r requirements.txt`
    5.  Run ComfyUI: `python main.py`

## Getting Started

*   **Examples:** Explore pre-built workflows and tutorials. ([Examples](https://comfyanonymous.github.io/ComfyUI_examples/))
*   **Shortcuts:** Quickly master the interface using these keyboard shortcuts:

    | Keybind                 | Action                                    |
    | ----------------------- | ----------------------------------------- |
    | `Ctrl + Enter`          | Queue generation                          |
    | `Ctrl + Shift + Enter`  | Queue as first                            |
    | `Ctrl + Alt + Enter`    | Cancel generation                         |
    | `Ctrl + Z`/`Ctrl + Y`   | Undo/Redo                                 |
    | ... (many more)         | ... (see original for full list)          |

## Advanced Usage & Tips

*   **High-Quality Previews:** Enable better previews using `--preview-method auto`.  For high-quality previews use TAESD: Place `taesd_decoder.pth`, `taesdxl_decoder.pth`, `taesd3_decoder.pth and taef1_decoder.pth` in the `models/vae_approx` folder, then restart with `--preview-method taesd`.
*   **TLS/SSL:** Secure your connection with `--tls-keyfile key.pem --tls-certfile cert.pem`.
*   **AMD ROCm Tips:** Consider experimental settings for AMD GPUs to increase performance.
*   **Dynamic Prompts:** Utilize wildcard prompts and C-style comments in your prompts.

## Frontend Development

*   The new frontend is in a separate repository: [ComfyUI Frontend](https://github.com/Comfy-Org/ComfyUI_frontend).
*   For the latest, use `--front-end-version Comfy-Org/ComfyUI_frontend@latest`.
*   Legacy frontend: `--front-end-version Comfy-Org/ComfyUI_legacy_frontend@latest`.

## Support

*   [Discord](https://comfy.org/discord): Get help from the community.
*   [Matrix Space](https://app.element.io/#/room/%23comfyui_space%3Amatrix.org)

## Contributing

ComfyUI welcomes contributions! Please refer to the [contribution guidelines](CONTRIBUTING.md).

## Hardware Recommendations

*   GPU Recommendations: Consult the [GPU Recommendations Wiki Page](https://github.com/comfyanonymous/ComfyUI/wiki/Which-GPU-should-I-buy-for-ComfyUI) for advice on selecting the right GPU.