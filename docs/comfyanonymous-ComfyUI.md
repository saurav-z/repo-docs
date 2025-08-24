<div align="center">

# ComfyUI: The Most Powerful and Modular Visual AI Engine 🚀

**Unleash your creativity with ComfyUI, a node-based interface for designing and executing advanced Stable Diffusion workflows!**

[![Website][website-shield]][website-url]
[![Discord][discord-shield]][discord-url]
[![Twitter][twitter-shield]][twitter-url]
[![Matrix][matrix-shield]][matrix-url]
<br>
[![GitHub Release][github-release-shield]][github-release-link]
[![Release Date][github-release-date-shield]][github-release-link]
[![Total Downloads][github-downloads-shield]][github-downloads-link]
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
[github-downloads-shield]: https://img.shields/github/downloads/comfyanonymous/ComfyUI/total?style=flat
[github-downloads-latest-shield]: https://img.shields.io/github/downloads/comfyanonymous/ComfyUI/latest/total?style=flat&label=downloads%40latest
[github-downloads-link]: https://github.com/comfyanonymous/ComfyUI/releases

![ComfyUI Screenshot](https://github.com/user-attachments/assets/7ccaf2c1-9b72-41ae-9a89-5688c94b7abe)
</div>

ComfyUI empowers you to create complex AI image, video, and audio workflows using a user-friendly, node-based interface.  Available on Windows, Linux, and macOS. Explore cutting-edge features and experiment with various models without writing a single line of code!

## Key Features

*   **Node-Based Workflow:** Design intricate Stable Diffusion pipelines with a visual, drag-and-drop interface.
*   **Extensive Model Support:** Access a vast library of models for image generation, editing, video creation, and audio processing.
    *   **Image Models:** SD1.x, SD2.x (unCLIP), SDXL, SDXL Turbo, Stable Cascade, SD3 and SD3.5, Pixart Alpha and Sigma, AuraFlow, HunyuanDiT, Flux, Lumina Image 2.0, HiDream, Cosmos Predict2, Qwen Image.
    *   **Image Editing Models:** Omnigen 2, Flux Kontext, HiDream E1.1, Qwen Image Edit.
    *   **Video Models:** Stable Video Diffusion, Mochi, LTX-Video, Hunyuan Video, Nvidia Cosmos and Cosmos Predict2, Wan 2.1, Wan 2.2.
    *   **Audio Models:** Stable Audio, ACE Step.
    *   **3D Models:** Hunyuan3D 2.0
*   **Optimized Performance:**  Enjoy an asynchronous queue system, smart memory management, and optimizations to minimize execution time.
*   **Hardware Flexibility:** Run ComfyUI even without a GPU using the `--cpu` flag.
*   **Model Compatibility:** Load ckpt, safetensors, and other model formats, along with VAEs and CLIP models.
*   **Advanced Features:** Explore embeddings/textual inversion, LoRAs, Hypernetworks, area composition, inpainting, ControlNet, and more.
*   **Workflow Management:** Load, save, and share workflows using PNG, WebP, FLAC, and JSON files.
*   **API Nodes:** Integrate with external AI services and leverage paid models via the Comfy API.
*   **High-Quality Previews:** Utilize TAESD for detailed image previews.
*   **Full Offline Support:** Core functionality operates completely offline.

## Getting Started

*   **Desktop Application:** The easiest way to get started, available for Windows & macOS: [Download](https://www.comfy.org/download)
*   **Windows Portable Package:** Get the latest commits and enjoy complete portability (Windows only).
*   **Manual Install:** Supports all operating systems and GPU types (NVIDIA, AMD, Intel, Apple Silicon, Ascend).

## Installation

### Windows Portable

Download, extract with [7-Zip](https://7-zip.org), and run. Place Stable Diffusion checkpoints/models in `ComfyUI\models\checkpoints`.

[Direct download link](https://github.com/comfyanonymous/ComfyUI/releases/latest/download/ComfyUI_windows_portable_nvidia.7z)

### comfy-cli
```bash
pip install comfy-cli
comfy install
```
### Manual Install (Windows, Linux)

1.  **Clone the Repository:** `git clone this repo`
2.  **Model Placement:** Place your SD checkpoints (ckpt/safetensors) in `models/checkpoints` and VAEs in `models/vae`.
3.  **Install Dependencies:** `pip install -r requirements.txt` within the ComfyUI directory.
4.  **Run ComfyUI:** `python main.py`

### Platform-Specific Instructions:

*   **AMD GPUs (Linux):** Install ROCm and PyTorch.
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.4
    ```
*   **Intel GPUs:** Instructions for using native PyTorch xpu and Intel Extension for PyTorch (IPEX).
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
    ```
*   **NVIDIA GPUs:** Install PyTorch with CUDA support:
    ```bash
    pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu129
    ```
*   **Apple Mac silicon:** [Apple Developer Guide](https://developer.apple.com/metal/pytorch/) (make sure to install the latest pytorch nightly)
*   **DirectML (AMD Cards on Windows):** Very badly supported and is not recommended.
    ```bash
    pip install torch-directml
    python main.py --directml
    ```
*   **Ascend NPUs** [installation](https://ascend.github.io/docs/sources/ascend/quick_install.html)
*   **Cambricon MLUs** [Installation](https://www.cambricon.com/docs/sdk_1.15.0/cntoolkit_3.7.2/cntoolkit_install_3.7.2/index.html)
*   **Iluvatar Corex** [Installation](https://support.iluvatar.com/#/DocumentCentre?id=1&nameCenter=2&productId=520117912052801536)

### Troubleshooting

*   If you get the "Torch not compiled with CUDA enabled" error, reinstall PyTorch.
*   For AMD cards not officially supported by ROCm and for AMD ROCm tips see the original README.

## Running

Run ComfyUI with the command: `python main.py`

## Essential Shortcuts

A comprehensive list of keyboard shortcuts can be found within the original [README](https://github.com/comfyanonymous/ComfyUI).

## Configuration and Customization

*   **Config File:** Use `extra_model_paths.yaml.example` (renamed to `extra_model_paths.yaml`) to specify custom model search paths.

## How to show high-quality previews?

To enable higher-quality previews with TAESD, download `taesd_decoder.pth, taesdxl_decoder.pth, taesd3_decoder.pth and taef1_decoder.pth` and place them in the `models/vae_approx` folder. Then restart ComfyUI and launch it with `--preview-method taesd`.

## How to use TLS/SSL?

Generate self-signed certificates and use `--tls-keyfile key.pem --tls-certfile cert.pem` to enable TLS/SSL.

## Support and Community

*   **Discord:** Join the community on Discord for help and feedback: [Discord](https://www.comfy.org/discord)
*   **Matrix:** Connect with other users on Matrix: [Matrix Space](https://app.element.io/#/room/%23comfyui_space%3Amatrix.org)
*   **Website:** Explore the ComfyUI website for tutorials and resources: [https://www.comfy.org/](https://www.comfy.org/)

## Frontend Development

Please use the [ComfyUI Frontend repository](https://github.com/Comfy-Org/ComfyUI_frontend) for frontend related issues or feature requests.

### Using the Latest Frontend

Launch ComfyUI with this command line argument:

```
--front-end-version Comfy-Org/ComfyUI_frontend@latest
```

### Accessing the Legacy Frontend

If needed, you can use the following command line argument:

```
--front-end-version Comfy-Org/ComfyUI_legacy_frontend@latest
```

## Further Resources

*   [Examples](https://comfyanonymous.github.io/ComfyUI_examples/)
*   [Which GPU should I buy for ComfyUI?](https://github.com/comfyanonymous/ComfyUI/wiki/Which-GPU-should-I-buy-for-ComfyUI)

**Contribute to the future of AI art!  Explore the power of ComfyUI!**