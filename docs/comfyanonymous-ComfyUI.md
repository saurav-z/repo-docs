<div align="center">

# ComfyUI: Unleash Your Creativity with Visual AI (Stable Diffusion)

**Create stunning AI-generated images and videos with ComfyUI, the most powerful and modular visual AI engine.**

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

*   **Node-Based Workflow:** Design complex Stable Diffusion pipelines using a visual, node-based interface, eliminating the need for extensive coding.
*   **Extensive Model Support:** Compatible with a vast array of image, video, and audio models, including SD1.x, SD2.x, SDXL, Stable Cascade, and more.
*   **Image & Video Editing:** Utilize powerful editing models like Omnigen 2, Flux Kontext, and Qwen Image Edit for advanced image manipulation.
*   **Video & Audio Generation:** Create stunning videos with Stable Video Diffusion and explore audio generation with Stable Audio and ACE Step.
*   **3D Model Integration:** Experiment with 3D models with Hunyuan3D 2.0 support.
*   **Asynchronous Queue System:** Manage your generation tasks efficiently with an asynchronous queue.
*   **Optimized Performance:** Benefit from optimizations like partial workflow execution and smart memory management for efficient resource utilization.
*   **Broad Hardware Compatibility:** Runs on various hardware configurations, including GPUs (NVIDIA, AMD, Intel) and CPUs.
*   **Flexible Model Loading:** Load ckpt, safetensors, and other model file types.
*   **Workflow Flexibility:** Load and save workflows as JSON files and import them from PNG, WebP, and FLAC files.
*   **Advanced Features:** Includes support for embeddings, LoRAs, hypernetworks, ControlNet, upscaling, and model merging.

### Getting Started

Choose your preferred installation method:

*   **Desktop Application:** The easiest way to begin; available for Windows & macOS via the [download](https://www.comfy.org/download) page.
*   **Windows Portable Package:** A fully portable version for Windows, offering the latest updates. Find it on the [releases page](https://github.com/comfyanonymous/ComfyUI/releases).
*   **Manual Install:** Supports all operating systems and GPU types.

### Links
*   **[Original Repository](https://github.com/comfyanonymous/ComfyUI)**
*   **[Examples](https://comfyanonymous.github.io/ComfyUI_examples/)**
*   **[ComfyUI Wiki](https://github.com/comfyanonymous/ComfyUI/wiki)**

**(Detailed installation and running instructions below)**

## Installation

### Windows Portable

1.  Download the latest portable build for Windows from the [releases page](https://github.com/comfyanonymous/ComfyUI/releases).
2.  Extract the archive using [7-Zip](https://7-zip.org).
3.  Place your Stable Diffusion checkpoints and models in `ComfyUI\models\checkpoints`.
4.  Run the executable.

*If you have trouble extracting the file, right-click the file, go to "Properties," and click "Unblock."*

### [comfy-cli](https://docs.comfy.org/comfy-cli/getting-started)

```bash
pip install comfy-cli
comfy install
```

### Manual Installation (Windows, Linux)

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/comfyanonymous/ComfyUI.git
    cd ComfyUI
    ```
2.  **Model Placement:** Place your Stable Diffusion checkpoints (ckpt/safetensors) in the `models/checkpoints` directory and your VAEs in the `models/vae` directory.
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **GPU-Specific Instructions:**

    *   **AMD GPUs (Linux):** Install rocm and pytorch using:
        ```bash
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.4
        ```
        or the nightly builds for performance improvements:
        ```bash
        pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.4
        ```
    *   **Intel GPUs (Windows and Linux):**
        *   **Option 1 (Intel Arc GPUs):** Install PyTorch with xpu support:
            ```bash
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
            ```
            or the nightly builds for performance improvements:
            ```bash
            pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/xpu
            ```
        *   **Option 2 (Intel Extension for PyTorch - IPEX):** Visit the [Installation](https://intel.github.io/intel-extension-for-pytorch/index.html#installation?platform=gpu) page for instructions.
    *   **NVIDIA:** Install stable PyTorch:
        ```bash
        pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu129
        ```
        or the nightly builds for performance improvements:
        ```bash
        pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu129
        ```

5.  **Run ComfyUI:**
    ```bash
    python main.py
    ```

    *If you get the "Torch not compiled with CUDA enabled" error, uninstall and reinstall PyTorch as described above.*

### Others
#### Apple Mac silicon
Follow the [ComfyUI manual installation](#manual-install-windows-linux) instructions.

#### DirectML (AMD Cards on Windows)
```bash
pip install torch-directml
python main.py --directml
```
*This is very badly supported and not recommended.*

#### Ascend NPUs
Follow the [Ascend installation](https://ascend.github.io/docs/sources/ascend/quick_install.html) and the [ComfyUI manual installation](#manual-install-windows-linux).

#### Cambricon MLUs
Follow the [Cambricon installation](https://www.cambricon.com/docs/sdk_1.15.0/cntoolkit_3.7.2/cntoolkit_install_3.7.2/index.html) and [PyTorch(torch_mlu) installation](https://www.cambricon.com/docs/sdk_1.15.0/cambricon_pytorch_1.17.0/user_guide_1.9/index.html), then launch ComfyUI.

#### Iluvatar Corex
Follow the [Iluvatar installation](https://support.iluvatar.com/#/DocumentCentre?id=1&nameCenter=2&productId=520117912052801536) and launch ComfyUI.

## Running ComfyUI

1.  Navigate to your ComfyUI directory in the terminal.
2.  Run the following command:

    ```bash
    python main.py
    ```

    **AMD cards (Troubleshooting)**

    ```bash
    # For older AMD cards
    HSA_OVERRIDE_GFX_VERSION=10.3.0 python main.py
    # For newer AMD cards
    HSA_OVERRIDE_GFX_VERSION=11.0.0 python main.py
    ```
3.  **AMD ROCm Tips**
    ```bash
    # Enable experimental memory efficient attention on AMD GPUs
    TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 python main.py --use-pytorch-cross-attention
    # Improve performance by using the below if the above doesn't work.
    PYTORCH_TUNABLEOP_ENABLED=1 python main.py
    ```

## Important Notes

*   Only parts of the graph with complete inputs and outputs will be executed.
*   ComfyUI re-executes only the parts of the workflow that have changed, optimizing execution time.
*   Drag and drop generated PNG images to load the full workflow and seeds.
*   Use `()` for emphasis and `// comment` or `/* comment */` for comments.
*   Use `{wild|card|test}` for wildcard/dynamic prompts.
*   Place embeddings/textual inversions in the `models/embeddings` directory.

## Preview Enhancements

To improve the quality of the previews, use `--preview-method auto` and download the `taesd_decoder.pth, taesdxl_decoder.pth, taesd3_decoder.pth and taef1_decoder.pth` files from [here](https://github.com/madebyollin/taesd/) and put them into the `models/vae_approx` folder. Then, restart ComfyUI and launch it with `--preview-method taesd` to enable high-quality previews.

## Using TLS/SSL

Generate a self-signed certificate and key:
`openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -sha256 -days 3650 -nodes -subj "/C=XX/ST=StateName/L=CityName/O=CompanyName/OU=CompanySectionName/CN=CommonNameOrHostname"`

Enable TLS/SSL using:
`--tls-keyfile key.pem --tls-certfile cert.pem`

## Support and Development

*   **Discord:** Join the #help or #feedback channels on the [Discord server](https://www.comfy.org/discord).
*   **Matrix:** Find us on the [Matrix space](https://app.element.io/#/room/%23comfyui_space%3Amatrix.org).
*   **Website:**  Visit [https://www.comfy.org/](https://www.comfy.org/)

## Frontend Development

The frontend is hosted in a separate repository: [ComfyUI Frontend](https://github.com/Comfy-Org/ComfyUI_frontend).

### Reporting Issues and Feature Requests

Report frontend-specific issues and feature requests in the [ComfyUI Frontend repository](https://github.com/Comfy-Org/ComfyUI_frontend).

### Using the Latest Frontend

*   Launch ComfyUI with the latest daily release:

    ```bash
    --front-end-version Comfy-Org/ComfyUI_frontend@latest
    ```
*   For a specific version:

    ```bash
    --front-end-version Comfy-Org/ComfyUI_frontend@1.2.2
    ```

### Accessing the Legacy Frontend

To access the legacy frontend:

```bash
--front-end-version Comfy-Org/ComfyUI_legacy_frontend@latest
```

## Q&A

### Which GPU should I buy for this?

See the [GPU Recommendations](https://github.com/comfyanonymous/ComfyUI/wiki/Which-GPU-should-I-buy-for-ComfyUI)