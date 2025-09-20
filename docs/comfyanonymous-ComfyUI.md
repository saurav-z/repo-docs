<div align="center">

# ComfyUI: Unleash Your Creativity with a Powerful Visual AI Engine

<p>
  <a href="https://www.comfy.org/">
    <img src="https://img.shields.io/badge/ComfyOrg-4285F4?style=flat" alt="Website">
  </a>
  <a href="https://discord.com/api/invites/comfyorg">
    <img src="https://img.shields.io/discord/772175995140395540?color=green&label=Discord&logo=discord&logoColor=white&suffix=%20total" alt="Discord">
  </a>
  <a href="https://x.com/ComfyUI">
    <img src="https://img.shields.io/twitter/follow/ComfyUI?style=flat" alt="Twitter">
  </a>
  <a href="https://app.element.io/#/room/%23comfyui_space%3Amatrix.org">
    <img src="https://img.shields.io/badge/Matrix-000000?style=flat&logo=matrix&logoColor=white" alt="Matrix">
  </a>
  <a href="https://github.com/comfyanonymous/ComfyUI/releases">
    <img src="https://img.shields.io/github/release/comfyanonymous/ComfyUI?sort=semver&style=flat" alt="GitHub Release">
  </a>
  <a href="https://github.com/comfyanonymous/ComfyUI/releases">
    <img src="https://img.shields.io/github/release-date/comfyanonymous/ComfyUI?style=flat" alt="Release Date">
  </a>
  <a href="https://github.com/comfyanonymous/ComfyUI/releases">
    <img src="https://img.shields.io/github/downloads/comfyanonymous/ComfyUI/latest/total?style=flat&label=downloads%40latest" alt="Downloads">
  </a>
</p>

</div>

**ComfyUI empowers you to create intricate AI art and workflows visually, offering unparalleled flexibility and control over Stable Diffusion and beyond.** Available on Windows, Linux, and macOS. [Explore the original repo](https://github.com/comfyanonymous/ComfyUI).

Key Features:

*   **Node-Based Workflow:** Design and execute complex Stable Diffusion pipelines visually using an intuitive graph/nodes/flowchart interface, eliminating the need for coding.
*   **Extensive Model Support:** Compatible with a wide array of models including SD1.x, SD2.x, SDXL, Stable Cascade, SD3, and more, including video, audio and 3D models.
*   **Advanced Image Editing:** Leverage powerful image editing models like Omnigen 2, Flux Kontext, and Qwen Image Edit for refined results.
*   **Video Generation Capabilities:** Create stunning videos using Stable Video Diffusion, Mochi, and other video models.
*   **Optimized Performance:** Benefit from an asynchronous queue system, smart memory management, and partial workflow execution for efficiency.
*   **Flexible Hardware Support:** Works with or without a GPU, offering options for CPU-based processing (`--cpu`) and supports AMD, Intel, NVIDIA and Apple Silicon.
*   **Model Compatibility:** Load ckpt, safetensors, and various other model formats, along with embeddings, LoRAs, and Hypernetworks.
*   **Workflow Management:** Import/export workflows as PNG, WebP, or JSON files, and utilize pre-built examples for quick starts.
*   **Advanced Features:** Includes support for Area Composition, Inpainting, ControlNet, T2I-Adapter, Upscale Models, GLIGEN, Model Merging, LCM models, and more.
*   **High-Quality Previews:** Option to enable high-quality previews with TAESD for detailed results.
*   **Offline Operation:** Operates entirely offline, ensuring privacy and control over your data.
*   **API Integration (Optional):** Connect to external providers using the Comfy API for access to paid models.

Get Started:

*   **Desktop Application:** The easiest way to get started. [Download Here](https://www.comfy.org/download) (Windows & macOS)
*   **Windows Portable Package:** Get the latest commits in a completely portable package. [Download Here](https://github.com/comfyanonymous/ComfyUI/releases/latest/download/ComfyUI_windows_portable_nvidia.7z) (Windows)
*   **Manual Install:** Supports all operating systems and GPU types. (See below for detailed instructions.)

[See Example Workflows](https://comfyanonymous.github.io/ComfyUI_examples/)

## Installation

### [comfy-cli](https://docs.comfy.org/comfy-cli/getting-started)

You can install and start ComfyUI using comfy-cli:
```bash
pip install comfy-cli
comfy install
```

### Manual Install (Windows, Linux)

1.  **Clone the Repository:**
    ```bash
    git clone this repo.
    ```

2.  **Model Placement:** Place your Stable Diffusion checkpoints (large .ckpt/.safetensors files) in `models/checkpoints`. Place VAE files in `models/vae`.

3.  **AMD GPUs (Linux only):** If you are an AMD GPU user, you can install rocm and pytorch with pip.
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.4
    ```
    or
    ```bash
    pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.4
    ```

4.  **Intel GPUs (Windows and Linux)** Option 1: Install Pytorch xpu
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
    ```
    or
    ```bash
    pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/xpu
    ```
    Option 2: Visit [Installation](https://intel.github.io/intel-extension-for-pytorch/index.html#installation?platform=gpu)

5.  **NVIDIA GPUs:** Nvidia users should install stable pytorch using this command:
    ```bash
    pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu129
    ```
    or
    ```bash
    pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu129
    ```

6.  **Troubleshooting NVIDIA CUDA Errors:** If you encounter errors, try:
    ```bash
    pip uninstall torch
    pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu129
    ```

7.  **Install Dependencies:** Navigate to the ComfyUI directory in your terminal and run:
    ```bash
    pip install -r requirements.txt
    ```

8.  **Run ComfyUI:**
    ```bash
    python main.py
    ```

### Platform-Specific Instructions
*   **Apple Mac silicon:** Follow the [Accelerated PyTorch training on Mac](https://developer.apple.com/metal/pytorch/) Apple Developer guide (make sure to install the latest pytorch nightly).
*   **DirectML (AMD Cards on Windows):** `pip install torch-directml`, then launch with `python main.py --directml`. (Less recommended)
*   **Ascend NPUs:** Follow the detailed guide [here](https://ascend.github.io/docs/sources/pytorch/install.html#pytorch).
*   **Cambricon MLUs:** Follow the detailed guide [here](https://www.cambricon.com/docs/sdk_1.15.0/cambricon_pytorch_1.17.0/user_guide_1.9/index.html).
*   **Iluvatar Corex:** Follow the detailed guide [here](https://support.iluvatar.com/#/DocumentCentre?id=1&nameCenter=2&productId=520117912052801536).

## Running

```bash
python main.py
```

### AMD ROCm Tips

Experimental memory efficient attention (RDNA3 default):
```bash
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 python main.py --use-pytorch-cross-attention
```
or
```bash
PYTORCH_TUNABLEOP_ENABLED=1 python main.py
```

## Keybinds
See original README.

## Additional Notes
*   See original README.

## How to show high-quality previews?
Enable high quality previews by:
*   Place taesd_decoder.pth, taesdxl_decoder.pth, taesd3_decoder.pth and taef1_decoder.pth into the models/vae_approx folder.
*   Restart and use `--preview-method taesd`

## How to use TLS/SSL?
*   Generate a self-signed certificate (not appropriate for shared/production use):
```bash
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -sha256 -days 3650 -nodes -subj "/C=XX/ST=StateName/L=CityName/O=CompanyName/OU=CompanySectionName/CN=CommonNameOrHostname"
```
*   Enable by using `--tls-keyfile key.pem --tls-certfile cert.pem`

## Support and dev channel

*   [Discord](https://comfy.org/discord): Try the #help or #feedback channels.
*   [Matrix space: #comfyui_space:matrix.org](https://app.element.io/#/room/%23comfyui_space%3Amatrix.org) (it's like discord but open source).
*   See also: [https://www.comfy.org/](https://www.comfy.org/)

## Frontend Development
See original README.

## QA
See original README.