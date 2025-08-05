# Stable Diffusion Web UI: Unleash Your Creativity with AI Art

Create stunning AI-generated images with the **Stable Diffusion web UI**, a powerful and user-friendly interface built with Gradio. ([Original Repository](https://github.com/AUTOMATIC1111/stable-diffusion-webui))

## Key Features:

*   **Versatile Generation Modes:**  Supports both `txt2img` (text-to-image) and `img2img` (image-to-image) generation.
*   **One-Click Installation:**  Easy setup with a simple script (requires Python and Git).
*   **Advanced Image Editing:**  Includes outpainting, inpainting, color sketch, and prompt matrix features.
*   **Image Enhancement Tools:**  Integrates GFPGAN, CodeFormer, RealESRGAN, ESRGAN, SwinIR, Swin2SR, and LDSR for face restoration and upscaling.
*   **Fine-Grained Control:**  Offers attention mechanisms, negative prompts, styles, variations, and seed resizing for precise image manipulation.
*   **Interactive Workflow:**  Features a live image generation preview, interrupt processing, and a progress bar.
*   **Prompt Optimization:**  Supports prompt editing, CLIP interrogator, and DeepDanbooru integration for advanced prompt creation.
*   **Batch Processing and Scripting:** Includes batch processing, custom scripts, and support for Composable Diffusion.
*   **Model Management:** Reloading checkpoints on the fly, Checkpoint Merger, Loras, and a dedicated UI for selecting embeddings, hypernetworks or Loras to add to your prompt.
*   **Advanced Features:**  Supports xformers for performance boosts, aesthetic gradients, Stable Diffusion 2.0, and Alt-Diffusion.
*   **Comprehensive API:** Offers an API for integration with other applications.
*   **Dedicated Inpainting Model Support:** Includes dedicated support for the inpainting model by RunwayML

## Installation and Running

Detailed installation instructions are available in the project's [wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki).  Choose the installation method that best suits your operating system and hardware:

*   **NVidia GPUs (Recommended):**  [Installation Guide](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-NVidia-GPUs)
*   **AMD GPUs:**  [Installation Guide](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-AMD-GPUs)
*   **Intel CPUs/GPUs:**  [Installation Guide](https://github.com/openvinotoolkit/stable-diffusion-webui/wiki/Installation-on-Intel-Silicon)
*   **Ascend NPUs:** [Installation Guide](https://github.com/wangshuai09/stable-diffusion-webui/wiki/Install-and-run-on-Ascend-NPUs)
*   **Apple Silicon:** [Installation Guide](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Installation-on-Apple-Silicon)
*   **Online Services (Google Colab, etc.):**  [List of Online Services](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Online-Services)

### Quick Start for Windows with NVidia GPUs (Release Package)

1.  Download `sd.webui.zip` from [v1.0.0-pre](https://github.com/AUTOMATIC1111/stable-diffusion-webui/releases/tag/v1.0.0-pre) and extract.
2.  Run `update.bat`.
3.  Run `run.bat`.

### Automatic Installation for Windows

1.  Install [Python 3.10.6](https://www.python.org/downloads/release/python-3106/) and Git.
2.  Clone the repository: `git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git`
3.  Run `webui-user.bat`.

### Automatic Installation for Linux

1.  Install dependencies (examples below, adjust based on your distribution):
    ```bash
    # Debian-based:
    sudo apt install wget git python3 python3-venv libgl1 libglib2.0-0
    # Red Hat-based:
    sudo dnf install wget git python3 gperftools-libs libglvnd-glx
    # openSUSE-based:
    sudo zypper install wget git python3 libtcmalloc4 libglvnd
    # Arch-based:
    sudo pacman -S wget git python3
    ```
    If your system is very new, you may need Python 3.11 or Python 3.10
    ```bash
    # Ubuntu 24.04
    sudo add-apt-repository ppa:deadsnakes/ppa
    sudo apt update
    sudo apt install python3.11

    # Manjaro/Arch
    sudo pacman -S yay
    yay -S python311 # do not confuse with python3.11 package

    # Only for 3.11
    # Then set up env variable in launch script
    export python_cmd="python3.11"
    # or in webui-user.sh
    python_cmd="python3.11"
    ```
2.  Run this in the directory where you want to install the webui:
    ```bash
    wget -q https://raw.githubusercontent.com/AUTOMATIC1111/stable-diffusion-webui/master/webui.sh
    ```
    Or git clone the repo: `git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui`
3.  Run `./webui.sh`.

## Contributing

Contribute to the project and help improve the web UI!  See the [Contributing](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Contributing) guidelines.

## Documentation

Comprehensive documentation and guides are available on the project's [wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki).  For search engine indexing, a crawlable version of the wiki is available [here](https://github-wiki-see.page/m/AUTOMATIC1111/stable-diffusion-webui/wiki).

## Credits

A full list of credits and licenses for borrowed code is available in the `Settings -> Licenses` screen and the `html/licenses.html` file.