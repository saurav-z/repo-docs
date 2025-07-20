# Stable Diffusion Web UI: Unleash Your Creativity with AI-Powered Image Generation

This powerful web interface, built with the Gradio library, provides an intuitive platform to generate stunning images from text prompts using the Stable Diffusion model. [Explore the original repository here](https://github.com/AUTOMATIC1111/stable-diffusion-webui).

## Key Features

*   **Core Functionality:**
    *   txt2img and img2img modes for versatile image generation.
    *   Outpainting and inpainting for image editing and expansion.
    *   Color Sketch for creating images from your initial sketches.
    *   Prompt Matrix for exploring variations of prompts.
    *   Stable Diffusion Upscale for enhanced image resolution.
*   **Advanced Prompting & Control:**
    *   Attention mechanism for focusing the model on specific text elements.
    *   Negative prompting to guide the model away from unwanted elements.
    *   Styles for saving and easily applying prompt templates.
    *   Variations for generating similar images with slight differences.
*   **Image Enhancement & Editing:**
    *   GFPGAN, CodeFormer, RealESRGAN, ESRGAN, SwinIR, Swin2SR, and LDSR for upscaling and face restoration.
    *   Resizing aspect ratio options for image control.
    *   Seed resizing for generating images with similar characteristics at different resolutions.
    *   Prompt Editing allows to change prompt mid-generation
    *   Batch Processing for image processing using img2img
*   **Customization & Extensibility:**
    *   Extensive settings page for customization.
    *   Support for custom scripts and community extensions.
    *   Checkpoint Merger for combining checkpoints.
    *   Custom scripts with community extensions.
    *   Loras (same as Hypernetworks but more pretty)
*   **Efficiency & Performance:**
    *   4GB video card support.
    *   Interrupt processing at any time.
    *   Progress bar and live image generation preview.
    *   xformers integration for major speed increases.
*   **Advanced features:**
    *   Tiling support, a checkbox to create images that can be tiled like textures
    *   CLIP interrogator, a button that tries to guess prompt from an image
    *   Highres Fix, a convenience option to produce high resolution pictures in one click without usual distortions
    *   Reloading checkpoints on the fly
    *   DeepDanbooru integration, creates danbooru style tags for anime prompts
    *   Training tab
*   **API & Integration:**
    *   API for programmatic access.
    *   Support for dedicated inpainting models.
    *   [Stable Diffusion 2.0](https://github.com/Stability-AI/stablediffusion) and [Alt-Diffusion](https://arxiv.org/abs/2211.06679) support.
    *   Segmind Stable Diffusion support
*   **User Experience:**
    *   Generation parameters saved with images (in PNG chunks or EXIF).
    *   Read Generation Parameters Button, loads parameters in promptbox to UI
    *   Mouseover hints for most UI elements.
    *   Reorder elements in the UI from settings screen

## Installation

Detailed instructions are available in the [installation section of the original README](https://github.com/AUTOMATIC1111/stable-diffusion-webui#installation-and-running). Key steps include:

*   Installing Python and Git.
*   Cloning the repository.
*   Running the appropriate startup script (`webui-user.bat` on Windows, `webui.sh` on Linux).

### Installation on Windows 10/11 with NVidia-GPUs using release package
1.  Download `sd.webui.zip` from [v1.0.0-pre](https://github.com/AUTOMATIC1111/stable-diffusion-webui/releases/tag/v1.0.0-pre) and extract its contents.
2.  Run `update.bat`.
3.  Run `run.bat`.
    > For more details see [Install-and-Run-on-NVidia-GPUs](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-NVidia-GPUs)

### Automatic Installation on Windows
1.  Install [Python 3.10.6](https://www.python.org/downloads/release/python-3106/) (Newer version of Python does not support torch), checking "Add Python to PATH".
2.  Install [git](https://git-scm.com/download/win).
3.  Download the stable-diffusion-webui repository, for example by running `git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git`.
4.  Run `webui-user.bat` from Windows Explorer as normal, non-administrator, user.

### Automatic Installation on Linux
1.  Install the dependencies:
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
    If your system is very new, you need to install python3.11 or python3.10:
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
2.  Navigate to the directory you would like the webui to be installed and execute the following command:
    ```bash
    wget -q https://raw.githubusercontent.com/AUTOMATIC1111/stable-diffusion-webui/master/webui.sh
    ```
    Or just clone the repo wherever you want:
    ```bash
    git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui
    ```

3.  Run `webui.sh`.
4.  Check `webui-user.sh` for options.

### Installation on Apple Silicon

Find the instructions [here](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Installation-on-Apple-Silicon).

## Contributing

Contribute to the project by following the guidelines in the [Contributing](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Contributing) section.

## Documentation

Explore the comprehensive [wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki) for detailed information and tutorials.