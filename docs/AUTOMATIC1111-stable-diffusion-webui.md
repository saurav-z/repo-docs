# Stable Diffusion Web UI: Unleash Your Creative Vision with AI

**Generate stunning images from text prompts with the Stable Diffusion web UI, a powerful and user-friendly interface for Stable Diffusion.** ([Original Repository](https://github.com/AUTOMATIC1111/stable-diffusion-webui))

## Key Features

*   **Text-to-Image & Image-to-Image Generation:** Create images from text descriptions or modify existing images.
*   **One-Click Installation:** Get started quickly with easy-to-use installation scripts (requires Python and Git).
*   **Advanced Editing Tools:** Outpainting, inpainting, color sketch, and prompt matrix to refine your creations.
*   **Attention Mechanism:** Control the model's focus using attention weighting for specific parts of your prompts.
*   **Prompting Flexibility:** No token limit for prompts, and support for features like negative prompts, styles, and variations.
*   **Upscaling & Enhancement:** Utilize GFPGAN, CodeFormer, RealESRGAN, ESRGAN, and other upscalers for improved image quality.
*   **Customization & Control:** Fine-tune your image generation with sampling method selection, seed control, and generation parameters.
*   **Extensive Community Support:** Leverage a wide array of extensions and scripts from the community for added functionality.
*   **Advanced Techniques:** Explore features like Textual Inversion, X/Y/Z plots, and Composable Diffusion for advanced image manipulation.
*   **Multiple Model & Technology Support:** Includes support for Stable Diffusion 2.0, Alt-Diffusion, and Segmind Stable Diffusion.
*   **Batch Processing:** Process multiple images with img2img.
*   **API Support:** Integrate the web UI into other applications and workflows.
*   **Training Tab:** Train Hypernetworks, Embeddings, and Loras

## Installation and Running

Detailed installation instructions are available for:

*   **NVIDIA GPUs** (Recommended)
*   **AMD GPUs**
*   **Intel CPUs/GPUs**
*   **Ascend NPUs**

Also includes instructions to install using online services such as Google Colab.

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

1.  Install Dependencies:
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

Contribute to the project by following the guide: [Contributing](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Contributing)

## Documentation

Comprehensive documentation is available on the project's [wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki).

## Credits

This project leverages contributions from many individuals and open-source projects, including:

*   Stable Diffusion, k-diffusion, Spandrel, LDSR, MiDaS, and more (See detailed credits in the original README)