# Stable Diffusion Web UI: Unleash Your Creativity with AI-Powered Image Generation

**Bring your imagination to life with the Stable Diffusion Web UI, a user-friendly web interface for generating stunning images from text prompts.** [Explore the original repo](https://github.com/AUTOMATIC1111/stable-diffusion-webui) for the latest updates.

## Key Features

*   **Text-to-Image and Image-to-Image Generation:** Generate images from text prompts or modify existing images.
*   **One-Click Installation:** Easy setup with a script (Python and Git required).
*   **Advanced Image Editing:** Outpainting, inpainting, color sketch, and more.
*   **Prompt Refinement:** Use attention mechanisms, negative prompts, and prompt editing for precise control.
*   **Upscaling & Enhancement:** Integrate GFPGAN, CodeFormer, RealESRGAN, ESRGAN, and other upscalers.
*   **Extensive Customization:** Configure settings, use styles, and generate variations.
*   **Batch Processing:** Process multiple images at once.
*   **Seed & Parameter Management:** Preserve generation parameters and reload them easily.
*   **Model and Extension Support:** Load checkpoints, merge models, and utilize community-created scripts and extensions.
*   **AI Tools:** DeepDanbooru integration, CLIP interrogator, and more.
*   **Advanced Techniques:** Includes X/Y/Z plots, Composable-Diffusion, and Loras.
*   **Hardware Support:** Optimized for various GPUs, with 4GB VRAM support and 2GB support reported.

## Installation and Running

Follow the installation instructions below for your operating system. Detailed guides are available in the project's wiki and linked below:

*   **Nvidia GPUs:** [Install-and-Run-on-NVidia-GPUs](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-NVidia-GPUs)
*   **AMD GPUs:** [Install-and-Run-on-AMD-GPUs](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-AMD-GPUs)
*   **Intel CPUs/GPUs:** [Installation-on-Intel-Silicon](https://github.com/openvinotoolkit/stable-diffusion-webui/wiki/Installation-on-Intel-Silicon) (external wiki page)
*   **Ascend NPUs:** [Install-and-run-on-Ascend-NPUs](https://github.com/wangshuai09/stable-diffusion-webui/wiki/Install-and-run-on-Ascend-NPUs) (external wiki page)
*   **Apple Silicon:** [Installation-on-Apple-Silicon](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Installation-on-Apple-Silicon)
*   **Online Services:** [List of Online Services](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Online-Services)

### Automatic Installation on Windows (using release package)

1.  Download `sd.webui.zip` from [v1.0.0-pre](https://github.com/AUTOMATIC1111/stable-diffusion-webui/releases/tag/v1.0.0-pre) and extract its contents.
2.  Run `update.bat`.
3.  Run `run.bat`.

>   For more details see [Install-and-Run-on-NVidia-GPUs](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-NVidia-GPUs)

### Automatic Installation on Windows

1.  Install [Python 3.10.6](https://www.python.org/downloads/release/python-3106/) (Important: Newer versions of Python may not support the required version of PyTorch), checking "Add Python to PATH".
2.  Install [git](https://git-scm.com/download/win).
3.  Download the stable-diffusion-webui repository, for example by running `git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git`.
4.  Run `webui-user.bat` from Windows Explorer as a normal, non-administrator, user.

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

## Contributing

Contribute to the project [here](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Contributing).

## Documentation

Comprehensive documentation is available in the project [wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki).

## Credits

Special thanks to the contributors. See the full list in the original README.