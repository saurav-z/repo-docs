# Stable Diffusion Web UI: Unleash Your Creativity with AI-Powered Image Generation

**Create stunning images from text prompts with the user-friendly Stable Diffusion web UI, a powerful interface built with Gradio.** ([Original Repository](https://github.com/AUTOMATIC1111/stable-diffusion-webui))

This project offers a comprehensive suite of features, providing both novice and expert users with robust tools to explore the potential of Stable Diffusion.

## Key Features

*   **Core Image Generation:**
    *   txt2img and img2img modes for versatile image creation and modification.
    *   Outpainting and Inpainting for seamless image extension and selective editing.
    *   Color Sketch: Generate images from color sketches, with the option to use inpainting as well.

*   **Advanced Control & Customization:**
    *   Prompt Matrix: Experiment with multiple prompts and parameters.
    *   Attention Mechanism: Fine-tune prompts with attention syntax (e.g., `((tuxedo))` or `(tuxedo:1.21)`).
    *   Negative Prompt: Specify elements to exclude from your generated images.
    *   Styles: Save and reuse prompt elements for consistent aesthetics.
    *   Variations: Generate slight variations of an image.
    *   Seed Control: Achieve consistent results with seed resizing.

*   **Enhancements & Tools:**
    *   Upscaling: Stable Diffusion Upscale, GFPGAN, CodeFormer, RealESRGAN, ESRGAN, SwinIR, Swin2SR, and LDSR for high-resolution outputs.
    *   Textual Inversion: Train and use custom embeddings for personalized image styles.
    *   CLIP Interrogator: Analyze images and generate prompts.
    *   Checkpoint Merger: Combine up to 3 checkpoints.
    *   Highres Fix: Produce high-resolution pictures with no distortion.

*   **Community & Integration:**
    *   Custom Scripts: Extend functionality with community-created scripts.
    *   Composable Diffusion: Combine multiple prompts.
    *   DeepDanbooru: Generate anime-style tags.
    *   xformers: Optimized performance.

*   **User-Friendly Interface:**
    *   Progress Bar & Live Preview: Monitor generation progress.
    *   Generation Parameters: Save and restore generation settings.
    *   Settings Page: Configure defaults and UI elements.
    *   API: Integrate with other applications.
    *   UI reorder from settings screen
    *   Estimated completion time in progress bar

*   **Other Features:**
    *   Batch Processing: Process multiple images at once.
    *   Seamless Loading: Load checkpoints in safetensors format
    *   Support for SD 2.0, Alt-Diffusion and Segmind SD.

## Installation and Running

Instructions for various platforms are included in the original repository. For your convenience, here are some quick start guides.

**Installation on Windows 10/11 with NVidia-GPUs using release package**
1. Download `sd.webui.zip` from [v1.0.0-pre](https://github.com/AUTOMATIC1111/stable-diffusion-webui/releases/tag/v1.0.0-pre) and extract its contents.
2. Run `update.bat`.
3. Run `run.bat`.
> For more details see [Install-and-Run-on-NVidia-GPUs](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-NVidia-GPUs)

**Automatic Installation on Windows**

1.  Install [Python 3.10.6](https://www.python.org/downloads/release/python-3106/), checking "Add Python to PATH".
2.  Install [git](https://git-scm.com/download/win).
3.  Clone the repository: `git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git`.
4.  Run `webui-user.bat` from Windows Explorer.

**Automatic Installation on Linux**
1.  Install dependencies. Use the following command, based on your system:
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
2.  Navigate to your desired installation directory and execute:
    ```bash
    wget -q https://raw.githubusercontent.com/AUTOMATIC1111/stable-diffusion-webui/master/webui.sh
    ```
    Or clone the repo.
    ```bash
    git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui
    ```

3.  Run `webui.sh`.
4.  Check `webui-user.sh` for options.

For more platform-specific instructions, including installation on Apple Silicon, please refer to the [original repository's wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki).

## Contributing

Contribute to the project [here](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Contributing).

## Documentation

Comprehensive documentation is available on the project's [wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki).

## Credits

This project is built upon the work of many contributors.  See `Settings -> Licenses` or the `html/licenses.html` file for a complete list.