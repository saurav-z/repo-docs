# Stable Diffusion Web UI: Unleash Your Creativity with AI-Powered Image Generation

Create stunning images from text prompts with the **Stable Diffusion Web UI**, a user-friendly interface built on the Gradio library.  Visit the original repository for more information:  [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui).

## Key Features

*   **Core Image Generation:**
    *   txt2img and img2img modes for generating images from text or transforming existing images.
    *   Outpainting and Inpainting capabilities.
    *   Color Sketch feature for creative exploration.
    *   Prompt Matrix for exploring variations based on different parameters.
    *   Highres Fix for producing high-resolution pictures.
    *   Tiling support for creating seamless textures.
    *   Negative Prompt to specify unwanted elements in your generated image.
*   **Advanced Control & Customization:**
    *   Attention mechanism to emphasize specific words or phrases within prompts.
    *   X/Y/Z plot to visualize image variations across multiple parameters.
    *   Textual Inversion for training and using custom embeddings.
    *   Styles to save and easily apply prompt elements.
    *   Variations for generating similar images with subtle differences.
    *   Seed resizing to produce variations at different resolutions.
    *   Prompt Editing to change prompts mid-generation.
    *   CLIP interrogator to generate prompts from images.
    *   Checkpoint Merger to combine up to 3 checkpoints.
    *   Hypernetworks, LoRAs and a UI for managing them in your prompts.
    *   Clip skip and VAE selection.
    *   Lora (same as Hypernetworks but more pretty)
*   **Enhancements & Tools:**
    *   Extras tab with GFPGAN, CodeFormer, RealESRGAN, ESRGAN, SwinIR, Swin2SR, and LDSR upscalers.
    *   Seamless integration with DeepDanbooru for anime prompts.
    *   Batch Processing for processing groups of images.
    *   Checkpoint reloading on the fly.
    *   Custom scripts via extensions.
    *   Composable Diffusion (using uppercase `AND` for multiple prompts).
    *   xformers for improved performance on select GPUs.
    *   History tab (via extension) for convenient image management.
    *   Generate forever option.
    *   Training tab for hypernetworks and embeddings.
    *   Aesthetic Gradients (via extension).
    *   Segmind Stable Diffusion support.
    *   Supports Stable Diffusion 2.0 and Alt-Diffusion.
*   **User Experience & Integration:**
    *   Progress bar and live image preview.
    *   Generation parameters saved with images (PNG chunks/EXIF data).
    *   Read Generation Parameters Button to load parameters from an image.
    *   Mouseover hints for UI elements.
    *   API access for programmatic use.
    *   Estimated completion time in the progress bar.
    *   Seamless support for dedicated inpainting model by RunwayML
*   **Optimization**
    *   4GB video card support (reports of 2GB working)
    *   No token limit for prompts.
    *   Reorder elements in the UI from the settings screen.
    *   Eased resolution restriction: generated image's dimensions must be a multiple of 8 rather than 64
    *   Restart sampling.
    *   Hypertile.
*   **Licensing**
    *   Now with a license!

## Installation and Running

### Installation on Windows 10/11 with NVidia-GPUs using release package
1.  Download `sd.webui.zip` from [v1.0.0-pre](https://github.com/AUTOMATIC1111/stable-diffusion-webui/releases/tag/v1.0.0-pre) and extract its contents.
2.  Run `update.bat`.
3.  Run `run.bat`.
>   For more details see [Install-and-Run-on-NVidia-GPUs](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-NVidia-GPUs)

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

See the [Contributing](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Contributing) guide for information on contributing to the project.

## Documentation

Detailed documentation is available on the project's [wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki).  For search engine indexing, see the [crawlable wiki](https://github-wiki-see.page/m/AUTOMATIC1111/stable-diffusion-webui/wiki).

## Credits

A list of the credits for the project and borrowed code can be found in `Settings -> Licenses` and in `html/licenses.html`.