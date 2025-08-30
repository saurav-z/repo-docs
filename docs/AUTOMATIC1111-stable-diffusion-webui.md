# Stable Diffusion Web UI: Unleash Your Creativity with AI-Powered Image Generation

[Link to Original Repo](https://github.com/AUTOMATIC1111/stable-diffusion-webui)

**Create stunning images effortlessly with Stable Diffusion Web UI, a user-friendly web interface built on the powerful Stable Diffusion model.** This comprehensive tool empowers you to bring your imaginative concepts to life with a wide array of features and customization options.

## Key Features:

*   **Core Image Generation:**
    *   Original `txt2img` and `img2img` modes for text-to-image and image-to-image generation.
    *   Outpainting and Inpainting capabilities for expanding and modifying images.
    *   Color Sketch feature to generate images from sketches.

*   **Advanced Control and Customization:**
    *   Prompt Matrix for generating images with different parameter combinations.
    *   Attention mechanisms for specifying text emphasis (e.g., `((tuxedo))`).
    *   Negative prompts to exclude unwanted elements from your images.
    *   Styles to save and easily apply prompt elements.
    *   Variations and Seed resizing for image iteration and minor adjustments.

*   **Image Enhancement and Upscaling:**
    *   GFPGAN, CodeFormer, and RealESRGAN for face restoration.
    *   Upscaling with ESRGAN, SwinIR, Swin2SR, and LDSR.

*   **User-Friendly Interface:**
    *   One-click install and run script for easy setup (Python and Git required).
    *   Progress bar with live image generation preview.
    *   Generation parameters saved with images for reproducibility.
    *   Read Generation Parameters button for loading parameters into the UI.
    *   Settings page for customization.
    *   Mouseover hints for UI elements.
    *   Tiling support for creating seamless textures.

*   **Extensive Features:**
    *   Prompt editing during generation.
    *   Batch Processing for processing groups of images.
    *   Highres Fix for generating high-resolution images without distortion.
    *   Checkpoint Merger for combining up to 3 checkpoints.
    *   Custom scripts with many extensions from the community.
    *   Composable-Diffusion for using multiple prompts (AND).
    *   No token limit for prompts.
    *   DeepDanbooru integration for anime prompts.
    *   Xformers integration for select cards (with `--xformers` commandline arg).
    *   Training tab for hypernetworks and embeddings.
    *   Loras (same as Hypernetworks but more pretty)
    *   API

*   **Model Support:**
    *   Support for dedicated inpainting models.
    *   Stable Diffusion 2.0 support.
    *   Alt-Diffusion support.
    *   Segmind Stable Diffusion support.
    *   VAE selection from settings screen

## Installation

Detailed installation instructions, including support for NVIDIA, AMD, Intel, and Apple Silicon, as well as online services are found in the project's [wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki).

### Quick Start for Windows with NVidia-GPUs (using release package)

1.  Download `sd.webui.zip` from [v1.0.0-pre](https://github.com/AUTOMATIC1111/stable-diffusion-webui/releases/tag/v1.0.0-pre) and extract its contents.
2.  Run `update.bat`.
3.  Run `run.bat`.

### Automatic Installation (Windows)

1.  Install [Python 3.10.6](https://www.python.org/downloads/release/python-3106/) (Newer version of Python does not support torch), checking "Add Python to PATH".
2.  Install [git](https://git-scm.com/download/win).
3.  Download the stable-diffusion-webui repository, for example by running `git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git`.
4.  Run `webui-user.bat` from Windows Explorer as normal, non-administrator, user.

### Automatic Installation (Linux)

1.  Install dependencies (Debian-based example below):

    ```bash
    sudo apt install wget git python3 python3-venv libgl1 libglib2.0-0
    ```
2.  Navigate to the desired installation directory and execute:

    ```bash
    wget -q https://raw.githubusercontent.com/AUTOMATIC1111/stable-diffusion-webui/master/webui.sh
    ```
    Or Clone the repo:
    ```bash
    git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui
    ```
3.  Run `webui.sh`.

## Contributing

Learn how to contribute to the project in the [Contributing](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Contributing) section of the wiki.

## Documentation

Comprehensive documentation is available on the project's [wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki). For search engine indexing, see the [crawlable wiki](https://github-wiki-see.page/m/AUTOMATIC1111/stable-diffusion-webui/wiki).

## Credits

The project relies on numerous open-source libraries and resources.  Credits and licenses are available in the `Settings -> Licenses` screen and `html/licenses.html` file.