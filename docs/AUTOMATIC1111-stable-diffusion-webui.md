# Stable Diffusion Web UI: Unleash Your Creativity with AI Art

Create stunning AI-generated images with the Stable Diffusion web UI, a powerful and user-friendly interface built using Gradio. ([Original Repository](https://github.com/AUTOMATIC1111/stable-diffusion-webui))

## Key Features

*   **Versatile Image Generation:**
    *   Original `txt2img` (text-to-image) and `img2img` (image-to-image) modes.
    *   Outpainting and inpainting for extending and modifying images.
    *   Color Sketch for starting with a color sketch.
*   **Advanced Image Editing and Control:**
    *   Prompt Matrix for exploring variations with different parameters.
    *   Attention mechanism for focusing on specific prompt elements.
    *   Loopback for iterative image processing.
    *   X/Y/Z plot for visualizing parameter variations.
    *   Textual Inversion for creating custom embeddings.
    *   Negative prompt to specify unwanted elements.
    *   Prompt editing mid-generation.
*   **Enhancement & Upscaling Tools:**
    *   GFPGAN, CodeFormer, RealESRGAN, ESRGAN, SwinIR/Swin2SR, and LDSR for face restoration and image upscaling.
    *   Highres Fix for generating high-resolution images.
*   **Customization & Efficiency:**
    *   Tiling support for creating tileable textures.
    *   Styles for saving and applying prompt snippets.
    *   Variations to generate images with slight differences.
    *   Seed resizing for generating similar images at different resolutions.
    *   CLIP interrogator to guess prompt from an image.
    *   Batch Processing for processing a group of files.
    *   Checkpoint Merger to merge checkpoints.
    *   Custom scripts support.
    *   xformers for speed increases on select cards.
*   **User-Friendly Interface:**
    *   Generation parameters are saved with the image (PNG chunks/EXIF).
    *   Read Generation Parameters Button to load prompt from images.
    *   Settings page for customization.
    *   Mouseover hints for UI elements.
    *   Progress bar and live image generation preview.
*   **Community & Integration:**
    *   DeepDanbooru integration.
    *   Composable Diffusion support.
    *   Loras (Low-Rank Adaptation) support.
    *   History tab via extension.
    *   API support.
    *   Aesthetic Gradients via extension.
    *   Stable Diffusion 2.0 and Alt-Diffusion support.
*   **Training Features:**
    *   Training tab with hypernetworks and embeddings options.
    *   Clip skip and Hypernetworks.

## Installation

Detailed installation instructions are available on the project's [wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki).

**Quick Windows Installation (NVidia GPU):**

1.  Download `sd.webui.zip` from [v1.0.0-pre](https://github.com/AUTOMATIC1111/stable-diffusion-webui/releases/tag/v1.0.0-pre) and extract its contents.
2.  Run `update.bat`.
3.  Run `run.bat`.

**Automatic Installation (Windows):**
1.  Install [Python 3.10.6](https://www.python.org/downloads/release/python-3106/), checking "Add Python to PATH".
2.  Install [git](https://git-scm.com/download/win).
3.  Download the stable-diffusion-webui repository, for example by running `git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git`.
4.  Run `webui-user.bat` from Windows Explorer as normal, non-administrator, user.

**Automatic Installation (Linux):**

1.  Install Dependencies (Debian-based: `sudo apt install wget git python3 python3-venv libgl1 libglib2.0-0`).
2.  Navigate to the desired directory.
3.  Run `wget -q https://raw.githubusercontent.com/AUTOMATIC1111/stable-diffusion-webui/master/webui.sh` or clone the repo.
4.  Run `./webui.sh`.
5.  Check `webui-user.sh` for options.

**Apple Silicon:**  Instructions available [here](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Installation-on-Apple-Silicon).

## Contributing

Learn how to contribute to this project in the [Contributing](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Contributing) section of the wiki.

## Credits

A comprehensive list of credits and licenses can be found in the `Settings -> Licenses` screen within the web UI and in the `html/licenses.html` file.