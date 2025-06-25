# Stable Diffusion Web UI: Unleash Your Creativity with AI-Powered Image Generation

**The Stable Diffusion web UI provides an intuitive interface for generating stunning images from text prompts, offering a wealth of features and customization options.** For the latest information and updates, see the original repository [here](https://github.com/AUTOMATIC1111/stable-diffusion-webui).

## Key Features

*   **Text-to-Image and Image-to-Image Generation:** Create images from text descriptions or modify existing images.
*   **One-Click Installation:** Simplifies the setup process (requires Python and Git).
*   **Advanced Editing Tools:** Includes Outpainting, Inpainting, Color Sketch, and Prompt Matrix for fine-grained control.
*   **Attention and Prompt Control:** Utilize features like Attention for focusing the model on specific parts of your prompts and negative prompts to exclude unwanted elements.
*   **Upscaling and Enhancement:** Leverage GFPGAN, CodeFormer, RealESRGAN, and other tools for face restoration and image upscaling.
*   **Extensive Customization:** Explore options for sampling methods, aspect ratios, and generation parameters.
*   **Batch Processing:** Process multiple images with img2img to get variations on a single source image.
*   **Integrated Extensions:** Supports a wide array of community-created extensions, including features like: History tab, Aesthetic Gradients, and more.
*   **Training Tools**: Fine-tune generation with hypernetworks, embeddings, and Loras, along with preprocessing and tagging.
*   **Checkpoint Merger:** Combine up to 3 checkpoints into one.
*   **API Support:** Integrate the web UI into other applications and services.
*   **DeepDanbooru Integration:** Automatic creation of danbooru tags for anime prompts.

## Installation and Running

Detailed installation instructions and troubleshooting tips are available in the project's [wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki).

### Quick Start (Windows with NVidia GPUs)

1.  Download the `sd.webui.zip` from [v1.0.0-pre](https://github.com/AUTOMATIC1111/stable-diffusion-webui/releases/tag/v1.0.0-pre) and extract it.
2.  Run `update.bat`.
3.  Run `run.bat`.

### Automatic Installation (Windows)

1.  Install Python 3.10.6 and Git, ensuring Python is added to your PATH.
2.  Clone the repository using `git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git`.
3.  Run `webui-user.bat`.

### Automatic Installation (Linux)

1.  Install Dependencies (using `apt` or `dnf`, depending on your distribution) and install python 3.10 or python 3.11.
2.  Run `wget -q https://raw.githubusercontent.com/AUTOMATIC1111/stable-diffusion-webui/master/webui.sh` to download or clone the repo.
3.  Run `webui.sh`.

### Installation on Apple Silicon

Find the instructions [here](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Installation-on-Apple-Silicon).

## Contributing

Learn how to contribute to the project at the [Contributing](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Contributing) page.

## Documentation

The comprehensive documentation is available on the project's [wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki).

## Credits

This project incorporates code from various sources, with licenses available in `Settings -> Licenses` and `html/licenses.html`. See the original README for an exhaustive list of credits.
