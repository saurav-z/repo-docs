# Stable Diffusion Web UI: Unleash Your Creativity with AI-Powered Image Generation

[Explore the power of Stable Diffusion](https://github.com/AUTOMATIC1111/stable-diffusion-webui), a cutting-edge web interface that lets you generate stunning images from text prompts using the Stable Diffusion AI model.

## Key Features

*   **Text-to-Image & Image-to-Image Generation:** Create images from text descriptions or modify existing images.
*   **One-Click Installation:**  Easy setup with scripts for quick deployment (Python and Git required).
*   **Advanced Image Editing Tools:**
    *   Outpainting and Inpainting for image expansion and editing.
    *   Color Sketch, Prompt Matrix, and Upscaling for creative control.
*   **Prompt Optimization:**
    *   Attention mechanism for fine-tuning prompt emphasis (e.g., `((tuxedo))`).
    *   Negative Prompt: Specify what you *don't* want in the image.
    *   Styles: Save and apply prompt styles easily.
*   **AI-Powered Enhancements:**
    *   GFPGAN and CodeFormer for face restoration.
    *   RealESRGAN and other upscalers for enhanced image resolution.
*   **Flexible Generation Options:**
    *   Batch Processing: Process multiple images at once.
    *   X/Y/Z Plot:  3D image generation with varying parameters.
    *   Seed Control, Variations, and Seed Resizing for iterative generation.
*   **Extensive Customization:**
    *   Custom Scripts with community extensions.
    *   Hypernetworks, LoRAs, and Embeddings for personalized image styles.
    *   Checkpoint Merger for combining model checkpoints.
*   **Advanced Techniques:**
    *   CLIP Interrogator:  Guess prompts from images.
    *   Prompt Editing:  Change prompts during generation.
    *   Composable-Diffusion and Alt-Diffusion support for advanced image composition.
*   **Performance and Compatibility:**
    *   4GB+ video card support (with reports of 2GB working).
    *   xformers for performance enhancements on select cards.
    *   Apple Silicon support.
*   **Additional Features:** Includes a built in API for easy integration, and a UI to select hypernetworks and embeddings to add to your prompt.

## Installation & Running

Detailed installation instructions are available for:

*   Nvidia GPUs
*   AMD GPUs
*   Intel CPUs and GPUs
*   Ascend NPUs
*   Online Services (Google Colab and more)

**Quick Start (Windows with Nvidia):**

1.  Download the latest release from the [Releases](https://github.com/AUTOMATIC1111/stable-diffusion-webui/releases) page.
2.  Run `update.bat`.
3.  Run `run.bat`.

**Automatic Installation (Windows):**

1.  Install Python 3.10.6 and Git.
2.  Clone the repository using `git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git`.
3.  Run `webui-user.bat`.

**Automatic Installation (Linux):**

1.  Install dependencies using `sudo apt install wget git python3 python3-venv libgl1 libglib2.0-0` (Debian-based) or equivalent for your distribution.
2.  Run `wget -q https://raw.githubusercontent.com/AUTOMATIC1111/stable-diffusion-webui/master/webui.sh` or clone the repo.
3.  Run `./webui.sh`.

## Contributing

Contribute to the project via the [Contributing Guidelines](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Contributing).

## Documentation

Comprehensive documentation is available on the project's [wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki).

## Credits

Special thanks to the numerous contributors and projects that have made this web UI possible; details can be found in the original README's "Credits" section.