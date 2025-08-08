# Stable Diffusion WebUI: Unleash Your Creativity with AI-Powered Image Generation

**Generate stunning images from text prompts with Stable Diffusion WebUI, a powerful and versatile web interface.** ([Original Repo](https://github.com/AUTOMATIC1111/stable-diffusion-webui))

## Key Features:

*   **Versatile Image Generation:**
    *   Text-to-Image (txt2img) and Image-to-Image (img2img) modes.
    *   Outpainting, Inpainting, and Color Sketch features for image manipulation.
    *   Prompt Matrix, Loopback, and X/Y/Z plot for advanced experimentation.
*   **Enhanced Prompting & Control:**
    *   Attention mechanism for fine-tuning prompt emphasis.
    *   Negative prompting to exclude unwanted elements.
    *   Styles to save and easily apply prompt components.
    *   Prompt editing for mid-generation adjustments.
    *   Composable Diffusion for multi-prompt generation.
    *   No token limit for prompts.
*   **Advanced Technologies & Integrations:**
    *   Textual Inversion for custom embeddings.
    *   Extras tab with face restoration (GFPGAN, CodeFormer) and upscaling tools (RealESRGAN, ESRGAN, SwinIR, Swin2SR, LDSR).
    *   CLIP interrogator to generate prompts from images.
    *   DeepDanbooru integration for anime-style prompting.
    *   xformers for performance enhancements.
    *   Support for custom scripts and extensions.
    *   LoRAs (LyCORIS) and Hypernetworks.
    *   Aesthetic Gradients.
    *   Support for Stable Diffusion 2.0, Alt-Diffusion, and Segmind Stable Diffusion.
*   **User-Friendly Experience:**
    *   One-click install and run script (Python and Git required).
    *   Live image generation preview and progress bar.
    *   Generation parameters saved with images.
    *   Settings page and UI customization options.
    *   API for programmatic access.
*   **Additional Features**
    *   Resizing aspect ratio options
    *   Sampling method selection
    *   Interrupt processing at any time
    *   Correct seeds for batches
    *   Live prompt token length validation

## Installation and Running

Detailed installation instructions, including options for NVidia, AMD, and Intel GPUs, as well as online service alternatives (like Google Colab) are available.

**Installation for Windows 10/11 with NVIDIA GPUs (using release package):**

1.  Download the `sd.webui.zip` from the releases.
2.  Extract the contents.
3.  Run `update.bat`.
4.  Run `run.bat`.

**Automatic Installation for Windows:**

1.  Install [Python 3.10.6](https://www.python.org/downloads/release/python-3106/) and [git](https://git-scm.com/download/win).
2.  Download the repository using `git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git`.
3.  Run `webui-user.bat`.

**Automatic Installation for Linux:**

1.  Install Dependencies (see README for specific commands based on your Linux distribution).
2.  Run `webui.sh` or clone the repo and then run the webui
3.  Check `webui-user.sh` for options.

**Installation on Apple Silicon:** Instructions are located [here](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Installation-on-Apple-Silicon).

## Contributing

Learn how to contribute to the project: [Contributing](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Contributing)

## Documentation

Comprehensive documentation is available on the project's [wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki).

## Credits

See the original README for detailed credit information, including the names of the creators of the tools used in the project.