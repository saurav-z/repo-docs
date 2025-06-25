# Stable Diffusion Web UI: Unleash Your Creativity with AI-Powered Image Generation

**Create stunning images from text prompts with the user-friendly Stable Diffusion web UI, a powerful open-source tool.** ([Original Repository](https://github.com/AUTOMATIC1111/stable-diffusion-webui))

## Key Features

*   **Versatile Image Generation:**
    *   txt2img and img2img modes for generating images from text or modifying existing ones.
    *   Outpainting and inpainting for expanding and editing images seamlessly.
    *   Color Sketch and Prompt Matrix for creative exploration.
*   **Advanced Prompting & Control:**
    *   Attention mechanism to emphasize specific parts of your prompt.
    *   Negative prompts to exclude unwanted elements from your creations.
    *   Styles for saving and easily applying prompt snippets.
    *   Prompt editing and generation variations for iterative refinement.
*   **Image Enhancement & Upscaling:**
    *   Highres Fix for generating high-resolution images.
    *   GFPGAN, CodeFormer, and RealESRGAN for face restoration and image upscaling.
    *   Support for various upscaling models (ESRGAN, SwinIR, LDSR).
*   **Customization & Extensions:**
    *   Textual Inversion and LoRAs for fine-tuning models with custom concepts.
    *   Checkpoint Merger for combining multiple models.
    *   Extensive community-developed extensions for enhanced functionality.
    *   Support for Stable Diffusion 2.0 and Alt-Diffusion.
*   **User-Friendly Interface:**
    *   One-click install and run scripts for easy setup (requires Python and Git).
    *   Live image generation preview and progress bar.
    *   Generation parameters saved with images for easy reproducibility.
    *   Integrated CLIP interrogator for generating prompts from images.
*   **Performance & Optimization:**
    *   4GB video card support (reports of 2GB working).
    *   X/Y/Z plot for exploring different parameters.
    *   xformers for potential performance gains on select cards.
*   **Additional Features:**
    *   Tiling support for creating seamless textures.
    *   Batch processing for automated image generation.
    *   API for programmatic access.
    *   Dedicated inpainting model support.
    *   DeepDanbooru integration for anime prompts.

## Installation

Detailed installation instructions are available in the repository's [wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki), including guides for:

*   NVidia GPUs
*   AMD GPUs
*   Intel CPUs and GPUs
*   Ascend NPUs
*   Apple Silicon
*   Online Services (Google Colab, etc.)

For Windows with NVidia-GPUs, a quick start is:
1.  Download `sd.webui.zip` from [v1.0.0-pre](https://github.com/AUTOMATIC1111/stable-diffusion-webui/releases/tag/v1.0.0-pre)
2.  Run `update.bat`.
3.  Run `run.bat`.

For automatic installation on Windows, refer to the original README.

## Contributing

Contribute to the project by following the [contributing guidelines](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Contributing).

## Documentation

Comprehensive documentation is available on the project's [wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki).

## Credits

The project leverages several open-source libraries and contributions. Detailed credits and licenses are listed in the `Settings -> Licenses` screen and `html/licenses.html` file within the application.