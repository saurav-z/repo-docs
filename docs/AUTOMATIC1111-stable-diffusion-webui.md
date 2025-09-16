# Stable Diffusion Web UI: Your Gateway to AI-Powered Image Generation

**Unleash your creativity and generate stunning images with the user-friendly Stable Diffusion web UI, a powerful interface built with Gradio.**  [Visit the original repository](https://github.com/AUTOMATIC1111/stable-diffusion-webui) for the source code and more details.

## Key Features:

*   **Core Functionality:**
    *   txt2img and img2img modes for versatile image generation.
    *   Outpainting and Inpainting for image extension and modification.
    *   Color Sketch for generating images from sketches.
*   **Advanced Features:**
    *   Prompt Matrix for exploring different parameter combinations.
    *   Attention mechanism for fine-tuning prompt emphasis (`((tuxedo))`, `(tuxedo:1.2)`).
    *   Loopback for iterative img2img processing.
    *   X/Y/Z plot for visualizing parameter variations.
*   **AI Enhancement Tools:**
    *   GFPGAN and CodeFormer for advanced face restoration.
    *   RealESRGAN, ESRGAN, SwinIR, Swin2SR, and LDSR for image upscaling.
    *   CLIP Interrogator for generating prompts from images.
    *   DeepDanbooru integration for anime-style prompts.
*   **Workflow Enhancements:**
    *   Seed resizing and variations for nuanced image variations.
    *   Negative prompt to exclude unwanted elements.
    *   Styles for saving and applying prompt components.
    *   Checkpoint merging and reloading.
    *   Batch processing.
*   **Customization and Control:**
    *   Tiling support for seamless textures.
    *   Prompt editing mid-generation.
    *   Custom scripts and community extensions.
    *   Composable-Diffusion for complex prompt combinations.
    *   Hypernetworks, Loras, and Embeddings for advanced customization.
    *   API for integration with other applications.
    *   Model support for Stable Diffusion 2.0, Alt-Diffusion, and Segmind Stable Diffusion.
*   **User Experience:**
    *   Live image generation preview with a progress bar.
    *   Generation parameters saved with images.
    *   Read Generation Parameters button for importing image settings.
    *   UI element customization.
    *   Estimated completion time in progress bar.

## Installation and Running

Find detailed installation instructions for various operating systems and hardware configurations, including NVidia, AMD, Intel CPUs/GPUs, and Apple Silicon, on the project's [wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki).

**Quick Installation (Windows with NVidia GPU):**

1.  Download `sd.webui.zip` from the latest release.
2.  Run `update.bat`.
3.  Run `run.bat`.

**Automatic Installation (Windows):**

1.  Install Python 3.10.6 and Git.
2.  Clone the repository: `git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git`.
3.  Run `webui-user.bat`.

**Automatic Installation (Linux):**

1.  Install dependencies.
2.  Run `webui.sh` or clone the repository and run `webui-user.sh`.

## Contributing
Contribute to the project by following the guide here: [Contributing](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Contributing)

## Documentation
Extensive documentation is available on the project's [wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki).

## Credits

The project utilizes code from various sources. Detailed credits and licenses can be found in the `Settings -> Licenses` screen within the web UI and in the `html/licenses.html` file.