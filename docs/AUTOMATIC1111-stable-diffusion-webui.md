# Stable Diffusion Web UI: Unleash Your Creativity with AI-Powered Image Generation

**Create stunning images from text prompts with the Stable Diffusion web UI, a user-friendly interface built with Gradio, bringing the power of AI art to your fingertips. ([Original Repo](https://github.com/AUTOMATIC1111/stable-diffusion-webui))**

## Key Features:

*   **Versatile Image Generation:**
    *   txt2img and img2img modes for generating images from text or modifying existing ones.
    *   Outpainting, Inpainting, and Color Sketch capabilities.
    *   Prompt Matrix for exploring variations and Prompt Editing for dynamic generation.
    *   Seed resizing to generate similar images at different resolutions.
    *   Loopback processing for iterative img2img refinement.
    *   Tiling support to create seamless textures.
*   **Advanced Control & Customization:**
    *   Attention mechanism for highlighting specific parts of prompts.
    *   Negative prompts to define what to exclude from the generated image.
    *   Styles feature to save and apply prompt elements easily.
    *   Variations feature to generate similar images with subtle differences.
    *   X/Y/Z plot for plotting images with different parameters.
    *   CLIP interrogator to derive prompts from images.
    *   Checkpoint Merger to combine multiple checkpoints.
*   **Extensive Upscaling & Enhancement:**
    *   Highres Fix for one-click upscaling without distortions.
    *   GFPGAN, CodeFormer, RealESRGAN, and ESRGAN for face and image restoration.
    *   LDSR, SwinIR, and Swin2SR for upscaling with various models.
*   **Flexible Workflow & Extensions:**
    *   Generation parameters saved with images (PNG chunks/EXIF data).
    *   Read Generation Parameters button to load image settings into the UI.
    *   Settings page for customization and UI element reordering.
    *   Support for Custom Scripts and Composable-Diffusion.
    *   Community extensions, including a History tab and Aesthetic Gradients.
    *   DeepDanbooru integration for anime prompts.
*   **Optimization & Performance:**
    *   xformers for major speed increases on select GPUs (add `--xformers` to commandline arguments).
    *   4GB video card support (with reports of 2GB working).
    *   Estimated completion time in progress bar.
    *   Progress bar and live image generation preview.
*   **Training & Integration:**
    *   Training tab for hypernetworks, embeddings, and LoRAs.
    *   Loras (same as Hypernetworks but more pretty)
    *   A separate UI where you can choose, with preview, which embeddings, hypernetworks or Loras to add to your prompt
    *   API support for integration with other applications.
    *   Support for dedicated inpainting models and Stable Diffusion 2.0.
    *   Alt-Diffusion support.
    *   Segmind Stable Diffusion support.
    *   VAE selection available from settings screen

## Installation

Detailed installation instructions are available in the [wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki) for various platforms, including:

*   **NVidia GPUs** (Recommended)
*   **AMD GPUs**
*   **Intel CPUs/GPUs**
*   **Apple Silicon**
*   **Ascend NPUs**
*   **Online Services:** (Google Colab and others).

### Quick Start for Windows (NVidia GPUs):

1.  Download the `sd.webui.zip` release package from [v1.0.0-pre](https://github.com/AUTOMATIC1111/stable-diffusion-webui/releases/tag/v1.0.0-pre) and extract.
2.  Run `update.bat`.
3.  Run `run.bat`.
> For more detailed Windows and Linux installation, please follow the instructions under "Installation and Running" in the original README.

## Contributing

Learn how to contribute to this project by following the guidelines in the [Contributing](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Contributing) section of the wiki.

## Documentation

Comprehensive documentation is available on the project's [wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki). For search engine indexing, a crawlable version of the wiki can be found [here](https://github-wiki-see.page/m/AUTOMATIC1111/stable-diffusion-webui/wiki).

## Credits

The project leverages the contributions of numerous individuals and open-source projects. A comprehensive list of credits and licenses for borrowed code can be found in the `Settings -> Licenses` screen within the web UI and in the `html/licenses.html` file.