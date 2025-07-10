# Stable Diffusion Web UI: Unleash Your Creativity with AI-Powered Image Generation

**Create stunning images with ease using the Stable Diffusion web UI, a powerful and user-friendly interface for Stable Diffusion models.** ([Original Repository](https://github.com/AUTOMATIC1111/stable-diffusion-webui))

## Key Features

*   **Core Image Generation:**
    *   txt2img and img2img modes for text-to-image and image-to-image generation.
    *   Outpainting and inpainting capabilities for extending and editing images.
    *   Color Sketch feature for generating images from sketches.
    *   Prompt Matrix to experiment with different prompts and parameters.
    *   Highres Fix for upscaling images.
    *   Negative prompts to specify what you *don't* want in your images.
*   **Advanced Prompting and Control:**
    *   Attention mechanism to highlight specific text in prompts.
    *   Styles to save and apply prompt snippets.
    *   Prompt editing mid-generation.
    *   DeepDanbooru integration for anime prompts.
    *   Composable-Diffusion for combining multiple prompts.
    *   No token limit for prompts.
*   **Upscaling and Enhancement:**
    *   GFPGAN, CodeFormer, and RealESRGAN for face restoration.
    *   ESRGAN, SwinIR, and Swin2SR for general upscaling.
    *   LDSR (Latent Diffusion Super Resolution) for upscaling.
*   **Workflow and Customization:**
    *   Batch processing for processing multiple files.
    *   X/Y/Z plot for generating 3D plots of image variations.
    *   Checkpoint Merger to merge checkpoints.
    *   Custom scripts and extensions for extended functionality.
    *   Seed resizing for generating variations of an image.
    *   Settings page for configuration.
    *   Run arbitrary python code from the UI (with `--allow-code`).
    *   Supports various VAE models.
*   **Training & Extensions:**
    *   Training tab for hypernetworks, embeddings, and Loras.
    *   Embeddings, hypernetworks, and Loras support with preview in the UI.
    *   History tab (via extension) to view and manage generated images.
    *   Aesthetic Gradients (via extension) for aesthetic control.
*   **Other Features:**
    *   Live image generation preview and progress bar.
    *   Generation parameters saved with images.
    *   CLIP interrogator to guess prompts from images.
    *   Automatic reloading of checkpoints.
    *   API Support.
    *   Dedicated inpainting model support.
    *   Tiling support.
    *   Multiple Model Support including Stable Diffusion 2.0 and Alt-Diffusion.
    *   Support for Segmind Stable Diffusion.

## Installation and Running

Detailed installation instructions are available in the project's [wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki), with specific guides for:

*   NVidia GPUs (recommended)
*   AMD GPUs
*   Intel CPUs and GPUs
*   Ascend NPUs
*   Apple Silicon
*   And also for Online Services (e.g. Google Colab).

### Quick Start: Windows (NVidia GPUs)

1.  Download the `sd.webui.zip` release package.
2.  Run `update.bat`.
3.  Run `run.bat`.

### Automatic Installation (Windows):

1.  Install Python 3.10.6, making sure to add it to your PATH.
2.  Install git.
3.  Clone the repository: `git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git`.
4.  Run `webui-user.bat`.

### Automatic Installation (Linux):

1.  Install dependencies (specific commands provided in the original README).
2.  Navigate to your desired installation directory.
3.  Run `wget -q https://raw.githubusercontent.com/AUTOMATIC1111/stable-diffusion-webui/master/webui.sh` or clone the repo with git.
4.  Run `webui.sh`.
5.  Customize with `webui-user.sh`.

## Contributing

Learn how to contribute to this project in the [Contributing](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Contributing) section of the wiki.

## Documentation

Comprehensive documentation is available on the project's [wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki).

## Credits

A full list of credits and licenses for borrowed code can be found in the `Settings -> Licenses` screen and in the `html/licenses.html` file.