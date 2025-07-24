# Stable Diffusion Web UI: Unleash Your Creativity with AI-Powered Image Generation

**Create stunning images from text prompts with the Stable Diffusion web UI, a user-friendly interface for the powerful Stable Diffusion model.** [(Original Repository)](https://github.com/AUTOMATIC1111/stable-diffusion-webui)

## Key Features:

*   **Versatile Image Generation Modes:**
    *   `txt2img`: Generate images from text prompts.
    *   `img2img`: Edit existing images with text prompts.
    *   Outpainting and Inpainting: Extend or modify images.
    *   Color Sketch: Generate images from color sketches.
*   **Enhanced Prompting Capabilities:**
    *   Attention mechanism with syntax highlighting: control the emphasis on specific words with `((emphasis))` or `(emphasis:1.2)`.
    *   Negative prompts: specify what you *don't* want in your image.
    *   Styles: save and apply prompt snippets easily.
    *   Prompt editing mid-generation.
    *   Composable-Diffusion: multiple prompts at once using uppercase `AND`, with weights for prompts.
    *   No token limit for prompts.
*   **Advanced Image Processing Tools:**
    *   Upscaling: SD Upscale, RealESRGAN, ESRGAN, SwinIR, Swin2SR, LDSR
    *   Face restoration tools: GFPGAN, CodeFormer
    *   Variations and Seed resizing for image refinement.
    *   Batch processing for bulk image generation.
    *   Highres Fix for creating high-resolution pictures in one click.
    *   Checkpoint Merger: Merge up to 3 checkpoints.
*   **Customization and Control:**
    *   Extensive settings: configure defaults, UI elements, and more.
    *   Sampling method selection with advanced noise control.
    *   Interrupt processing at any time.
    *   Generation parameters saved with images (PNG chunks/EXIF).
    *   Live image generation preview with progress bar.
    *   CLIP interrogator: generate prompts from images.
    *   Training tab with hypernetworks, embeddings, Loras, and image preprocessing.
    *   Custom scripts and extensions support.
*   **Integration and Compatibility:**
    *   API for programmatic access.
    *   Support for Stable Diffusion 2.0, Alt-Diffusion, and Segmind Stable Diffusion.
    *   Xformers integration for speed improvements.
    *   Support for dedicated inpainting models.
    *   Loading checkpoints in safetensors format.
    *   Eased resolution restriction: generated image's dimensions must be a multiple of 8 rather than 64.
    *   Reorder elements in the UI from settings screen.
*   **Additional Features:**
    *   Loopback processing.
    *   X/Y/Z plot for parameter exploration.
    *   Textual Inversion for custom embeddings.
    *   Tiling support for creating seamless textures.
    *   DeepDanbooru integration for anime prompts.
    *   Estimated completion time in progress bar.
    *   Aesthetic Gradients extension.

## Installation and Running

Detailed installation instructions are available for:

*   [NVidia GPUs](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-NVidia-GPUs)
*   [AMD GPUs](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-AMD-GPUs)
*   [Intel CPUs and GPUs](https://github.com/openvinotoolkit/stable-diffusion-webui/wiki/Installation-on-Intel-Silicon)
*   [Ascend NPUs](https://github.com/wangshuai09/stable-diffusion-webui/wiki/Install-and-run-on-Ascend-NPUs)
*   [Apple Silicon](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Installation-on-Apple-Silicon)

You can also use online services:

*   [List of Online Services](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Online-Services)

### Quick Start (Windows with NVidia GPUs):

1.  Download the `sd.webui.zip` from [v1.0.0-pre](https://github.com/AUTOMATIC1111/stable-diffusion-webui/releases/tag/v1.0.0-pre) and extract its contents.
2.  Run `update.bat`.
3.  Run `run.bat`.

### Alternative Automatic Installation on Windows:

1.  Install [Python 3.10.6](https://www.python.org/downloads/release/python-3106/), checking "Add Python to PATH".
2.  Install [git](https://git-scm.com/download/win).
3.  Clone the repository: `git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git`.
4.  Run `webui-user.bat`.

### Automatic Installation on Linux:

1.  Install dependencies.
2.  Navigate to your desired installation directory.
3.  Run `wget -q https://raw.githubusercontent.com/AUTOMATIC1111/stable-diffusion-webui/master/webui.sh` or `git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui`.
4.  Run `webui.sh`.

## Contributing

Contribute to the project via the [Contributing](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Contributing) guide.

## Documentation

Comprehensive documentation is available in the project's [wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki).

## Credits

Licenses for borrowed code can be found in `Settings -> Licenses` screen, and also in `html/licenses.html` file.