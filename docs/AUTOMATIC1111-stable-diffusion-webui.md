# Stable Diffusion Web UI: Unleash Your Creativity with AI-Powered Image Generation

**Create stunning images from text prompts with the Stable Diffusion web UI, a powerful and user-friendly interface.**  ([Visit the Original Repo](https://github.com/AUTOMATIC1111/stable-diffusion-webui))

![Stable Diffusion Web UI Screenshot](screenshot.png)

## Key Features

*   **Core Image Generation:**
    *   Original `txt2img` and `img2img` modes for generating images from text or modifying existing images.
    *   Outpainting, Inpainting, and Color Sketch modes for creative image manipulation.
    *   Highres Fix for producing high-resolution pictures in one click.
*   **Advanced Control & Customization:**
    *   Prompt Matrix and  Textual Inversion for fine-grained control over image generation.
    *   Negative prompts to specify unwanted elements in the generated images.
    *   Prompt Editing to change prompts mid-generation.
    *   Styles feature for saving and applying prompt snippets easily.
    *   X/Y/Z plot for exploring different parameters combinations.
    *   Variations & Seed resizing to generate images with tiny differences.
    *   Sampling method selection with adjust sampler eta values (noise multiplier).
    *   CLIP Interrogator for generating prompts from images.
    *   Control over attention to specific text parts.
    *   Custom scripts with many community extensions.
    *   Composable-Diffusion a way to use multiple prompts at once.
    *   No token limit for prompts.
    *   Hypernetworks & LORAs for custom styles.
    *   DeepDanbooru integration for anime prompts.
*   **Enhanced Features:**
    *   Extras tab for face restoration and upscaling using GFPGAN, CodeFormer, RealESRGAN, ESRGAN, SwinIR, Swin2SR, and LDSR.
    *   Checkpoint Merger to merge up to 3 checkpoints into one.
    *   Tiling support to create tileable images.
    *   Generation parameters saved with images (PNG/EXIF).
    *   Batch Processing for processing multiple images.
    *   Estimated completion time in progress bar.
    *   Support for dedicated inpainting models by RunwayML.
    *   API for programmatic access.
    *   Support for Stable Diffusion 2.0 & Alt-Diffusion.
    *   Now with a license!
    *   Reorder elements in the UI from the settings screen.
    *   Segmind Stable Diffusion support.
*   **User-Friendly Interface:**
    *   One-click install script (requires Python and Git).
    *   Live preview during image generation.
    *   Settings page for customization.
    *   Mouseover hints for UI elements.
    *   Read Generation Parameters Button.
    *   Ability to change defaults/mix/max/step values for UI elements via text config.
    *   Reloading checkpoints on the fly.
    *   Optional History Tab
*   **Performance and Compatibility:**
    *   4GB video card support (also reports of 2GB working).
    *   Correct seeds for batches.
    *   Live prompt token length validation.
    *   xformers support for faster processing on select GPUs.
    *   Eased resolution restriction.

## Installation and Running

Detailed installation instructions are available for various platforms.
*   **[NVidia GPUs](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-NVidia-GPUs)** (Recommended)
*   **[AMD GPUs](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-AMD-GPUs)**
*   **[Intel CPUs & GPUs](https://github.com/openvinotoolkit/stable-diffusion-webui/wiki/Installation-on-Intel-Silicon)** (External Wiki)
*   **[Ascend NPUs](https://github.com/wangshuai09/stable-diffusion-webui/wiki/Install-and-run-on-Ascend-NPUs)** (External Wiki)
*   **[Apple Silicon](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Installation-on-Apple-Silicon)**

### Windows Installation (Simplified)

1.  Download the latest release (sd.webui.zip).
2.  Run `update.bat`.
3.  Run `run.bat`.
    For more detailed instructions, see the wiki.

### Automatic Installation (Windows)

1.  Install Python 3.10.6 and Git.
2.  Clone the repository.
3.  Run `webui-user.bat`.

### Automatic Installation (Linux)

1.  Install dependencies.
2.  Run the installation script.
3.  Run `webui.sh`.

## Contributing

Contribute to the project: [Contributing](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Contributing)

## Documentation

Comprehensive documentation is available on the project's [wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki).

## Credits

This project utilizes code and ideas from various open-source projects. See `Settings -> Licenses` or `html/licenses.html` for detailed credits.