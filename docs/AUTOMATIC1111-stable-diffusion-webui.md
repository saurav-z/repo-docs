# Stable Diffusion Web UI: Unleash Your Creativity with AI Image Generation

**Transform your imagination into stunning visuals with the Stable Diffusion web UI, a user-friendly interface for creating AI-generated images.** ([Back to the original repo](https://github.com/AUTOMATIC1111/stable-diffusion-webui))

## Key Features

*   **Versatile Image Generation:**
    *   txt2img and img2img modes for diverse image creation.
    *   Outpainting and Inpainting for expanding and editing images.
    *   Color Sketch for generating images from sketches.
*   **Advanced Control & Customization:**
    *   Prompt Matrix for exploring image variations.
    *   Attention mechanism to refine specific text prompts.
    *   Loopback for iterative img2img processing.
    *   X/Y/Z plot for 3D image parameter exploration.
*   **Enhanced Image Quality & Editing:**
    *   Highres Fix for generating high-resolution images without distortions.
    *   Extras tab with GFPGAN, CodeFormer, RealESRGAN, ESRGAN, SwinIR, Swin2SR, and LDSR for upscaling and face restoration.
    *   Negative prompt for excluding unwanted elements.
    *   Styles for saving and applying prompt elements easily.
    *   Variations and Seed resizing for generating similar images with slight differences.
*   **AI-Powered Enhancements:**
    *   CLIP Interrogator for generating prompts from images.
    *   DeepDanbooru integration for anime prompt generation.
    *   Composable Diffusion: Supports multiple prompts for complex compositions.
*   **User-Friendly Experience:**
    *   Progress bar and live image generation preview.
    *   Generation parameters saved with images (PNG chunks/EXIF data).
    *   Settings page for customization.
    *   Mouseover hints for intuitive UI navigation.
    *   Automatic Installation via webui-user.bat (Windows) or webui.sh (Linux)
    *   API
*   **Community-Driven & Extensible:**
    *   Custom scripts and extensions (link to Wiki)
    *   Supports popular third-party models, like [Segmind Stable Diffusion](https://huggingface.co/segmind/SSD-1B).

## Installation and Running

Detailed installation instructions are available for various platforms:

*   **Nvidia GPUs:** (See [Install-and-Run-on-NVidia-GPUs](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-NVidia-GPUs))
*   **AMD GPUs:** (See [Install-and-Run-on-AMD-GPUs](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-AMD-GPUs))
*   **Intel CPUs/GPUs:** (See [Installation-on-Intel-Silicon](https://github.com/openvinotoolkit/stable-diffusion-webui/wiki/Installation-on-Intel-Silicon))
*   **Ascend NPUs:** (See [Install-and-run-on-Ascend-NPUs](https://github.com/wangshuai09/stable-diffusion-webui/wiki/Install-and-run-on-Ascend-NPUs))
*   **Apple Silicon:** (See [Installation-on-Apple-Silicon](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Installation-on-Apple-Silicon))

Alternatively, explore online services: ([List of Online Services](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Online-Services)).

### Windows Installation (Simplified)

1.  Download the latest release (`sd.webui.zip`).
2.  Run `update.bat`.
3.  Run `run.bat`.

### Linux Installation (Simplified)

1.  Install dependencies (see original README for details).
2.  Run `webui.sh` or clone the repo and then run `webui.sh`.

## Contributing

Contribute to the project by following the guidelines ([Contributing](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Contributing)).

## Documentation

Comprehensive documentation is available on the project's [wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki).

## Credits

The project utilizes code from various sources, with licenses available in `Settings -> Licenses` and `html/licenses.html`. A detailed list of credits is provided in the original README.