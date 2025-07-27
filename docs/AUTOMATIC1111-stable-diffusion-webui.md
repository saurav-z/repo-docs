# Stable Diffusion Web UI: Generate Stunning Images with Ease

**Unleash your creativity and effortlessly generate breathtaking images with the Stable Diffusion web UI, a user-friendly interface built with Gradio.** ([Original Repository](https://github.com/AUTOMATIC1111/stable-diffusion-webui))

## Key Features

*   **Versatile Image Generation:**
    *   Original txt2img and img2img modes for text-to-image and image-to-image creation.
    *   Outpainting and inpainting capabilities for extending and modifying images.
    *   Color Sketch feature to generate images from sketches.
*   **Advanced Image Editing & Control:**
    *   Prompt Matrix for experimenting with multiple prompts.
    *   Attention mechanism for focusing the model on specific parts of your prompt.
    *   Negative prompt to specify what you *don't* want in the image.
    *   Loopback for iterative img2img processing.
    *   X/Y/Z plot for generating 3D plots of images with varying parameters.
    *   Prompt editing during image generation for real-time adjustments.
*   **Powerful Image Enhancement & Upscaling:**
    *   Stable Diffusion Upscale for higher-resolution images.
    *   Extras tab with integrated tools like GFPGAN, CodeFormer, RealESRGAN, ESRGAN, SwinIR, Swin2SR, and LDSR for face restoration and image upscaling.
    *   Highres Fix for creating high-resolution pictures without distortions.
*   **Customization & Extensibility:**
    *   Textual Inversion: Train and use custom embeddings.
    *   Styles for saving and applying prompt elements.
    *   Variations for generating slight image variations.
    *   Custom scripts via community extensions.
    *   Checkpoint Merger to merge multiple checkpoints.
    *   Hypernetworks and LoRAs for advanced image control.
    *   Aesthetic Gradients for generating images with a specific aesthetic.
*   **User-Friendly Interface & Features:**
    *   One-click installation and run scripts for ease of use.
    *   Read Generation Parameters button to load image generation parameters.
    *   Seed resizing for generating similar images at different resolutions.
    *   Live prompt token length validation.
    *   Generation parameters saved with images for easy reproduction.
    *   Settings page for comprehensive customization.
    *   Mouseover hints for UI elements.
    *   Tiling support for creating seamless textures.
    *   Progress bar with live image preview.
    *   CLIP interrogator for generating prompts from images.
    *   Batch Processing for processing multiple files with img2img.
    *   Reloading checkpoints on the fly.
    *   DeepDanbooru integration for anime prompts.
    *   API.
    *   Estimated completion time in progress bar.
    *   Reorder elements in the UI from settings screen.
*   **Performance & Compatibility:**
    *   4GB video card support (with reports of 2GB working).
    *   xformers integration for significant speed increases on select GPUs.
    *   Supports Stable Diffusion 2.0 and Alt-Diffusion models.
    *   Supports Segmind Stable Diffusion.
    *   Eased resolution restriction.
    *   Support for dedicated inpainting model by RunwayML.
*   **Advanced Control & Analysis**
    *   Comprehensive image analysis features, and generation parameter preservation.

## Installation and Running

Detailed installation instructions are available for various platforms:

*   **NVidia GPUs:** [Install and Run on NVidia GPUs](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-NVidia-GPUs)
*   **AMD GPUs:** [Install and Run on AMD GPUs](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-AMD-GPUs)
*   **Intel CPUs & GPUs:** [Installation on Intel Silicon](https://github.com/openvinotoolkit/stable-diffusion-webui/wiki/Installation-on-Intel-Silicon)
*   **Ascend NPUs:** [Install and run on Ascend NPUs](https://github.com/wangshuai09/stable-diffusion-webui/wiki/Install-and-run-on-Ascend-NPUs)
*   **Apple Silicon:** [Installation on Apple Silicon](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Installation-on-Apple-Silicon)

Or use online services: [List of Online Services](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Online-Services)

### Quick Installation on Windows 10/11 (NVidia)
1. Download `sd.webui.zip` from [v1.0.0-pre](https://github.com/AUTOMATIC1111/stable-diffusion-webui/releases/tag/v1.0.0-pre) and extract its contents.
2. Run `update.bat`.
3. Run `run.bat`.

### Automatic Installation on Windows
1. Install Python 3.10.6 and git.
2. Download the repository.
3. Run `webui-user.bat`.

### Automatic Installation on Linux
1. Install dependencies.
2. Execute `webui.sh`.
3. Check `webui-user.sh` for options.

## Contributing
Contribute to the project via the [Contributing](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Contributing) guide.

## Documentation
Comprehensive documentation is available in the project's [wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki).

## Credits
See `Settings -> Licenses` or `html/licenses.html` for licenses.  Additional credits are available in the original README.