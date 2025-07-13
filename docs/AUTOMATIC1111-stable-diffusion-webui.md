# Stable Diffusion Web UI: Unleash Your Creativity with AI-Powered Image Generation

**Create stunning images from text prompts with the user-friendly Stable Diffusion web UI, your gateway to the exciting world of AI art.** ([Original Repo](https://github.com/AUTOMATIC1111/stable-diffusion-webui))

## Key Features

*   **Versatile Generation Modes:** Leverage original `txt2img` and `img2img` modes.
*   **Effortless Setup:** One-click install and run scripts (Python and Git installation required).
*   **Advanced Editing & Control:**
    *   Outpainting and inpainting capabilities
    *   Color Sketch feature
    *   Prompt Matrix for experimentation
    *   Attention mechanism to emphasize specific text in prompts.
    *   Negative prompts to refine images.
    *   Prompt Editing, batch processing, Highres Fix.
*   **Upscaling & Enhancement:**
    *   Stable Diffusion Upscale.
    *   GFPGAN, CodeFormer for face restoration.
    *   RealESRGAN, ESRGAN, SwinIR, Swin2SR, and LDSR upscalers.
*   **Customization & Control:**
    *   Tiling support for seamless textures.
    *   Styles to save and apply prompt elements.
    *   Variations and Seed resizing options.
    *   CLIP interrogator for prompt generation from images.
    *   X/Y/Z plot to draw a 3D plot of images.
*   **Community-Driven Enhancements:**
    *   Extensive support for custom scripts and extensions.
    *   Composable-Diffusion for complex prompt combinations.
    *   DeepDanbooru integration for anime prompts.
    *   Support for LORAs, Hypernetworks, and embeddings.
*   **Performance & Compatibility:**
    *   4GB video card support (with reports of 2GB working).
    *   Progress bar and real-time image preview.
    *   xformers integration for improved performance.
    *   Loading checkpoints in safetensors format.
*   **Additional Features:**
    *   Read generation parameters from images.
    *   Settings page for customization.
    *   API for integration.
    *   Support for dedicated inpainting models.

## Installation and Running

Follow the installation instructions based on your operating system and hardware:

*   **NVidia GPUs (Recommended):** See the [installation guide](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-NVidia-GPUs)
*   **AMD GPUs:** [Installation guide](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-AMD-GPUs)
*   **Intel CPUs/GPUs:** [Installation guide](https://github.com/openvinotoolkit/stable-diffusion-webui/wiki/Installation-on-Intel-Silicon) (external wiki page)
*   **Ascend NPUs:** [Installation guide](https://github.com/wangshuai09/stable-diffusion-webui/wiki/Install-and-run-on-Ascend-NPUs) (external wiki page)
*   **Apple Silicon:** [Installation instructions](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Installation-on-Apple-Silicon)

For quick setup on Windows with NVidia GPUs, using the release package:

1.  Download `sd.webui.zip` from [v1.0.0-pre](https://github.com/AUTOMATIC1111/stable-diffusion-webui/releases/tag/v1.0.0-pre) and extract its contents.
2.  Run `update.bat`.
3.  Run `run.bat`.

Alternatively, you can use the automatic installation methods for Windows and Linux, or explore online services detailed on the wiki.

## Contributing

Contribute to the project by following the guide [Contributing](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Contributing)

## Documentation

The detailed documentation is available on the project's [wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki).

## Credits

The Stable Diffusion web UI builds upon the work of numerous individuals and projects.  See the `Credits` section in the original README.