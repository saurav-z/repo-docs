# Stable Diffusion Web UI: Generate Stunning Images with Ease

**Unleash your creativity and bring your imagination to life with the Stable Diffusion Web UI, a user-friendly web interface built with Gradio for creating AI-generated images.**  ([Original Repo](https://github.com/AUTOMATIC1111/stable-diffusion-webui))

## Key Features

*   **Versatile Image Generation Modes:** Utilize original txt2img and img2img modes for diverse image creation.
*   **Effortless Installation:** One-click installation and run scripts (Python and Git are required).
*   **Advanced Editing Tools:**
    *   Outpainting and Inpainting for expanding and refining images.
    *   Color Sketch for generating images from sketches.
    *   Prompt Matrix for exploring multiple variations.
*   **Powerful Upscaling and Enhancement:**
    *   Stable Diffusion Upscale for high-resolution results.
    *   Multiple Upscalers: GFPGAN, CodeFormer, RealESRGAN, ESRGAN, SwinIR, Swin2SR, and LDSR.
*   **Refined Prompt Control:**
    *   Attention mechanism for focusing on specific text elements within prompts.
    *   Negative prompt for specifying what you don't want in the image.
    *   Styles for saving and applying prompt snippets easily.
    *   Prompt Editing for changing prompts mid-generation.
*   **Innovative Techniques:**
    *   Textual Inversion for training custom embeddings.
    *   X/Y/Z plot for creating 3D image plots with varying parameters.
    *   Loopback for iterative img2img processing.
    *   Composable-Diffusion for using multiple prompts simultaneously.
    *   CLIP Interrogator for generating prompts from images.
*   **Batch Processing and Automation:**
    *   Batch Processing for processing multiple files with img2img.
    *   Generation parameters are saved with images (in PNG chunks or EXIF) for easy restoration and sharing.
*   **User-Friendly Interface:**
    *   Live image generation preview with progress bar.
    *   Settings page for customization.
    *   Mouseover hints for UI elements.
    *   Tiling support for creating seamless textures.
*   **Extensive Community Support and Extensions:**
    *   Custom scripts and a History tab for convenient image management.
    *   Hypernetworks, Loras, and a UI for easy integration of custom models.
    *   Aesthetic Gradients extension for generating images with specific aesthetics.
*   **Hardware Support and Optimization:**
    *   Supports 4GB video cards (with reports of 2GB working).
    *   xformers for significant speed increases on select GPUs.
*   **Advanced Features:**
    *   Checkpoint Merger for combining models.
    *   Support for Stable Diffusion 2.0 and Alt-Diffusion.
    *   API for programmatic access and integration.
    *   Training tab with options for hypernetworks and embeddings.
    *   VAE selection in settings.
    *   Support for Segmind Stable Diffusion.

## Installation and Running

Detailed installation instructions are available for various platforms, including:

*   NVIDIA GPUs (Recommended)
*   AMD GPUs
*   Intel CPUs/GPUs
*   Apple Silicon
*   Ascend NPUs

**For a quick start on Windows with NVIDIA GPUs, follow these steps:**

1.  Download `sd.webui.zip` from the [releases](https://github.com/AUTOMATIC1111/stable-diffusion-webui/releases).
2.  Extract the contents.
3.  Run `update.bat`.
4.  Run `run.bat`.

**Alternatively, use automatic installation on Windows:**

1.  Install [Python 3.10.6](https://www.python.org/downloads/release/python-3106/).
2.  Install [Git](https://git-scm.com/download/win).
3.  Clone the repository: `git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git`.
4.  Run `webui-user.bat`.

**For Linux and Apple Silicon instructions, please refer to the detailed guides within the project's wiki.**

## Contributing

Contribute to the project - [Contributing](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Contributing)

## Documentation

The comprehensive documentation is available on the project's [wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki).

## Credits

This project incorporates code and ideas from numerous sources. A complete list of credits and licenses can be found in the `Settings -> Licenses` screen and `html/licenses.html` file within the web UI.