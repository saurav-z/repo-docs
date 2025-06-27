# Stable Diffusion Web UI: Your Gateway to AI-Powered Image Generation

Unleash your creativity and bring your imagination to life with the **Stable Diffusion web UI**, a powerful and user-friendly interface for the Stable Diffusion image generation model.  Explore and experiment with the latest in AI image generation, developed by [AUTOMATIC1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui).

## Key Features

*   **Core Image Generation:**
    *   txt2img and img2img modes for generating images from text prompts or modifying existing images.
    *   Outpainting and Inpainting capabilities for extending and editing images.
    *   Color Sketch feature for image generation from sketches.
*   **Advanced Prompting and Control:**
    *   Attention mechanisms to prioritize specific parts of your prompts (`((tuxedo))`).
    *   Negative prompts to specify what you *don't* want in your images.
    *   Prompt editing during image generation for dynamic control.
    *   Styles to save and reuse prompt elements.
*   **AI Enhancement Tools:**
    *   GFPGAN and CodeFormer for face restoration.
    *   RealESRGAN and ESRGAN for image upscaling.
    *   SwinIR, Swin2SR, and LDSR for advanced upscaling.
*   **Workflow & Customization:**
    *   Batch processing for efficient image generation workflows.
    *   Seed resizing and variations for generating similar images with slight differences.
    *   X/Y/Z plot for exploring parameter combinations.
    *   Custom scripts and extensions for enhanced functionality.
    *   Checkpoint merging for combining models.
    *   Loras, Hypernetworks, Embeddings for greater creative control.
*   **Convenience & Efficiency:**
    *   One-click installation (with Python and Git pre-requisites).
    *   Generation parameters are saved with images (PNG chunks/EXIF) for easy reproduction and sharing.
    *   Interrupt processing, 4GB video card support.
    *   Progress bar and live image generation preview.
    *   No token limit for prompts.
    *   Support for dedicated inpainting models.
    *   Aesthetic Gradients: generate images with a specific aesthetic
    *   Support for Stable Diffusion 2.0 and Alt-Diffusion.
*   **Additional features:**
    *   Tiling Support
    *   Clip Interrogator
    *   DeepDanbooru integration
    *   API

## Installation and Running

Detailed installation instructions, including options for NVIDIA, AMD, and Intel GPUs, as well as online service options are found in the [project's wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki).

**Quick Installation (Windows with NVIDIA):**

1.  Download `sd.webui.zip` from the latest release.
2.  Run `update.bat`.
3.  Run `run.bat`.

## Contributing

Learn how to contribute to this project by checking out the [contributing guidelines](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Contributing).

## Credits

A comprehensive list of credits and licenses for the libraries and models used can be found in the `Settings -> Licenses` screen within the web UI and in the `html/licenses.html` file.