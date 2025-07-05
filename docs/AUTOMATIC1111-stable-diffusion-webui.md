# Stable Diffusion Web UI: Generate Stunning Images with Ease

Unleash your creativity and bring your imagination to life with the Stable Diffusion Web UI, a powerful and user-friendly web interface for the Stable Diffusion image generation model. [See the original repository](https://github.com/AUTOMATIC1111/stable-diffusion-webui) for the source.

## Key Features

*   **Core Functionality:**
    *   Text-to-Image (txt2img) and Image-to-Image (img2img) modes for versatile generation.
    *   Outpainting and Inpainting for seamless image editing.
    *   Color Sketch to bring your vision to life with a colored outline.
*   **Advanced Prompting:**
    *   Attention mechanism for fine-grained control over image elements.
    *   Negative prompts to specify unwanted elements in the generated image.
    *   Styles feature to save and apply prompt elements for consistency.
    *   Prompt editing to adjust prompts mid-generation.
    *   No token limit for prompts, giving you more freedom.
*   **Image Enhancement and Upscaling:**
    *   Upscaling capabilities with options such as GFPGAN, CodeFormer, RealESRGAN, ESRGAN, SwinIR, Swin2SR, and LDSR for high-quality results.
    *   Highres Fix for generating high-resolution images without distortion.
    *   Seed resizing for generating variations of the same image at slightly different resolutions.
*   **Workflow & Efficiency:**
    *   One-click installation and run script for ease of use (after installing Python and Git).
    *   Live image generation preview and progress bar.
    *   Batch Processing to process a group of files using img2img
    *   X/Y/Z plot for 3D image parameter exploration.
    *   Interrupt processing at any time.
    *   Reloading checkpoints on the fly.
    *   Checkpoint Merger for combining multiple checkpoints.
    *   Generate forever option
*   **Customization & Extensions:**
    *   Custom scripts and community extensions for added functionality.
    *   Integration with DeepDanbooru for anime prompts.
    *   Support for Composable-Diffusion to use multiple prompts.
    *   Loras (same as Hypernetworks but more pretty)
    *   A separate UI where you can choose, with preview, which embeddings, hypernetworks or Loras to add to your prompt
*   **Additional Features:**
    *   Generation parameters are saved with the image (PNG chunks/EXIF).
    *   Read Generation Parameters Button to load parameters into the UI.
    *   Settings page for UI customization.
    *   Support for multiple Samplers.
    *   CLIP interrogator for prompt generation.
    *   Training tab for hypernetworks and embeddings.
    *   API for programmatic access.
    *   Support for dedicated inpainting models.
    *   Support for Stable Diffusion 2.0 and Alt-Diffusion.
    *   Segmind Stable Diffusion support
    *   Load checkpoints in safetensors format

## Installation

Detailed installation instructions are available in the wiki. General steps include:

1.  **Prerequisites:** Ensure you have Python and Git installed.
2.  **Download:** Download the repository.
3.  **Run:** Execute the appropriate script (`webui-user.bat` for Windows, `webui.sh` for Linux) to start the web UI.

[Installation instructions and other helpful information is located in the project's wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki).

## Contributing

Contribute to the project by following the guidelines in the [Contributing](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Contributing) section of the wiki.

## Credits

The Stable Diffusion Web UI leverages various open-source projects. Please refer to the `Settings -> Licenses` screen or the `html/licenses.html` file for a complete list of credits and licenses.