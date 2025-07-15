# Stable Diffusion Web UI: Unleash Your Creativity with AI Image Generation

**Create stunning AI-generated images with the user-friendly Stable Diffusion web UI, offering a comprehensive suite of features for both beginners and experienced users.** ([See the original repository](https://github.com/AUTOMATIC1111/stable-diffusion-webui))

## Key Features:

*   **Core Generation:**
    *   Txt2img and Img2img modes for text-to-image and image-to-image generation.
    *   Inpainting, outpainting, and color sketch features for image editing and expansion.
    *   Prompt Matrix and other advanced features to generate multiple images with different parameters.
*   **Advanced Image Manipulation:**
    *   Upscaling options: Stable Diffusion Upscale, RealESRGAN, CodeFormer, GFPGAN, LDSR, and more.
    *   Attention mechanism for controlling the focus of the model on specific text prompts.
    *   Loopback and X/Y/Z plot features for advanced image generation and parameter exploration.
    *   Variations and seed resizing for generating slight variations of your images.
*   **Customization & Control:**
    *   Negative prompts to exclude unwanted elements from your images.
    *   Styles feature to save and apply prompt snippets easily.
    *   Textual Inversion for training and using custom embeddings.
    *   Checkpoint merging for combining different models.
    *   Custom scripts via community extensions for extending functionality.
    *   Comprehensive settings page for configuring the UI.
    *   Dynamic prompt editing, batch processing, and high-res fix options for workflow efficiency.
*   **AI-Powered Enhancements:**
    *   CLIP interrogator to generate prompts from images.
    *   DeepDanbooru integration for anime-style prompts.
    *   Aesthetic Gradients extension for aesthetic-focused image generation.
    *   Support for Stable Diffusion 2.0, Alt-Diffusion, and Segmind Stable Diffusion models.
*   **Efficiency & Performance:**
    *   4GB video card support, with reports of functionality on 2GB cards.
    *   Xformers integration for significant speed improvements on select GPUs.
    *   Progress bar and live image preview.
    *   Estimated completion time displayed in the progress bar.
*   **User Experience:**
    *   Generation parameters saved with images (in PNG chunks/EXIF data).
    *   "Read Generation Parameters" button to load existing image parameters into the UI.
    *   Mouseover hints and UI element customization via text configuration.
    *   Support for loading and utilizing various VAEs (Variational Autoencoders).
    *   Reorder elements in the UI via settings screen.

## Installation and Running

Detailed installation instructions for various operating systems and hardware configurations are available in the project's [wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki).  Here are some summarized installation steps:

### Automatic Installation on Windows
1. Install Python 3.10.6 (and add it to PATH) and git.
2. Download the repository using `git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git`.
3. Run `webui-user.bat`.

### Automatic Installation on Linux
1. Install dependencies.
2. Clone the repository.
3. Run `webui.sh`.

## Contributing

Contributions are welcome.  See the [Contributing](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Contributing) guidelines for more information.

## Credits

See the `Credits` section in the original README, or go to `Settings -> Licenses` in the UI for a list of contributors and libraries used.