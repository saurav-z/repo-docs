# Stable Diffusion Web UI: Your Gateway to AI-Powered Image Generation

Unleash your creativity and generate stunning images with the Stable Diffusion Web UI, a user-friendly web interface built on the Gradio library, providing a powerful suite of features for text-to-image and image-to-image generation.  [Visit the original repository for more details](https://github.com/AUTOMATIC1111/stable-diffusion-webui).

## Key Features

*   **Core Generation Modes:**
    *   txt2img and img2img modes for text-to-image and image-to-image generation.
    *   Outpainting and Inpainting for extending or modifying existing images.
    *   Color Sketch to generate images from your sketches.
    *   Prompt Matrix to explore variations using multiple prompts.

*   **Advanced Image Manipulation:**
    *   Stable Diffusion Upscale for enhancing image resolution.
    *   Attention mechanism control to fine-tune the model's focus on specific words.
    *   Loopback to iterate on image processing.
    *   X/Y/Z plot for creating 3D image variations.

*   **Enhanced Prompting and Customization:**
    *   Negative prompt to specify unwanted elements in the images.
    *   Styles for saving and applying prompt snippets.
    *   Variations to generate similar images with slight differences.
    *   Seed resizing to generate different image sizes from the same seed.
    *   CLIP interrogator to generate prompts from images.
    *   Prompt Editing for mid-generation adjustments.
    *   No token limit for prompts

*   **AI-Powered Enhancements:**
    *   GFPGAN and CodeFormer for face restoration.
    *   RealESRGAN, ESRGAN, SwinIR, Swin2SR, and LDSR for image upscaling.
    *   DeepDanbooru integration for generating anime-style tags.
    *   CLIP interrogator for generating prompts from images.

*   **User-Friendly Interface & Features:**
    *   Live image generation preview.
    *   Generation parameters saved with images (PNG chunks and EXIF).
    *   Read Generation Parameters Button to load generation parameters into the UI.
    *   Settings page for customization.
    *   Mouseover hints for UI elements.
    *   Tiling support for creating tileable textures.
    *   Batch Processing for processing a group of files using img2img.
    *   Highres Fix to produce high resolution pictures without usual distortions.
    *   Reloading checkpoints on the fly.
    *   Checkpoint Merger tab for merging checkpoints.
    *   Hypernetworks, LoRAs and embeddings options.
    *   A separate UI where you can choose, with preview, which embeddings, hypernetworks or LoRAs to add to your prompt
    *   Estimated completion time in progress bar.
    *   API.

*   **Advanced Features & Extensions:**
    *   Custom scripts via community extensions.
    *   Composable-Diffusion for combining multiple prompts.
    *   Xformers for accelerated performance on select GPUs.
    *   Support for various Stable Diffusion models (2.0 and Alt-Diffusion).
    *   Aesthetic Gradients extension for generating images with a specific aesthetic.

## Installation

Detailed installation instructions for different platforms can be found in the wiki, including NVidia, AMD, Intel, and Apple Silicon.

*   **Windows:** Instructions are available for both automatic and manual setups.
*   **Linux:** Automatic installation scripts are provided.
*   **Apple Silicon:**  See the wiki for instructions.

## Contributing

Contributions are welcome!  Consult the project's wiki for guidelines on how to contribute.

## Documentation

Comprehensive documentation, including installation guides, feature explanations, and troubleshooting tips, is available in the project's [wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki).

## Credits

The project relies on a number of open-source technologies. Licenses for borrowed code are listed in the `Settings -> Licenses` screen and in the `html/licenses.html` file.