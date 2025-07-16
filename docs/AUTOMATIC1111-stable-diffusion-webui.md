# Stable Diffusion Web UI: Your Gateway to AI-Powered Image Generation

Unleash your creativity with the Stable Diffusion web UI, a powerful and user-friendly interface built with Gradio for generating stunning images from text prompts. [Explore the original repository](https://github.com/AUTOMATIC1111/stable-diffusion-webui) for more details and the latest updates.

## Key Features

*   **Versatile Image Generation:**
    *   Original `txt2img` and `img2img` modes for text-to-image and image-to-image creation.
    *   Outpainting and Inpainting for expanding and modifying images.
    *   Color Sketch for generating images from sketches.
    *   Prompt Matrix to explore variations based on different prompts.
*   **Advanced Image Enhancement:**
    *   Stable Diffusion Upscale for high-resolution results.
    *   Face restoration tools: GFPGAN and CodeFormer.
    *   Multiple upscalers: RealESRGAN, ESRGAN, SwinIR, Swin2SR, and LDSR.
*   **Intuitive Control and Customization:**
    *   Attention mechanism to specify important text elements using parentheses.
    *   Loopback processing for iterative image refinement.
    *   X/Y/Z plot to visualize parameter variations.
    *   Textual Inversion for custom embeddings, supporting multiple embeddings and half-precision floating-point numbers.
    *   Negative prompts to specify unwanted elements.
    *   Styles to save and easily apply prompt snippets.
    *   Variations for generating similar images with subtle differences.
    *   Seed resizing for resolution adjustments.
*   **User-Friendly Interface:**
    *   Live image generation preview and progress bar.
    *   Generation parameters saved with each image (in PNG chunks or EXIF).
    *   Drag-and-drop functionality for parameters and images.
    *   "Read Generation Parameters" button to load saved parameters.
    *   Mouseover hints for UI elements.
    *   Settings page for customization.
    *   CLIP interrogator to guess prompts from images.
    *   Prompt editing for mid-generation changes.
*   **Advanced Features:**
    *   Batch processing for multiple images.
    *   Highres Fix for enhanced resolution.
    *   Checkpoint Merger for combining models.
    *   Custom scripts via extensions.
    *   Composable-Diffusion for multiple prompts.
    *   DeepDanbooru integration for anime prompts.
    *   xformers integration for performance boost (use `--xformers` command-line argument).
    *   Training tab for hypernetworks, embeddings, and LORAs.
    *   API for integration with other applications.
    *   Support for Stable Diffusion 2.0 and Alt-Diffusion.
    *   Segmind Stable Diffusion support.

## Installation

Detailed installation instructions are available in the [Installation and Running](#installation-and-running) section of the original README.