# Stable Diffusion Web UI: Unleash Your Creativity with AI-Powered Image Generation

**Stable Diffusion web UI is a user-friendly web interface for Stable Diffusion, enabling anyone to generate stunning images from text prompts with ease.** (Check out the original repo: [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui))

## Key Features

*   **Versatile Image Generation:** Create images using text-to-image (txt2img) and image-to-image (img2img) modes.
*   **One-Click Installation:** Easily set up the web UI with a simple install script (Python and Git required).
*   **Advanced Editing Tools:**
    *   **Outpainting & Inpainting:** Extend and refine images with intuitive editing features.
    *   **Color Sketch:** Generate images from color sketches.
    *   **Prompt Matrix:** Explore multiple variations of prompts efficiently.
    *   **Attention Mechanism:** Fine-tune image generation with attention to specific keywords using `((emphasis))` syntax.
    *   **Prompt Editing:** Modify prompts mid-generation for dynamic results.
*   **Upscaling and Enhancement:**
    *   **Stable Diffusion Upscale:** Improve image resolution directly within the UI.
    *   **Extras Tab:** Access advanced tools like GFPGAN, CodeFormer, RealESRGAN, ESRGAN, SwinIR, Swin2SR, and LDSR for face restoration and image upscaling.
*   **Customization & Control:**
    *   **Negative Prompt:** Specify what you *don't* want to see in your images.
    *   **Styles:** Save and apply prompt styles for consistent results.
    *   **Variations & Seed Resizing:** Generate diverse images with subtle variations.
    *   **Sampling Methods:** Choose from various sampling methods and adjust noise settings.
    *   **Checkpoint Merger:** Merge up to 3 checkpoints.
    *   **CLIP Interrogator:** Generate prompts from existing images.
    *   **Custom Scripts:** Extend functionality with community-created scripts.
*   **Advanced Techniques:**
    *   **Textual Inversion:** Train and use custom embeddings for unique artistic styles.
    *   **Loopback:** Run img2img multiple times.
    *   **X/Y/Z Plot:** Create 3D plots of images with different parameters.
    *   **Composable Diffusion:** Use multiple prompts simultaneously with weights.
    *   **DeepDanbooru Integration:** Generate anime-style tags for prompts.
*   **Additional Features:**
    *   Live prompt token length validation.
    *   Generation parameters saved with images (PNG/EXIF).
    *   Drag and drop functionality for images and prompts.
    *   Settings page for customization.
    *   Support for various extensions.
    *   API access.

## Installation

Detailed installation instructions are available in the [wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki).

**Quick Start (Windows with NVIDIA GPUs):**

1.  Download `sd.webui.zip` from [v1.0.0-pre](https://github.com/AUTOMATIC1111/stable-diffusion-webui/releases/tag/v1.0.0-pre) and extract its contents.
2.  Run `update.bat`.
3.  Run `run.bat`.

## Resources

*   **Documentation:** Comprehensive documentation is located in the project's [wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki).
*   **Credits:**  The web UI leverages a wealth of open-source projects; credits are detailed in the `Settings -> Licenses` screen and in `html/licenses.html`.

## Contributing

Contribute to the project [here](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Contributing).