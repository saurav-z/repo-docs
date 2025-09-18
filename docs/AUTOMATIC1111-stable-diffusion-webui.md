# Stable Diffusion Web UI: Unleash Your Creativity with AI-Powered Image Generation

**Create stunning images from text prompts using the Stable Diffusion web UI – a powerful and user-friendly interface for generating AI art.**  [Explore the original repo](https://github.com/AUTOMATIC1111/stable-diffusion-webui) to get started.

## Key Features

*   **Core Image Generation:**
    *   txt2img and img2img modes for versatile image creation.
    *   Outpainting and inpainting capabilities for image manipulation.
    *   Color Sketch feature for creative guidance.
*   **Advanced Prompting & Control:**
    *   Prompt Matrix for exploring variations.
    *   Attention mechanism (`((emphasis))` syntax) for refined control over image elements.
    *   Negative prompts to specify undesired elements.
    *   Styles feature to save and apply prompt snippets.
    *   Textual Inversion for custom embeddings and artistic styles.
    *   Composable Diffusion for creating images from multiple prompts.
*   **Enhancements & Upscaling:**
    *   GFPGAN, CodeFormer, and RealESRGAN for face restoration.
    *   Upscalers: ESRGAN, SwinIR, Swin2SR, and LDSR.
    *   Highres Fix for generating high-resolution images without distortion.
    *   Seed resizing for creating variations.
*   **User-Friendly Interface:**
    *   One-click installation (requires Python and Git).
    *   Progress bar and live image preview.
    *   Generation parameters saved with images (PNG/EXIF).
    *   Read Generation Parameters Button to load parameters.
    *   Settings page for customization.
    *   Mouseover hints for UI elements.
    *   Tiling support for creating seamless textures.
*   **Advanced Features & Extensions:**
    *   X/Y/Z plot for parameter exploration.
    *   Checkpoint Merger for combining models.
    *   Custom scripts and community extensions for enhanced functionality.
    *   CLIP interrogator to guess prompts.
    *   Batch Processing and Img2img Alternative methods.
    *   Training tab for hypernetworks and embeddings.
    *   API for integration.
    *   Support for Stable Diffusion 2.0 and Alt-Diffusion.
*   **Integration and Models:**
    *   DeepDanbooru integration for anime prompts.
    *   Support for dedicated inpainting model by RunwayML.
    *   Support for Segmind Stable Diffusion.
    *   Loras (same as Hypernetworks but more pretty)
    *   Aesthetic Gradients extension

## Installation and Running

Follow the detailed instructions in the project's [wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki) for NVidia, AMD, and Intel GPUs, or explore online service options.

**Windows Quick Start (NVidia GPU):**

1.  Download `sd.webui.zip` from the releases.
2.  Run `update.bat`.
3.  Run `run.bat`.

**Automatic Installation (Windows):**

1.  Install Python 3.10.6 and add it to your PATH.
2.  Install Git.
3.  Clone the repository using `git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git`.
4.  Run `webui-user.bat`.

**Automatic Installation (Linux):**
Follow the instructions in the original README or the wiki.

**Installation on Apple Silicon:**
Follow the instructions [here](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Installation-on-Apple-Silicon).

## Contributing

Contribute to the project by following the [contributing guidelines](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Contributing).

## Documentation

Find comprehensive documentation on the project's [wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki).

## Credits

This project leverages code from various open-source projects. See the `Settings -> Licenses` screen and `html/licenses.html` for a complete list of credits.