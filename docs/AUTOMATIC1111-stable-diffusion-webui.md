# Stable Diffusion Web UI: Unleash Your Creativity with AI-Powered Image Generation

This powerful web interface, built with Gradio, puts the incredible potential of Stable Diffusion at your fingertips, allowing you to generate stunning images from text prompts. [Check out the original repository](https://github.com/AUTOMATIC1111/stable-diffusion-webui) for more details.

## Key Features

*   **Core Functionality:**
    *   Original txt2img and img2img modes for versatile image creation.
    *   Outpainting and Inpainting capabilities for extending and modifying images.
    *   Color Sketch for generating images from basic sketches.
    *   Prompt Matrix to explore variations and combinations of prompts.
*   **Advanced Image Manipulation:**
    *   Stable Diffusion Upscale for enhancing image resolution.
    *   Attention mechanism for fine-tuning prompt focus (e.g., `((tuxedo))` for emphasis).
    *   Loopback for iterative img2img processing.
    *   X/Y/Z plot for generating 3D image plots with different parameters.
*   **AI-Powered Enhancements:**
    *   GFPGAN, CodeFormer, and RealESRGAN for face restoration and upscaling.
    *   Support for ESRGAN, SwinIR, Swin2SR, and LDSR upscalers.
    *   CLIP interrogator for generating prompts from images.
    *   DeepDanbooru integration for anime-style prompts.
*   **Workflow & Customization:**
    *   Seamless Seed and Generation Parameter Management: parameters saved with images, drag-and-drop functionality.
    *   Textual Inversion for custom embeddings.
    *   Styles for saving and applying prompt elements.
    *   Negative prompt for specifying unwanted elements.
    *   Comprehensive Settings Page.
    *   Custom scripts with community extensions.
    *   Checkpoint Merger for combining models.
    *   Hypernetworks and Loras support for advanced image generation styles.
    *   Composable Diffusion for using multiple prompts.
*   **Performance and Efficiency:**
    *   Optimized for 4GB+ video cards (with reports of 2GB working).
    *   xformers support for significant speed increases on select GPUs.
    *   Live preview and progress bar with estimated completion time.
*   **Additional Features:**
    *   Tiling support for creating seamless textures.
    *   Batch Processing for processing multiple files at once.
    *   API for programmatic access.
    *   Support for Stable Diffusion 2.0, Alt-Diffusion, and Segmind Stable Diffusion.
    *   Reloading checkpoints on the fly.

## Installation and Running

Detailed installation instructions are available for various setups:

*   **NVidia GPUs:**  [NVidia Installation Guide](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-NVidia-GPUs)
*   **AMD GPUs:**  [AMD Installation Guide](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-AMD-GPUs)
*   **Intel CPUs/GPUs:** [Intel Installation Guide](https://github.com/openvinotoolkit/stable-diffusion-webui/wiki/Installation-on-Intel-Silicon) (external wiki page)
*   **Ascend NPUs:** [Ascend Installation Guide](https://github.com/wangshuai09/stable-diffusion-webui/wiki/Install-and-run-on-Ascend-NPUs) (external wiki page)
*   **Apple Silicon:** [Apple Silicon Installation Guide](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Installation-on-Apple-Silicon)

**Windows Installation (NVidia GPUs - using release package):**

1.  Download `sd.webui.zip` from [v1.0.0-pre](https://github.com/AUTOMATIC1111/stable-diffusion-webui/releases/tag/v1.0.0-pre) and extract its contents.
2.  Run `update.bat`.
3.  Run `run.bat`.

**Automatic Installation (Windows):**

1.  Install [Python 3.10.6](https://www.python.org/downloads/release/python-3106/) (and add it to your PATH).
2.  Install [git](https://git-scm.com/download/win).
3.  Clone the repository: `git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git`.
4.  Run `webui-user.bat`.

**Automatic Installation (Linux):**

1.  Install dependencies (Debian/Ubuntu, Red Hat/Fedora, openSUSE, or Arch-based).
2.  Navigate to your desired install directory and use either `wget` or `git clone` to get the webui files.
3.  Run `webui.sh`.
4.  Check `webui-user.sh` for options.

## Contributing

Contribute to the project! [See the Contributing Guide](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Contributing).

## Documentation

Comprehensive documentation is available in the project's [wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki).

## Credits

A list of credits and licenses is available in `Settings -> Licenses` and `html/licenses.html`.