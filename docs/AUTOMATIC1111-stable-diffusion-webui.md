# Stable Diffusion Web UI: Unleash Your Creativity with AI-Powered Image Generation

This powerful web interface, built with Gradio, puts the magic of Stable Diffusion at your fingertips, enabling you to create stunning images from text prompts. [Explore the original repo](https://github.com/AUTOMATIC1111/stable-diffusion-webui) for the latest updates and community contributions.

## Key Features:

*   **Text-to-Image and Image-to-Image Generation:** Bring your ideas to life with versatile modes.
*   **One-Click Installation:** Get started quickly with a simple installation script (Python and Git required).
*   **Advanced Editing Tools:**
    *   Outpainting and Inpainting for creative image manipulation.
    *   Color Sketch to generate images from sketches.
    *   Prompt Matrix, allowing you to experiment with various prompts at once.
    *   Attention mechanism, fine-tune the model's focus.
    *   Negative prompt, control what you *don't* want to see.
    *   Prompt Editing, change prompts mid-generation.
    *   Styles and Variations, save and apply prompts to iterate faster.
*   **Upscaling and Enhancement:**
    *   Stable Diffusion Upscale to enhance resolution.
    *   GFPGAN, CodeFormer, and other face restoration tools.
    *   Multiple neural network upscalers (RealESRGAN, ESRGAN, SwinIR, Swin2SR, LDSR).
*   **Flexible Generation:**
    *   Adjustable aspect ratio and sampling methods.
    *   Interrupt processing at any time.
    *   Live preview and progress bar.
    *   Generation parameters saved with images.
    *   Seed resizing for subtle variations.
*   **Advanced Techniques:**
    *   Textual Inversion, train custom embeddings.
    *   X/Y/Z plot for parameter exploration.
    *   CLIP interrogator to generate prompts from images.
    *   Batch Processing for efficient image creation.
    *   Composable-Diffusion, create images with multiple prompts simultaneously.
    *   DeepDanbooru integration for anime prompts.
*   **Community & Extensibility:**
    *   Extensive custom scripts support.
    *   API for integration.
    *   Lorals, Hypernetworks and Embedding support.
*   **Performance & Compatibility:**
    *   4GB video card support (and even lower).
    *   Checkpoint Merger.
    *   Support for Stable Diffusion 2.0 and Alt-Diffusion.
    *   Support for Segmind Stable Diffusion.
    *   And much more!

## Installation

Detailed installation instructions can be found in the [wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki).
Choose the appropriate guide for your operating system:

*   **NVidia GPUs (Recommended)**
*   **AMD GPUs**
*   **Intel CPUs and GPUs**
*   **Ascend NPUs**
*   **Apple Silicon**
*   **Online Services**

**Quick Start for Windows with NVidia GPUs:**

1.  Download `sd.webui.zip` from the [releases](https://github.com/AUTOMATIC1111/stable-diffusion-webui/releases).
2.  Run `update.bat`.
3.  Run `run.bat`.

**Automatic Installation (Windows):**

1.  Install [Python 3.10.6](https://www.python.org/downloads/release/python-3106/) (and add to PATH).
2.  Install [Git](https://git-scm.com/download/win).
3.  Clone the repository: `git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git`.
4.  Run `webui-user.bat`.

**Automatic Installation (Linux):**

1.  Install dependencies (example for Debian-based systems: `sudo apt install wget git python3 python3-venv libgl1 libglib2.0-0`).
2.  Navigate to your desired installation directory and run: `wget -q https://raw.githubusercontent.com/AUTOMATIC1111/stable-diffusion-webui/master/webui.sh`.
3.  Run `./webui.sh`.

## Documentation

For comprehensive information, tutorials, and troubleshooting, consult the project's [wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki).

## Contributing

Contribute to the project by following the guidelines [here](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Contributing).

## Credits

Licenses and credits for borrowed code can be found in `Settings -> Licenses` or in the `html/licenses.html` file.