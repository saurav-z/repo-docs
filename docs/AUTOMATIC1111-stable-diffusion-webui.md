# Stable Diffusion Web UI: Unleash Your Creativity with AI-Powered Image Generation

**Create stunning images from text prompts with the Stable Diffusion web UI, a user-friendly interface built with Gradio.** ([Original Repository](https://github.com/AUTOMATIC1111/stable-diffusion-webui))

[![](screenshot.png)]

## Key Features:

*   **txt2img and img2img Modes:** Generate images from text prompts or modify existing images.
*   **One-Click Installation:** Simplify setup with a convenient install script (Python and Git required).
*   **Advanced Image Editing:** Utilize outpainting, inpainting, color sketches, and prompt editing.
*   **Prompt Optimization:** Refine image generation using attention mechanisms, negative prompts, and styles.
*   **Upscaling & Enhancement:** Improve image quality with integrated GFPGAN, CodeFormer, RealESRGAN, and other upscalers.
*   **Batch Processing & Automation:** Process multiple images at once and automate workflows with scripts and extensions.
*   **Extensive Customization:** Fine-tune image generation with sampling methods, seed control, and generation parameters.
*   **Model & Extension Support:**  Integrate embeddings, hypernetworks, LoRAs, and a wide range of community-created extensions.
*   **API Access:** Integrate the web UI into your own applications via API.
*   **Cross-Platform Compatibility:** Run on NVIDIA, AMD, and Intel GPUs, with options for online services.

## Getting Started:

### Installation

Detailed installation instructions can be found in the [wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki)

You can find options for:
*   **NVidia GPUs**
*   **AMD GPUs**
*   **Intel CPUs and GPUs**
*   **Ascend NPUs**
*   **Online Services**

### Quick Installation on Windows (NVidia GPUs):

1.  Download the `sd.webui.zip` release package.
2.  Run `update.bat`.
3.  Run `run.bat`.

### Automatic Installation on Windows:

1.  Install Python 3.10.6 and Git.
2.  Clone the repository: `git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git`
3.  Run `webui-user.bat`.

### Automatic Installation on Linux:

1.  Install dependencies (example Debian-based: `sudo apt install wget git python3 python3-venv libgl1 libglib2.0-0`).
2.  Navigate to your desired directory and execute:
    `wget -q https://raw.githubusercontent.com/AUTOMATIC1111/stable-diffusion-webui/master/webui.sh`
    or clone the repo
    `git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui`
3.  Run `webui.sh`.
4.  Check `webui-user.sh` for options.

### Installation on Apple Silicon

Find the instructions [here](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Installation-on-Apple-Silicon).

## Contributing

Contribute to the project by following the guidelines in the [Contributing](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Contributing) section of the wiki.

## Documentation

Explore the extensive documentation on the project's [wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki) for detailed feature explanations, usage guides, and troubleshooting tips.

## Credits

This project is built upon the work of many contributors and utilizes code from various open-source projects.  See the [Credits](https://github.com/AUTOMATIC1111/stable-diffusion-webui) section in the original repository for a full list of acknowledgements.