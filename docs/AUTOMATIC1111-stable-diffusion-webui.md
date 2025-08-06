# Stable Diffusion Web UI: Unleash Your Creativity with AI-Powered Image Generation

Create stunning images from text prompts with the Stable Diffusion web UI, a user-friendly interface built on the Gradio library.  [Explore the original repository here](https://github.com/AUTOMATIC1111/stable-diffusion-webui).

## Key Features

*   **Text-to-Image & Image-to-Image Generation:** Transform text prompts into captivating visuals or modify existing images.
*   **One-Click Installation:**  Get up and running quickly (Python and Git required).
*   **Advanced Image Editing:** Utilize outpainting, inpainting, color sketches, and prompt matrices.
*   **Attention Mechanism:** Fine-tune image generation by emphasizing specific text elements using attention.
*   **Flexible Upscaling:** Improve image resolution with Stable Diffusion Upscale and various neural network upscalers (GFPGAN, CodeFormer, RealESRGAN, ESRGAN, SwinIR, Swin2SR, LDSR).
*   **Textual Inversion & LoRAs:** Train and integrate custom embeddings to personalize your image generation.
*   **Comprehensive Tools:**  Includes a CLIP interrogator, prompt editing, batch processing, and seed control.
*   **Generation Parameters:**  Save and easily restore image generation settings.
*   **Community-Driven:** Extensive support for custom scripts and extensions.
*   **Hardware Support:** Supports a range of hardware including 4GB video cards and 2GB with some limitations.

## Installation and Usage

Detailed installation instructions are available for:

*   [NVidia GPUs](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-NVidia-GPUs) (Recommended)
*   [AMD GPUs](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-AMD-GPUs)
*   [Intel CPUs and GPUs](https://github.com/openvinotoolkit/stable-diffusion-webui/wiki/Installation-on-Intel-Silicon)
*   [Ascend NPUs](https://github.com/wangshuai09/stable-diffusion-webui/wiki/Install-and-run-on-Ascend-NPUs)
*   [Apple Silicon](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Installation-on-Apple-Silicon)

**Quick Start for Windows with NVidia GPUs:**

1.  Download `sd.webui.zip` from [v1.0.0-pre](https://github.com/AUTOMATIC1111/stable-diffusion-webui/releases/tag/v1.0.0-pre).
2.  Extract the contents.
3.  Run `update.bat`.
4.  Run `run.bat`.

**Automatic Installation (Windows):**

1.  Install Python 3.10.6, ensuring it's added to your PATH.
2.  Install Git.
3.  Clone the repository: `git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git`.
4.  Run `webui-user.bat`.

**Automatic Installation (Linux):**

1.  Install dependencies (example for Debian-based systems: `sudo apt install wget git python3 python3-venv libgl1 libglib2.0-0`).
2.  Clone or download the repository.
3.  Run `webui.sh`.

## Contributing

Learn how to contribute to the project at [Contributing](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Contributing).

## Documentation

Comprehensive documentation is available on the project [wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki).

## Credits

This project incorporates code and ideas from numerous contributors; full credit listings can be found in the `Settings -> Licenses` screen and `html/licenses.html` file.