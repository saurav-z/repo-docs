# Stable Diffusion Web UI: Unleash Your Creativity with AI-Powered Image Generation

**Create stunning images from text prompts with the Stable Diffusion Web UI, a user-friendly interface built on the Gradio library.** ([View on GitHub](https://github.com/AUTOMATIC1111/stable-diffusion-webui))

## Key Features:

*   **Versatile Generation Modes:** Utilize txt2img, img2img, inpainting, outpainting, and color sketch to explore diverse image creation possibilities.
*   **One-Click Installation:** Easily get started with a simple install and run script (Python and Git required).
*   **Advanced Prompting:** Refine image generation using attention mechanisms, negative prompts, styles, and prompt editing.
*   **Upscaling & Enhancement:** Improve image quality with GFPGAN, CodeFormer, RealESRGAN, ESRGAN, and other powerful upscalers.
*   **Textual Inversion & Training:** Train custom embeddings, hypernetworks, and LoRAs to personalize your creations.
*   **Interactive Features:** Utilize progress bars, live previews, seed resizing, and a CLIP interrogator for a dynamic user experience.
*   **Batch Processing & Automation:** Process multiple images at once and automate repetitive tasks.
*   **Extensive Extensions:** Leverage a vast community of custom scripts and extensions for extended functionality, including Aesthetic Gradients and Composable Diffusion.
*   **Hardware Support:** Compatible with a range of hardware, including NVIDIA, AMD, and Intel GPUs (with 4GB+ VRAM recommended).
*   **API Integration:** Utilize the integrated API to integrate Stable Diffusion Web UI with other applications.
*   **Multi-Model Support:** Supports Stable Diffusion 2.0, Alt-Diffusion, and Segmind Stable Diffusion models.

## Installation and Running

Choose the installation method that best suits your system. Detailed instructions are available for:

*   [NVidia GPUs](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-NVidia-GPUs) (Recommended)
*   [AMD GPUs](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-AMD-GPUs)
*   [Intel CPUs and GPUs](https://github.com/openvinotoolkit/stable-diffusion-webui/wiki/Installation-on-Intel-Silicon) (External Wiki)
*   [Ascend NPUs](https://github.com/wangshuai09/stable-diffusion-webui/wiki/Install-and-run-on-Ascend-NPUs) (External Wiki)
*   [Apple Silicon](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Installation-on-Apple-Silicon)

### Quick Start - Windows (NVidia)

1.  Download the `sd.webui.zip` from the [latest release](https://github.com/AUTOMATIC1111/stable-diffusion-webui/releases).
2.  Run `update.bat`.
3.  Run `run.bat`.

### Automatic Installation (Windows):

1.  Install [Python 3.10.6](https://www.python.org/downloads/release/python-3106/) and Git, ensuring Python is added to your PATH.
2.  Clone the repository: `git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git`.
3.  Run `webui-user.bat`.

### Automatic Installation (Linux):

1.  Install dependencies using your distribution's package manager (examples provided in the original README).
2.  Navigate to your desired install directory and run `wget -q https://raw.githubusercontent.com/AUTOMATIC1111/stable-diffusion-webui/master/webui.sh` or clone the repo with git.
3.  Run `./webui.sh` or `./webui.sh`.

## Contributing

Contribute to the project through the [Contributing Guidelines](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Contributing).

## Documentation

The comprehensive documentation is located on the project's [wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki).

## Credits

This project leverages many open-source resources.  See `Settings -> Licenses` within the Web UI and the `html/licenses.html` file for a complete list of credits.