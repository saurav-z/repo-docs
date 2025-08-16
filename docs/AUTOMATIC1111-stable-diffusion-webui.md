# Stable Diffusion Web UI: Unleash Your Creativity with AI-Powered Image Generation

This web interface for Stable Diffusion, built with Gradio, empowers you to generate stunning images from text prompts with ease.  Explore the original [Stable Diffusion Web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) for unparalleled image generation control.

## Key Features:

*   **Versatile Image Generation:** Leverage both txt2img and img2img modes.
*   **One-Click Setup:** Simplifies installation with pre-built scripts (requires Python and Git).
*   **Advanced Editing:** Includes outpainting, inpainting, color sketching, and prompt editing.
*   **Image Enhancement:** Integrate tools like GFPGAN, CodeFormer, and RealESRGAN for superior image quality.
*   **Flexible Upscaling:** Utilize SD upscale, ESRGAN, SwinIR, and LDSR for detailed results.
*   **Fine-Grained Control:**  Use attention mechanisms, negative prompts, and styles for precision.
*   **Interactive Generation:** Includes live previews, progress bars, and the ability to interrupt processing.
*   **Extensive Functionality:** Supports prompt matrices, loopback, X/Y/Z plots, textual inversion, and more.
*   **Advanced Features:** Includes support for variations, seed resizing, CLIP interrogator, batch processing, and high-res fixes.
*   **Community Driven:** Benefit from custom scripts and extensions, plus Composable-Diffusion integration.
*   **Training & Integration:** Training tabs, LoRAs, and a dedicated UI for selecting embeddings and models.
*   **API Support:** Integrate the web UI's functionalities into other projects via an API.
*   **Model Support:** Supports for dedicated inpainting models by RunwayML, Stable Diffusion 2.0, Alt-Diffusion, and Segmind Stable Diffusion.
*   **Customization**: Ability to reorder UI elements, change defaults, and apply aesthetic gradients.

## Installation & Running

Detailed installation instructions are available for:

*   [NVidia GPUs](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-NVidia-GPUs) (recommended)
*   [AMD GPUs](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-AMD-GPUs)
*   [Intel CPUs/GPUs](https://github.com/openvinotoolkit/stable-diffusion-webui/wiki/Installation-on-Intel-Silicon)
*   [Ascend NPUs](https://github.com/wangshuai09/stable-diffusion-webui/wiki/Install-and-run-on-Ascend-NPUs)
*   [Apple Silicon](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Installation-on-Apple-Silicon)

**Quick Start for Windows with NVidia GPUs:**

1.  Download `sd.webui.zip` from [v1.0.0-pre](https://github.com/AUTOMATIC1111/stable-diffusion-webui/releases/tag/v1.0.0-pre) and extract.
2.  Run `update.bat`.
3.  Run `run.bat`.

**Automatic Installation (Windows):**

1.  Install [Python 3.10.6](https://www.python.org/downloads/release/python-3106/) (check "Add Python to PATH").
2.  Install [git](https://git-scm.com/download/win).
3.  Clone the repository: `git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git`.
4.  Run `webui-user.bat`.

**Automatic Installation (Linux):**

1.  Install dependencies: (Debian/Ubuntu) `sudo apt install wget git python3 python3-venv libgl1 libglib2.0-0`
2.  Navigate to your desired directory and run: `wget -q https://raw.githubusercontent.com/AUTOMATIC1111/stable-diffusion-webui/master/webui.sh` or clone the repo using `git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui`.
3.  Run `webui.sh`.
4.  Check `webui-user.sh` for configuration options.

## Contributing

Contribute to the project: [Contributing](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Contributing)

## Documentation

Comprehensive documentation is available on the project's [wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki).

## Credits

A list of credits and licenses for borrowed code is available in `Settings -> Licenses` and `html/licenses.html`.