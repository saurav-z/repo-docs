# Stable Diffusion WebUI: Your Gateway to AI-Powered Image Generation

**Unleash your creativity with Stable Diffusion WebUI, a powerful and user-friendly web interface for generating stunning images from text prompts.** ([Back to Original Repo](https://github.com/AUTOMATIC1111/stable-diffusion-webui))

## Key Features

*   **Versatile Generation Modes:** Seamlessly create images with `txt2img` and `img2img` modes, including outpainting, inpainting, and color sketch options.
*   **One-Click Setup:** Simplify your workflow with a convenient one-click install and run script (Python and Git required).
*   **Advanced Prompting:** Utilize attention mechanisms, negative prompts, styles, and variations for fine-tuned image generation.
*   **Powerful Upscaling & Enhancement:** Integrate GFPGAN, CodeFormer, RealESRGAN, ESRGAN, SwinIR, Swin2SR, and LDSR for enhanced image quality.
*   **Flexible Control & Customization:** Explore features like X/Y/Z plots, textual inversion, prompt editing, and seed resizing for unparalleled control.
*   **Optimized Performance:** Leverage xformers for significant speed improvements on select GPUs.
*   **Community-Driven Ecosystem:** Benefit from a thriving community with custom scripts, extensions, and support for cutting-edge models like Stable Diffusion 2.0 and Alt-Diffusion.
*   **Comprehensive Training Capabilities**: Includes training tab options for hypernetworks, embeddings, Loras, and more.
*   **User-Friendly Interface**: Offering a separate UI to add, preview, and manage embeddings, hypernetworks, and Loras.

## Installation and Running

Detailed installation instructions are available for various setups:

*   **NVIDIA GPUs:** ([Wiki Link](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-NVidia-GPUs)) (Recommended)
*   **AMD GPUs:** ([Wiki Link](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-AMD-GPUs))
*   **Intel CPUs/GPUs:** ([External Wiki Link](https://github.com/openvinotoolkit/stable-diffusion-webui/wiki/Installation-on-Intel-Silicon))
*   **Ascend NPUs:** ([External Wiki Link](https://github.com/wangshuai09/stable-diffusion-webui/wiki/Install-and-run-on-Ascend-NPUs))
*   **Apple Silicon:** ([Wiki Link](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Installation-on-Apple-Silicon))

**Quick Start (Windows with NVIDIA GPU):**

1.  Download `sd.webui.zip` from [v1.0.0-pre](https://github.com/AUTOMATIC1111/stable-diffusion-webui/releases/tag/v1.0.0-pre) and extract.
2.  Run `update.bat`.
3.  Run `run.bat`.

**Automatic Installation (Windows):**

1.  Install [Python 3.10.6](https://www.python.org/downloads/release/python-3106/), checking "Add Python to PATH".
2.  Install [git](https://git-scm.com/download/win).
3.  Clone the repository: `git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git`.
4.  Run `webui-user.bat`.

**Automatic Installation (Linux):**

1.  Install dependencies (Debian-based example): `sudo apt install wget git python3 python3-venv libgl1 libglib2.0-0`. (See the README for other distributions)
2.  Navigate to your desired directory.
3.  Run `wget -q https://raw.githubusercontent.com/AUTOMATIC1111/stable-diffusion-webui/master/webui.sh` or clone the repo.
4.  Run `./webui.sh`.

**For other options including online services, please see the [Wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki).**

## Contributing

Contribute to the project by following the guidelines outlined in the [Contributing](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Contributing) section.

## Documentation

Comprehensive documentation is available on the project's [wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki).
(crawlable wiki: https://github-wiki-see.page/m/AUTOMATIC1111/stable-diffusion-webui/wiki)

## Credits

This project incorporates code and ideas from numerous sources. Detailed credits and licenses are available in the `Settings -> Licenses` screen and the `html/licenses.html` file.