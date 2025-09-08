# Stable Diffusion Web UI: Unleash Your Creativity with AI-Powered Image Generation

**Create stunning images from text prompts using Stable Diffusion, all within an easy-to-use web interface.** ([Original Repository](https://github.com/AUTOMATIC1111/stable-diffusion-webui))

## Key Features:

*   **Intuitive Interface:** Enjoy a user-friendly web UI built with Gradio, making image generation accessible for everyone.
*   **Versatile Generation Modes:** Explore diverse image creation options with txt2img, img2img, inpainting, outpainting, and more.
*   **Advanced Prompting:** Fine-tune your creations with features like attention mechanisms, negative prompts, styles, and prompt editing.
*   **Upscaling & Enhancement:** Improve image quality with built-in tools like GFPGAN, CodeFormer, RealESRGAN, ESRGAN, and others.
*   **Extensive Customization:** Control your generation with sampling method selection, seed resizing, and X/Y/Z plot for exploring different parameters.
*   **Training & Customization:** Train custom embeddings, hypernetworks, and LoRAs to personalize the model.
*   **Community-Driven:** Benefit from numerous extensions, custom scripts, and community support, including support for popular models like Stable Diffusion 2.0 and Alt-Diffusion.
*   **Seamless Workflow:** Save and load generation parameters, integrate with image browsers, and leverage features like batch processing for efficient workflows.
*   **Optimized Performance:** Experience faster generation speeds with xformers support and various optimization techniques.
*   **API:** Access the functionality through an API.

## Installation

Detailed installation instructions are available on the project's wiki, covering various platforms:

*   **NVidia GPUs (Recommended):** [Installation Guide for NVidia](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-NVidia-GPUs)
*   **AMD GPUs:** [Installation Guide for AMD GPUs](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-AMD-GPUs)
*   **Intel CPUs/GPUs:** [Installation Guide for Intel Silicon](https://github.com/openvinotoolkit/stable-diffusion-webui/wiki/Installation-on-Intel-Silicon)
*   **Ascend NPUs:** [Installation Guide for Ascend NPUs](https://github.com/wangshuai09/stable-diffusion-webui/wiki/Install-and-run-on-Ascend-NPUs)
*   **Apple Silicon:** [Installation Guide for Apple Silicon](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Installation-on-Apple-Silicon)

You can also utilize online services such as Google Colab.  See the [List of Online Services](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Online-Services).

### Quick Install for Windows with NVidia GPUs (using release package):

1.  Download `sd.webui.zip` from [v1.0.0-pre](https://github.com/AUTOMATIC1111/stable-diffusion-webui/releases/tag/v1.0.0-pre).
2.  Extract the contents.
3.  Run `update.bat`.
4.  Run `run.bat`.

### Automatic Installation on Windows:

1.  Install [Python 3.10.6](https://www.python.org/downloads/release/python-3106/) (check "Add Python to PATH").
2.  Install [git](https://git-scm.com/download/win).
3.  Clone the repository: `git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git`.
4.  Run `webui-user.bat`.

### Automatic Installation on Linux:

1.  Install dependencies (examples below; adapt to your distribution):

```bash
# Debian-based:
sudo apt install wget git python3 python3-venv libgl1 libglib2.0-0
# Red Hat-based:
sudo dnf install wget git python3 gperftools-libs libglvnd-glx
# openSUSE-based:
sudo zypper install wget git python3 libtcmalloc4 libglvnd
# Arch-based:
sudo pacman -S wget git python3
```

    If needed, install Python 3.10 or 3.11 as described in the original readme.

2.  Navigate to your desired installation directory.
3.  Download the installation script:

```bash
wget -q https://raw.githubusercontent.com/AUTOMATIC1111/stable-diffusion-webui/master/webui.sh
```
    Or clone the repository:
```bash
git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui
```
4. Run `webui.sh`.
5. Customize by editing `webui-user.sh`.

## Contributing

Contribute to the project by following the [Contributing](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Contributing) guidelines.

## Documentation

Comprehensive documentation is available on the project's [wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki).

## Credits

The project utilizes code from various sources.  See `Settings -> Licenses` and `html/licenses.html` for license details.  Credits are also listed in the original README.