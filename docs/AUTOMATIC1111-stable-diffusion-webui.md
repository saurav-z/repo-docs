# Stable Diffusion Web UI: Unleash Your Creativity with AI Art

**Create stunning images from text prompts with the Stable Diffusion web UI, a user-friendly interface built with Gradio.** [(Original Repository)](https://github.com/AUTOMATIC1111/stable-diffusion-webui)

## Key Features:

*   **Versatile Generation Modes:** Explore text-to-image (txt2img) and image-to-image (img2img) generation.
*   **One-Click Setup:** Simplify installation with a convenient script (Python and Git required).
*   **Advanced Image Editing:** Utilize outpainting, inpainting, and color sketch features.
*   **Prompt Engineering Tools:** Leverage prompt matrix, attention mechanisms, and negative prompts for precise control.
*   **Upscaling and Enhancement:** Improve image quality with Stable Diffusion Upscale and other integrated tools like GFPGAN, CodeFormer, RealESRGAN, and more.
*   **Iterative Refinement:** Use loopback and variations for creating unique images.
*   **Batch Processing:** Process multiple images at once using img2img.
*   **Seed Control:** Utilize correct seeds for batches and seed resizing to generate similar images with slight variations.
*   **Extensive Customization:** Benefit from textual inversion, custom scripts, and a settings page for personalized control.
*   **Seamless Integration:** Drag and drop images to generate parameters or load them in the promptbox.
*   **Community Driven**: Benefit from the custom scripts, extensions and integrations with the community.

## Installation and Running

### Prerequisites
Ensure you have the required dependencies before proceeding. More information can be found here: [Dependencies](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Dependencies)

### Instructions

Follow the installation instructions for your operating system:

*   **Nvidia GPUs:** Detailed instructions are available in the [wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-NVidia-GPUs). (Recommended)
*   **AMD GPUs:** [Installation on AMD GPUs](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-AMD-GPUs).
*   **Intel CPUs/GPUs:** [Installation on Intel Silicon](https://github.com/openvinotoolkit/stable-diffusion-webui/wiki/Installation-on-Intel-Silicon)
*   **Ascend NPUs:** [Install and run on Ascend NPUs](https://github.com/wangshuai09/stable-diffusion-webui/wiki/Install-and-run-on-Ascend-NPUs)

**Windows (Simplified)**

1.  Download the `sd.webui.zip` from the [releases](https://github.com/AUTOMATIC1111/stable-diffusion-webui/releases)
2.  Run `update.bat`.
3.  Run `run.bat`.

**Automatic Installation on Windows**

1.  Install [Python 3.10.6](https://www.python.org/downloads/release/python-3106/) (Newer version of Python does not support torch), checking "Add Python to PATH".
2.  Install [git](https://git-scm.com/download/win).
3.  Download the stable-diffusion-webui repository, for example by running `git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git`.
4.  Run `webui-user.bat` from Windows Explorer as normal, non-administrator, user.

**Automatic Installation on Linux**

1.  Install dependencies. Follow these commands based on your system:
    *   Debian-based: `sudo apt install wget git python3 python3-venv libgl1 libglib2.0-0`
    *   Red Hat-based: `sudo dnf install wget git python3 gperftools-libs libglvnd-glx`
    *   openSUSE-based: `sudo zypper install wget git python3 libtcmalloc4 libglvnd`
    *   Arch-based: `sudo pacman -S wget git python3`

2.  Navigate to your desired installation directory and execute:
    `wget -q https://raw.githubusercontent.com/AUTOMATIC1111/stable-diffusion-webui/master/webui.sh`
    OR
    `git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui`

3.  Run `webui.sh`.
4.  Customize with `webui-user.sh`

**Apple Silicon**

Find instructions [here](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Installation-on-Apple-Silicon).

## Contributing

Learn how to contribute to the project here: [Contributing](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Contributing).

## Credits

*   [See Original README for a complete list of credits](https://github.com/AUTOMATIC1111/stable-diffusion-webui)