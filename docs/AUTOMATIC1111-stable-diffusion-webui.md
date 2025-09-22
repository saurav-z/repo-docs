# Stable Diffusion Web UI: Your Gateway to AI-Powered Image Generation

**Unleash your creativity with Stable Diffusion Web UI, a powerful and user-friendly web interface for generating stunning images from text prompts using the Stable Diffusion model.  Access the original repository [here](https://github.com/AUTOMATIC1111/stable-diffusion-webui).**

[![](screenshot.png)](https://github.com/AUTOMATIC1111/stable-diffusion-webui)

## Key Features

*   **Text-to-Image and Image-to-Image Generation:** Create images from text prompts or modify existing images with ease.
*   **One-Click Installation:**  Simple scripts for easy setup (requires Python and Git).
*   **Advanced Editing Tools:** Outpainting, inpainting, color sketch, and prompt matrix for precise control over your creations.
*   **Prompt Refinement:**  Fine-tune your prompts with attention mechanisms, negative prompts, and prompt editing features.
*   **Upscaling & Enhancement:**  Integrates with GFPGAN, CodeFormer, RealESRGAN, ESRGAN, SwinIR, Swin2SR, and LDSR for superior image quality.
*   **Customization Options:**  Explore a wealth of settings, including sampling methods, seed control, tiling support, and generation parameter saving.
*   **Extensive Community Support:** Leverage custom scripts and extensions developed by a vibrant community, including features like aesthetic gradients and history tabs.
*   **Model Integration:**  Supports various models, including Stable Diffusion 2.0, Alt-Diffusion, and Segmind Stable Diffusion, offering diverse generation styles.
*   **Additional Features:** Includes variations, seed resizing, CLIP interrogator, batch processing, and a checkpoint merger for enhanced workflows.
*   **API:** Offers API access for integration with other applications.

## Installation and Running

Detailed installation instructions are available for various platforms:

*   **NVidia GPUs (Recommended):**  Follow the instructions provided in the [NVidia install guide](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-NVidia-GPUs).
*   **AMD GPUs:** Refer to the [AMD GPU installation guide](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-AMD-GPUs).
*   **Intel CPUs/GPUs:**  See the instructions on the [Intel silicon installation guide](https://github.com/openvinotoolkit/stable-diffusion-webui/wiki/Installation-on-Intel-Silicon).
*   **Ascend NPUs:**  Consult the [Ascend NPU installation guide](https://github.com/wangshuai09/stable-diffusion-webui/wiki/Install-and-run-on-Ascend-NPUs).
*   **Apple Silicon:**  Follow the specific instructions for [Apple Silicon](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Installation-on-Apple-Silicon).

**Quick Install (Windows with NVidia GPUs):**

1.  Download the `sd.webui.zip` release package from the [releases page](https://github.com/AUTOMATIC1111/stable-diffusion-webui/releases).
2.  Extract the contents.
3.  Run `update.bat`.
4.  Run `run.bat`.

**Automatic Installation (Windows):**

1.  Install [Python 3.10.6](https://www.python.org/downloads/release/python-3106/) and ensure "Add Python to PATH" is checked.
2.  Install [Git](https://git-scm.com/download/win).
3.  Clone the repository: `git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git`.
4.  Run `webui-user.bat` from the Windows Explorer.

**Automatic Installation (Linux):**

1.  Install dependencies (Debian-based, Red Hat-based, openSUSE-based, and Arch-based instructions are provided in the original README).
2.  Navigate to your desired install directory and execute `webui.sh` or clone the repository with `git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui`.
3.  Run `webui.sh`.
4.  Customize the `webui-user.sh` file for additional options.

## Contributing

Learn how to contribute to the project by reviewing the [contributing guide](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Contributing).

## Documentation

The comprehensive documentation is available on the project's [wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki).
For search engine indexing, see the [crawlable wiki](https://github-wiki-see.page/m/AUTOMATIC1111/stable-diffusion-webui/wiki).

## Credits

This project leverages code from many sources.  Licenses for borrowed code can be found in `Settings -> Licenses` and `html/licenses.html`.  See the original README for a full list of credits.