# Stable Diffusion Web UI: Unleash Your Creativity with AI Image Generation

**Generate stunning images from text prompts with the Stable Diffusion web UI, a powerful and user-friendly interface built with Gradio.** ([Original Repo](https://github.com/AUTOMATIC1111/stable-diffusion-webui))

## Key Features:

*   **Versatile Image Generation Modes:** Utilize original txt2img and img2img modes for diverse image creation.
*   **One-Click Installation:** Simplify setup with an easy-to-use install script (requires Python and Git).
*   **Advanced Image Editing:** Includes outpainting, inpainting, and color sketch features for precise control.
*   **Prompt Refinement:** Fine-tune image generation with attention mechanisms, negative prompts, and prompt editing.
*   **Upscaling & Enhancement:** Integrate GFPGAN, CodeFormer, RealESRGAN, and other upscalers for superior image quality.
*   **Customization & Flexibility:** Offers settings pages, script execution, and numerous extensions from the community.
*   **High-Resolution Support:** Generate high-resolution images efficiently with Highres Fix and advanced sampling options.
*   **Batch Processing:** Streamline workflow with batch processing of images using img2img.
*   **Community Support:** Training tabs for hypernetworks and embeddings, the UI is supported by numerous extensions.
*   **Model Support:** Includes Segmind Stable Diffusion and support for Stable Diffusion 2.0 and Alt-Diffusion.

## Installation and Usage

Follow the installation instructions for your operating system:

*   **[NVidia](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-NVidia-GPUs)** (Recommended)
*   **[AMD](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-AMD-GPUs)**
*   **[Intel CPUs, Intel GPUs](https://github.com/openvinotoolkit/stable-diffusion-webui/wiki/Installation-on-Intel-Silicon)**
*   **[Ascend NPUs](https://github.com/wangshuai09/stable-diffusion-webui/wiki/Install-and-run-on-Ascend-NPUs)**
*   **[Apple Silicon](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Installation-on-Apple-Silicon)**

**Windows (NVidia GPU) Quick Start:**

1.  Download `sd.webui.zip` from [v1.0.0-pre](https://github.com/AUTOMATIC1111/stable-diffusion-webui/releases/tag/v1.0.0-pre).
2.  Extract the contents.
3.  Run `update.bat`.
4.  Run `run.bat`.

**Automatic Installation on Windows:**

1.  Install [Python 3.10.6](https://www.python.org/downloads/release/python-3106/) (and add it to your PATH).
2.  Install [git](https://git-scm.com/download/win).
3.  Clone the repository using `git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git`.
4.  Run `webui-user.bat`.

**Automatic Installation on Linux:**

1.  Install dependencies (example for Debian-based systems: `sudo apt install wget git python3 python3-venv libgl1 libglib2.0-0`).
2.  Install python3.10 or python3.11
3.  Navigate to the desired installation directory.
4.  Download and run the installation script:
    ```bash
    wget -q https://raw.githubusercontent.com/AUTOMATIC1111/stable-diffusion-webui/master/webui.sh
    ./webui.sh
    ```
    Or just clone the repo wherever you want:
    ```bash
    git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui
    ```

## Contributing

Contribute to the project by following the guidelines in the [Contributing](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Contributing) section of the wiki.

## Documentation

Comprehensive documentation and tutorials are available on the project's [wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki).

## Credits

A full list of credits and licenses for borrowed code can be found in the `Settings -> Licenses` screen and `html/licenses.html` file.