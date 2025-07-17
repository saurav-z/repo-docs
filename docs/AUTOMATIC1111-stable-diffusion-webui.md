# Stable Diffusion Web UI: Unleash Your Creativity with AI-Powered Image Generation

Create stunning visuals with the Stable Diffusion web UI, a user-friendly interface built with Gradio that empowers you to generate images from text prompts. [Visit the original repository for more details.](https://github.com/AUTOMATIC1111/stable-diffusion-webui)

## Key Features

*   **Intuitive Interface:** Enjoy a streamlined and easy-to-use web interface for generating images.
*   **txt2img and img2img Modes:** Generate images from text prompts (txt2img) or modify existing images (img2img).
*   **One-Click Installation:** Quickly set up the web UI with a simple install script (Python and Git required).
*   **Advanced Image Manipulation:** Explore features like outpainting, inpainting, color sketches, and prompt editing for detailed control.
*   **AI-Powered Upscaling and Enhancement:** Utilize integrated tools like GFPGAN, CodeFormer, RealESRGAN, ESRGAN, SwinIR, Swin2SR, and LDSR for superior image quality.
*   **Flexible Prompting:**
    *   **Attention:** Fine-tune the model's focus with attention weighting (e.g., `((tuxedo))` or `(tuxedo:1.21)`).
    *   **Negative Prompts:** Specify what you *don't* want to see in your images.
    *   **Styles:** Save and apply prompt snippets for consistent results.
    *   **Prompt Matrix:** Generate images with various combinations of parameters.
    *   **Composable Diffusion:** Combine multiple prompts using uppercase `AND` with optional weights.
*   **Workflow Enhancements:**
    *   **Batch Processing:** Process multiple images at once.
    *   **Seed Control:** Correct seeds for batches and seed resizing for subtle variations.
    *   **Generation Parameters:** Save and restore image generation parameters (PNG chunks/EXIF).
    *   **CLIP Interrogator:** Analyze images and generate corresponding prompts.
    *   **Checkpoint Merger:** Combine up to 3 checkpoints into one.
*   **Customization and Extensibility:**
    *   **Custom Scripts:** Extend functionality with community-developed scripts.
    *   **Extensions:** Access a growing library of extensions for added features.
    *   **Settings Page:** Customize defaults and UI elements.
*   **Hardware Support:** Optimized for both low and high VRAM.

## Installation and Running

**Note:** Ensure you meet the [dependencies](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Dependencies).

Choose your operating system and follow the relevant installation instructions.

*   [NVidia GPUs (Recommended)](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-NVidia-GPUs)
*   [AMD GPUs](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-AMD-GPUs)
*   [Intel CPUs/GPUs](https://github.com/openvinotoolkit/stable-diffusion-webui/wiki/Installation-on-Intel-Silicon)
*   [Ascend NPUs](https://github.com/wangshuai09/stable-diffusion-webui/wiki/Install-and-run-on-Ascend-NPUs)
*   [Apple Silicon](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Installation-on-Apple-Silicon)

**Windows Installation (NVidia GPUs - release package):**

1.  Download `sd.webui.zip` from [releases](https://github.com/AUTOMATIC1111/stable-diffusion-webui/releases).
2.  Extract the contents.
3.  Run `update.bat`.
4.  Run `run.bat`.

**Automatic Installation on Windows:**

1.  Install Python 3.10.6 (add to PATH) and Git.
2.  Clone the repository: `git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git`.
3.  Run `webui-user.bat`.

**Automatic Installation on Linux:**

1.  Install Dependencies (example for Debian-based systems): `sudo apt install wget git python3 python3-venv libgl1 libglib2.0-0`
2.  Navigate to your desired installation directory.
3.  Download and run the installation script: `wget -q https://raw.githubusercontent.com/AUTOMATIC1111/stable-diffusion-webui/master/webui.sh && bash webui.sh` or clone the repo with `git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui`
4.  Run the webui.sh script to launch.

## Documentation and Contributing

*   **Documentation:** Comprehensive documentation is available on the project's [wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki).
*   **Contributing:** Learn how to contribute to the project by visiting the [Contributing](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Contributing) section.

## Credits

This project leverages contributions from a wide range of individuals and organizations.  See the full list of credits, including licenses, in the `Settings -> Licenses` screen, and `html/licenses.html` file, and the original README.