# Stable Diffusion WebUI: Unleash Your Creativity with AI-Powered Image Generation

**Create stunning images from text prompts using the power of Stable Diffusion with this user-friendly web interface.** ([See the original repo](https://github.com/AUTOMATIC1111/stable-diffusion-webui))

## Key Features:

*   **Versatile Image Generation:** Generate images from text prompts (txt2img) or modify existing images (img2img) with a wide range of options.
*   **One-Click Installation:** Easily install and run the web UI with a convenient script. (Requires Python and Git)
*   **Advanced Features:**
    *   **Outpainting and Inpainting:** Extend and modify images beyond their original boundaries.
    *   **Prompting Enhancements:** Fine-tune image generation with attention mechanisms (e.g., `((tuxedo))`), negative prompts, and style saving.
    *   **Upscaling Tools:** Enhance image resolution using advanced upscaling models like GFPGAN, CodeFormer, RealESRGAN, and more.
    *   **X/Y/Z Plot:** Generate image variations using different parameters.
    *   **Textual Inversion:** Train and use custom embeddings for personalized image generation.
    *   **Custom Scripts and Extensions:** Extend the functionality with community-created scripts and extensions.
*   **User-Friendly Interface:**
    *   **Real-time Preview:** View live image generation progress.
    *   **Generation Parameters:** Save and restore generation parameters for consistent results.
    *   **Interactive Elements:** Utilize mouseover hints, settings pages, and a progress bar.
*   **Performance Optimization:** Support for 4GB+ video cards, and optional xformers for increased speed.
*   **Integration with Cutting-Edge Models:** Includes support for Stable Diffusion 2.0, Alt-Diffusion, and more.
*   **Additional Tools:**
    *   CLIP interrogator.
    *   Checkpoint Merger
    *   Batch Processing
    *   Highres Fix

## Installation and Running

Detailed installation instructions are available for various platforms, including:

*   **Nvidia GPUs (Recommended)**
*   **AMD GPUs**
*   **Intel CPUs and GPUs**
*   **Apple Silicon**
*   **Ascend NPUs**

Instructions are also provided for using online services (e.g., Google Colab).

**For Windows (NVidia GPUs):**

1.  Download `sd.webui.zip` from [v1.0.0-pre](https://github.com/AUTOMATIC1111/stable-diffusion-webui/releases/tag/v1.0.0-pre)
2.  Run `update.bat`.
3.  Run `run.bat`.

**For Automatic Installation on Windows:**

1.  Install Python 3.10.6 and Git.
2.  Clone the repository: `git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git`.
3.  Run `webui-user.bat`.

**For Automatic Installation on Linux:**

1.  Install dependencies (Debian-based: `sudo apt install wget git python3 python3-venv libgl1 libglib2.0-0`).
2.  Navigate to your desired installation directory and run: `wget -q https://raw.githubusercontent.com/AUTOMATIC1111/stable-diffusion-webui/master/webui.sh`. Or clone the repo.
3.  Run `webui.sh`.
4.  Configure options using `webui-user.sh`.

**For installation on Apple Silicon**, please follow instructions at [Installation on Apple Silicon](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Installation-on-Apple-Silicon).

## Contributing

Contribute to the project by following the guidelines: [Contributing](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Contributing)

## Documentation

Refer to the comprehensive documentation available on the project's [wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki).

For search engine indexing of the wiki, see the [crawlable wiki](https://github-wiki-see.page/m/AUTOMATIC1111/stable-diffusion-webui/wiki).

## Credits

A list of credits is in the original README in the Credits section.