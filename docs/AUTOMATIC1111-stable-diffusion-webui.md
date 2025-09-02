# Stable Diffusion Web UI: Unleash Your Creativity with AI-Powered Image Generation

**Create stunning images from text prompts using Stable Diffusion's powerful web interface!** This repository provides a user-friendly web UI built with Gradio, making it easy to generate, edit, and enhance images using the latest advancements in AI image generation. ([Original Repo](https://github.com/AUTOMATIC1111/stable-diffusion-webui))

## Key Features

*   **Versatile Generation Modes:** Utilize txt2img and img2img modes for diverse image creation.
*   **One-Click Installation:** Quickly get started with a simplified install script (Python and Git required).
*   **Advanced Editing & Enhancement:**
    *   **Outpainting & Inpainting:** Extend and refine your images seamlessly.
    *   **Color Sketch & Prompt Matrix:** Explore creative variations and experiment with prompts.
    *   **Upscaling Tools:** Enhance image resolution with Stable Diffusion Upscale, GFPGAN, CodeFormer, RealESRGAN, and more.
*   **Fine-Grained Control:**
    *   **Attention Mechanism:** Refine prompts using `((emphasis))` syntax for specific image details.
    *   **Negative Prompts:** Exclude unwanted elements from your generated images.
    *   **Styles:** Save and easily apply prompt components.
*   **Customization & Flexibility:**
    *   **Textual Inversion, Hypernetworks, and LoRAs:** Train and use custom embeddings for personalized results.
    *   **X/Y/Z Plot:** Create 3D image plots with varying parameters.
    *   **Custom Scripts:** Extend functionality with community-developed scripts.
    *   **Settings Page:** Customize the UI and control defaults.
*   **User-Friendly Interface:**
    *   **Live Preview & Progress Bar:** Monitor generation progress in real-time.
    *   **Generation Parameters:** Save and reuse image generation settings.
    *   **CLIP Interrogator:** Analyze images to generate prompts.
    *   **Drag & Drop:** Simplify image uploads and prompt integration.
*   **Performance & Optimization:**
    *   **4GB+ VRAM Support:** Compatible with a wide range of GPUs.
    *   **Xformers Integration:** Improve performance on select cards (add `--xformers` to commandline args).
*   **Advanced Capabilities:**
    *   **Batch Processing:** Process multiple images simultaneously.
    *   **Composable-Diffusion:** Combine prompts with `AND` and weights.
    *   **DeepDanbooru Integration:** Generate anime-style tags for prompts.
    *   **API Support:** Integrate the web UI with other applications.
    *   **Stable Diffusion 2.0 and Alt-Diffusion Support:** Leverage the latest model advancements.
*   **Community Support:**  Benefit from a vibrant community and numerous extensions.

## Installation

Detailed installation instructions are available in the project's [wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki).  You'll find specific guides for:

*   **NVIDIA GPUs:** (Recommended)
*   **AMD GPUs**
*   **Intel CPUs and GPUs**
*   **Ascend NPUs**
*   **Apple Silicon**

You can also use online services like Google Colab (See [Online Services](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Online-Services)).

### Quick Installation (Windows with NVIDIA GPUs)

1.  Download the release package from the [releases page](https://github.com/AUTOMATIC1111/stable-diffusion-webui/releases).
2.  Run `update.bat`.
3.  Run `run.bat`.

### Automatic Installation (Windows)

1.  Install [Python 3.10.6](https://www.python.org/downloads/release/python-3106/) (and add it to PATH).
2.  Install [Git](https://git-scm.com/download/win).
3.  Clone the repository: `git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git`.
4.  Run `webui-user.bat`.

### Automatic Installation (Linux)

1.  Install dependencies (see instructions in the original README).
2.  Navigate to your desired installation directory.
3.  Run `wget -q https://raw.githubusercontent.com/AUTOMATIC1111/stable-diffusion-webui/master/webui.sh` or clone the repo with git.
4.  Run `webui.sh`.

## Contributing

Learn how to contribute to the project by reviewing the [Contributing guidelines](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Contributing).

## Documentation

Comprehensive documentation is available in the project's [wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki).

## Credits

See the original README for a full list of credits, including links to the projects and contributors.