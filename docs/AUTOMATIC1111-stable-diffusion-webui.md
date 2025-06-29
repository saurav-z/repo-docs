# Stable Diffusion WebUI: Unleash Your Creativity with AI-Powered Image Generation

This powerful web interface, built with Gradio, allows you to generate stunning images from text prompts using Stable Diffusion, the cutting-edge text-to-image AI model. [Explore the original repository](https://github.com/AUTOMATIC1111/stable-diffusion-webui) for more details.

## Key Features

*   **Versatile Image Generation:** Utilize original `txt2img` and `img2img` modes for diverse image creation.
*   **One-Click Installation:** Easily set up and run the web UI with a convenient script (Python and Git required).
*   **Advanced Editing & Control:** Features like outpainting, inpainting, color sketches, prompt matrix, attention mechanisms, loopback processing, and X/Y/Z plots provide extensive creative control.
*   **Textual Inversion:** Train and utilize custom embeddings for unique image styles and concepts.
*   **AI-Powered Enhancements:** Integrate tools like GFPGAN, CodeFormer, RealESRGAN, ESRGAN, SwinIR, Swin2SR, and LDSR for face restoration and image upscaling.
*   **Flexible Parameters:** Fine-tune generation with aspect ratio options, sampling method selection (with adjustable eta values and noise settings), and the ability to interrupt processing.
*   **Seamless Integration:** Save and restore generation parameters with images, drag-and-drop functionality for prompts and images, and a "Read Generation Parameters" button.
*   **User-Friendly Interface:** Benefit from a settings page, mouseover hints, UI element customization, tiling support, a progress bar with live preview, negative prompts, and style saving.
*   **Creative Freedom:** Leverage variations, seed resizing, CLIP interrogator, prompt editing, batch processing, high-resolution fixes, and checkpoint merging.
*   **Community-Driven Extensions:** Explore a rich ecosystem of custom scripts and extensions for enhanced functionality, including Composable Diffusion, DeepDanbooru integration, and xformers optimization.
*   **Additional Enhancements:** Features include no token limits for prompts, a training tab, hypernetworks, LoRAs, a dedicated UI for managing embeddings, hypernetworks, and LoRAs, and an API.
*   **Multi Model Support:** Compatible with Stable Diffusion 2.0, Alt-Diffusion, and Segmind Stable Diffusion.
*   **Improved Resolution:** Generate images with dimensions as multiples of 8, overcoming previous size restrictions.
*   **Support for specialized inpainting models**: Enhance detail within images

## Installation and Running

Detailed instructions and links to installation guides are available for:

*   NVidia GPUs (Recommended)
*   AMD GPUs
*   Intel CPUs and GPUs
*   Ascend NPUs

You can also utilize online services such as Google Colab.

**For Windows 10/11 with NVidia GPUs (using release package):**

1.  Download `sd.webui.zip` from [v1.0.0-pre](https://github.com/AUTOMATIC1111/stable-diffusion-webui/releases/tag/v1.0.0-pre) and extract its contents.
2.  Run `update.bat`.
3.  Run `run.bat`.

**Automatic Installation on Windows:**

1.  Install Python 3.10.6 (and add it to your PATH).
2.  Install Git.
3.  Clone the repository using `git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git`.
4.  Run `webui-user.bat`.

**Automatic Installation on Linux:**

1.  Install dependencies (using commands provided in the original README - see above).
2.  Navigate to your desired installation directory.
3.  Run `wget -q https://raw.githubusercontent.com/AUTOMATIC1111/stable-diffusion-webui/master/webui.sh` or clone the repo directly.
4.  Run `webui.sh`.
5.  Consult `webui-user.sh` for configuration options.

**Installation on Apple Silicon:**

Refer to the installation instructions [here](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Installation-on-Apple-Silicon).

## Contributing

Contribute to the project by following the guidelines in the [Contributing](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Contributing) section.

## Documentation

Comprehensive documentation is available on the project's [wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki).

## Credits

(See the original README for a complete list of credits).