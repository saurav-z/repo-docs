# Stable Diffusion Web UI: Unleash Your Creativity with AI Image Generation

**Create stunning images from text prompts with the Stable Diffusion web UI, a powerful and user-friendly interface built on the Gradio library.** (Link back to original repo: [https://github.com/AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui))

## Key Features:

*   **Intuitive Image Generation:** Generate images using the original `txt2img` and `img2img` modes.
*   **One-Click Installation:** Easily set up the web UI with a simple install script (Python and Git required).
*   **Advanced Editing Tools:** Utilize outpainting, inpainting, color sketch, and prompt matrix for versatile image manipulation.
*   **Attention and Prompt Control:** Fine-tune image generation with attention mechanisms (`((tuxedo))`, `(tuxedo:1.21)`) and prompt editing capabilities.
*   **Upscaling & Enhancement:** Improve image quality using Stable Diffusion Upscale and various neural network upscalers (GFPGAN, CodeFormer, RealESRGAN, ESRGAN, SwinIR, Swin2SR, LDSR).
*   **Flexible Parameters:** Adjust sampling methods, eta values, and seed settings for customized results.
*   **Integrated Extensions:** Enhance the UI with community-created extensions (History tab, Aesthetic Gradients).
*   **Batch Processing and Automation:** Process multiple files using `img2img` and generate images with unique variations.
*   **Negative Prompts:** Refine images by specifying elements you *don't* want to see.
*   **Comprehensive Training Tools:** Train hypernetworks, embeddings, and Loras directly within the UI.
*   **Extensive Model Support:** Works with Stable Diffusion 2.0, Alt-Diffusion, and Segmind Stable Diffusion.
*   **Seamless Integration:** Drag and drop images and parameters for easy editing and sharing.
*   **User-Friendly Interface:** Features mouseover hints, a settings page, and a customizable UI layout.
*   **API Support:** Access the functionality of the web UI through a robust API.

## Installation and Running:

Detailed installation instructions are available for:

*   **NVidia GPUs** (Recommended)
*   **AMD GPUs**
*   **Intel CPUs and GPUs**
*   **Ascend NPUs**
*   **Apple Silicon**

Refer to the [wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki) for specific instructions and dependencies.

### Quick Start (Windows with NVidia GPUs):

1.  Download `sd.webui.zip` from [v1.0.0-pre](https://github.com/AUTOMATIC1111/stable-diffusion-webui/releases/tag/v1.0.0-pre).
2.  Extract the contents.
3.  Run `update.bat`.
4.  Run `run.bat`.

### Alternative Installation (Windows - Automatic):

1.  Install Python 3.10.6 (and add to PATH).
2.  Install Git.
3.  Clone the repository: `git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git`.
4.  Run `webui-user.bat`.

### Installation on Linux:

See instructions above in original README.

## Contributing:

Contribute to the project by following the [Contributing guidelines](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Contributing).

## Documentation:

Comprehensive documentation is available in the project's [wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki).

## Credits:

This project incorporates code from numerous sources. See `Settings -> Licenses` or `html/licenses.html` for licenses and a full list of credits.