# Stable Diffusion Web UI: Unleash Your Creativity with AI-Powered Image Generation

**[AUTOMATIC1111's Stable Diffusion Web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) is a user-friendly web interface built on the Gradio library, empowering you to generate stunning images from text prompts using the power of Stable Diffusion.**

## Key Features

*   **Core Image Generation:**
    *   Original txt2img and img2img modes for text-to-image and image-to-image creation.
    *   Outpainting and Inpainting capabilities for expanding and modifying images.
    *   Color Sketch and Prompt Matrix features for creative exploration.
*   **Advanced Control & Customization:**
    *   Attention mechanism for fine-tuning prompt focus (`((tuxedo))` syntax).
    *   Negative prompt functionality to specify unwanted elements.
    *   Styles feature for saving and applying prompt snippets.
    *   Seed resizing and Variations for iterative image refinement.
    *   X/Y/Z plot for parameter exploration and image comparison.
*   **Upscaling & Enhancement:**
    *   Integrated upscalers: GFPGAN, CodeFormer, RealESRGAN, ESRGAN, SwinIR, Swin2SR, and LDSR for enhanced image quality.
*   **AI-Driven Tools & Integrations:**
    *   CLIP interrogator to guess prompts from images.
    *   DeepDanbooru integration for generating anime-style tags.
    *   Composable-Diffusion for advanced prompt combinations.
    *   Support for Stable Diffusion 2.0, Alt-Diffusion, and Segmind Stable Diffusion.
    *   Aesthetic Gradients
*   **User-Friendly Interface:**
    *   One-click install and run scripts (Python and Git required).
    *   Live image generation preview with progress bar.
    *   Generation parameters saved with images (PNG/EXIF).
    *   Read Generation Parameters Button to load parameters from an image.
    *   Settings page for extensive customization.
    *   Mouseover hints for UI elements.
*   **Powerful Extensions & Training:**
    *   Custom scripts and community extensions.
    *   Training tab for embeddings, hypernetworks, and Loras.
    *   Hypernetworks, Loras, and embeddings UI for adding these to prompts.
    *   History tab via extension: view, direct and delete images conveniently within the UI
*   **Performance & Efficiency:**
    *   4GB video card support (with reports of 2GB working).
    *   xformers optimization for speed on select cards.
*   **Additional Features:**
    *   Tiling support for creating seamless textures.
    *   Prompt editing mid-generation.
    *   Batch Processing for processing a group of files using img2img.
    *   Reloading checkpoints on the fly.
    *   Checkpoint Merger.
    *   API Support.
    *   Support for dedicated inpainting model by RunwayML.
    *   Eased resolution restriction.

## Installation and Running

Detailed installation instructions are available for various platforms, including:

*   NVidia GPUs (Recommended)
*   AMD GPUs
*   Intel CPUs and GPUs
*   Ascend NPUs
*   Apple Silicon

You can also leverage online services such as Google Colab.  Refer to the [wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki) for specific instructions and dependencies.

### Quick Start (Windows with NVidia GPUs):

1.  Download `sd.webui.zip` from the releases.
2.  Run `update.bat`.
3.  Run `run.bat`.

### Automatic Installation (Windows):

1.  Install Python 3.10.6 and add it to your PATH.
2.  Install Git.
3.  Clone the repository using `git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git`.
4.  Run `webui-user.bat`.

### Automatic Installation (Linux):

1.  Install dependencies using your package manager (e.g., `apt`, `dnf`).  See the wiki for specific commands.
2.  Run `webui.sh` in the desired installation directory or clone the repo, then run `webui.sh`.

### Installation on Apple Silicon

Find the instructions [here](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Installation-on-Apple-Silicon).

## Contributing

Contribute to this project by following the guidelines on the [Contributing](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Contributing) page.

## Documentation

Comprehensive documentation, including installation guides, feature explanations, and troubleshooting tips, is available on the project's [wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki).

## Credits

This project relies on contributions from many individuals and open-source projects.  License information for borrowed code is found in `Settings -> Licenses` and `html/licenses.html`.  See the original README for a full list of credits.