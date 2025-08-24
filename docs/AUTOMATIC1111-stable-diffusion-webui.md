# Stable Diffusion Web UI: Unleash Your Creative Vision

**Harness the power of Stable Diffusion with this intuitive web interface, bringing AI-powered image generation to your fingertips.** ([Back to Original Repo](https://github.com/AUTOMATIC1111/stable-diffusion-webui))

## Key Features:

*   **Versatile Image Generation:** Create stunning images from text prompts (txt2img) or edit existing images (img2img) with ease.
*   **One-Click Setup:** Simplify the installation process with a straightforward script (Python and Git required).
*   **Advanced Editing Tools:**
    *   **Outpainting & Inpainting:** Extend and modify images creatively.
    *   **Prompt Refinement:** Fine-tune your prompts with features like attention mechanisms (`((tuxedo))`) and prompt editing during generation.
    *   **X/Y/Z Plot:** Visualize and experiment with different parameters using 3D plots.
*   **AI-Powered Enhancements:**
    *   **Face Restoration:** Utilize tools like GFPGAN and CodeFormer to enhance facial details.
    *   **Upscaling:** Improve image resolution with RealESRGAN, ESRGAN, SwinIR, and LDSR.
*   **Flexible Control & Customization:**
    *   **Negative Prompts:** Exclude unwanted elements from your images.
    *   **Styles & Variations:** Save and apply prompt styles, generate image variations with subtle differences.
    *   **Seed Control:** Maintain consistency and generate variations with seed resizing.
    *   **Settings Page:** Customize defaults and UI element behavior.
*   **Workflow Efficiency:**
    *   **Batch Processing:** Process multiple images simultaneously.
    *   **Drag-and-Drop:** Easily load images and parameters.
    *   **Progress Preview:** Monitor generation progress with a live preview.
*   **Advanced Features:**
    *   **Textual Inversion:** Train and use custom embeddings.
    *   **Checkpoint Management:** Merge and reload checkpoints on the fly.
    *   **Custom Scripts:** Extend functionality with community-created scripts.
    *   **Composable Diffusion:** Utilize multiple prompts with weights.
    *   **DeepDanbooru Integration:** Generate anime-style tags.
    *   **API Support:** Integrate with other applications.
    *   **Loras, Hypernetworks, and Embeddings Support:** Customize prompts with pre-trained models.
*   **Optimized Performance:** Benefit from features like xformers for faster performance.
*   **Expanded Model Support:**
    *   Stable Diffusion 2.0 & Alt-Diffusion support.
    *   Segmind Stable Diffusion support.

## Installation

Detailed installation instructions for various platforms (NVidia, AMD, Intel, Apple Silicon, and others) are available on the [project wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki).

### Quick Start (Windows with NVidia GPU):

1.  Download `sd.webui.zip` from the [releases](https://github.com/AUTOMATIC1111/stable-diffusion-webui/releases).
2.  Run `update.bat`.
3.  Run `run.bat`.

### Automatic Installation (Windows):

1.  Install [Python 3.10.6](https://www.python.org/downloads/release/python-3106/) (and add to PATH).
2.  Install [git](https://git-scm.com/download/win).
3.  Clone the repository: `git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git`
4.  Run `webui-user.bat`.

### Automatic Installation (Linux):

1.  Install dependencies (instructions in original README).
2.  Run `wget -q https://raw.githubusercontent.com/AUTOMATIC1111/stable-diffusion-webui/master/webui.sh` or clone repo.
3.  Run `./webui.sh`.

## Contributing

Contribute to the project by following the [Contributing](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Contributing) guidelines.

## Documentation

Access comprehensive documentation on the project's [wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki).  For search engine indexing, a crawlable version is available [here](https://github-wiki-see.page/m/AUTOMATIC1111/stable-diffusion-webui/wiki).

## Credits

See `Settings -> Licenses` screen or `html/licenses.html` for information on borrowed code licenses and a list of contributors (see original README for full list).