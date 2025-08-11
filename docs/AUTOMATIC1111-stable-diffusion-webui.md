# Stable Diffusion Web UI: Your Gateway to AI-Powered Image Generation

Unleash your creativity with the **Stable Diffusion web UI**, a user-friendly interface built with Gradio, offering unparalleled control and customization for generating stunning images from text prompts.  [Explore the original repo here](https://github.com/AUTOMATIC1111/stable-diffusion-webui).

## Key Features

*   **Core Image Generation:** Utilize the original `txt2img` and `img2img` modes for generating images from text and modifying existing images.
*   **One-Click Installation:** Get up and running quickly with a simplified install and run script (Python and Git required).
*   **Advanced Editing & Control:**
    *   **Outpainting & Inpainting:** Expand and refine your images with powerful editing tools.
    *   **Prompting Enhancements:** Leverage attention mechanisms (e.g., `((tuxedo))`), negative prompts, styles, and variations for precise control over image composition.
    *   **Prompt Editing:**  Modify your prompt mid-generation for iterative refinement.
*   **AI Upscaling & Enhancement:**
    *   **Face Restoration:** Integrate with GFPGAN and CodeFormer for realistic face enhancement.
    *   **Image Upscaling:**  Utilize RealESRGAN, ESRGAN, LDSR, SwinIR, and Swin2SR for high-resolution outputs.
*   **Flexible Workflow:**
    *   **Batch Processing:** Process multiple images at once with img2img.
    *   **Generation Parameters:** Images are saved with generation parameters (in PNG chunks or EXIF data), making it easy to reproduce or modify images.  Drag and drop images to promptbox.
    *   **Checkpoint Merging:** Merge up to 3 checkpoints with checkpoint merger.
    *   **Custom Scripts:** Extend functionality with community-created scripts.
*   **Innovative Features:**
    *   **X/Y/Z Plot:**  Create 3D plots of images with varying parameters.
    *   **Textual Inversion:** Train and use custom embeddings for personalized image styles.
    *   **Composable-Diffusion:** Combine multiple prompts using `AND` for complex compositions.
    *   **DeepDanbooru Integration:** Generate anime-style tags.
    *   **Aesthetic Gradients Integration:** Generate images with a specific aesthetic.
    *   **Loras, Hypernetworks, Embeddings:**  Use a separate UI where you can choose, with preview, which embeddings, hypernetworks or Loras to add to your prompt
*   **Performance & Compatibility:**
    *   **Hardware Support:** Optimized for 4GB+ video cards (reports of 2GB working), with xformers for enhanced speed on select cards.
    *   **Platform Support:**  Supports NVIDIA, AMD, and Intel GPUs, and Intel CPUs
*   **Community & Extensions:**
    *   **API Access:** Integrate the web UI into other applications.
    *   **Extensible:** Benefit from a vibrant community creating custom scripts.
*   **Recent Updates:**
    *   Safetensors format
    *   Eased resolution restrictions
    *   Segmind Stable Diffusion Support

## Installation and Running

Detailed installation instructions are available on the [wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki) for various platforms, including:

*   **NVidia GPUs (Recommended)**
*   **AMD GPUs**
*   **Intel CPUs/GPUs**
*   **Ascend NPUs**
*   **Apple Silicon**
*   **Online Services (Google Colab)**

### Quick Start (Windows with NVidia GPUs)

1.  Download the `sd.webui.zip` from a release.
2.  Run `update.bat`.
3.  Run `run.bat`.

### Automatic Installation (Windows)

1.  Install [Python 3.10.6](https://www.python.org/downloads/release/python-3106/) and Git.
2.  Clone the repository: `git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git`.
3.  Run `webui-user.bat`.

### Automatic Installation (Linux)

1.  Install dependencies (examples provided in original README).
2.  Navigate to your desired installation directory and run:
    *   `wget -q https://raw.githubusercontent.com/AUTOMATIC1111/stable-diffusion-webui/master/webui.sh`
    *   or `git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui`
3.  Run `webui.sh`.
4.  Configure options in `webui-user.sh`.

### Installation on Apple Silicon

Find the instructions [here](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Installation-on-Apple-Silicon).

## Contributing

Contribute to the project by following the guidelines outlined in the [Contributing](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Contributing) section of the wiki.

## Documentation

Comprehensive documentation, including tutorials and troubleshooting guides, is available on the project's [wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki).
For Google search index purposes, here's a link to the [crawlable wiki](https://github-wiki-see.page/m/AUTOMATIC1111/stable-diffusion-webui/wiki).

## Credits

A complete list of credits and licenses for the borrowed code can be found in the `Settings -> Licenses` screen and `html/licenses.html` file.