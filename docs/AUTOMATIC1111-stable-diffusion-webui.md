# Stable Diffusion Web UI: Your Gateway to Cutting-Edge AI Image Generation

[Visit the original repository](https://github.com/AUTOMATIC1111/stable-diffusion-webui) for the source code and the latest updates.

## Unleash Your Creativity with Powerful Features

This web interface, built with Gradio, provides an accessible and feature-rich environment for generating images with Stable Diffusion.

**Key Features:**

*   **Core Generation Modes:** Harness the power of txt2img and img2img to create stunning visuals from text prompts or by modifying existing images.
*   **One-Click Installation:** Simplify setup with a straightforward install script (Python and Git required).
*   **Advanced Image Editing:** Utilize inpainting, outpainting, color sketches, and more to refine your creations.
*   **Image Enhancement Tools:**
    *   Upscale your images with GFPGAN, CodeFormer, RealESRGAN, ESRGAN, SwinIR, Swin2SR, and LDSR.
*   **Prompt Engineering Power:**
    *   **Attention:** Fine-tune the model's focus with attention mechanisms.
    *   **Negative Prompt:** Specify what you *don't* want to see in your generated images.
    *   **Styles:** Save and apply prompt snippets for consistent aesthetics.
    *   **Prompt Editing:** Modify prompts mid-generation for dynamic results.
    *   **Composable Diffusion:** Combine multiple prompts with weights for complex images.
    *   **CLIP Interrogator:** Automatically generate prompts from an image.
    *   **No token limit for prompts**
*   **Workflow Enhancements:**
    *   **Loopback:** Process images iteratively.
    *   **X/Y/Z Plot:** Generate 3D plots of images with varying parameters.
    *   **Seed Control:** Maintain consistency with batch seeds or vary them for diverse outputs.
    *   **Batch Processing:** Process multiple images at once with img2img.
    *   **Variations:** Generate similar images with slight differences.
    *   **Highres Fix:** Produce high-resolution images with one click.
*   **Customization & Extensibility:**
    *   **Settings Page:** Customize your experience.
    *   **Custom Scripts:** Extend functionality with community-created scripts.
    *   **Training Tab:** Fine-tune models with hypernetworks, embeddings, and Loras.
    *   **Checkpoint Merger:** Merge up to 3 checkpoints.
    *   **API:** Integrate with other applications.
    *   **Extensions:**
        *   History Tab
        *   Aesthetic Gradients

*   **Performance & Compatibility:**
    *   4GB video card support (and even reported 2GB compatibility).
    *   xformers for faster processing on select GPUs.
    *   Support for Stable Diffusion 2.0 and Alt-Diffusion.
    *   Eased resolution restrictions.
    *   Support for Segmind Stable Diffusion.

## Installation and Running

Detailed installation instructions are available in the [wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki), including:

*   **NVidia GPUs** (Recommended)
*   **AMD GPUs**
*   **Intel CPUs and GPUs**
*   **Ascend NPUs**
*   **Apple Silicon**
*   **Online Services** (Google Colab, etc.)

### Quick Start for Windows with NVidia GPUs

1.  Download the `sd.webui.zip` from [v1.0.0-pre](https://github.com/AUTOMATIC1111/stable-diffusion-webui/releases/tag/v1.0.0-pre) and extract its contents.
2.  Run `update.bat`.
3.  Run `run.bat`.

### Automatic Installation on Windows

1.  Install [Python 3.10.6](https://www.python.org/downloads/release/python-3106/) (and add it to your PATH).
2.  Install [git](https://git-scm.com/download/win).
3.  Clone the repository: `git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git`.
4.  Run `webui-user.bat`.

### Automatic Installation on Linux

1.  Install dependencies:
    *   (Debian-based): `sudo apt install wget git python3 python3-venv libgl1 libglib2.0-0`
    *   (Red Hat-based): `sudo dnf install wget git python3 gperftools-libs libglvnd-glx`
    *   (openSUSE-based): `sudo zypper install wget git python3 libtcmalloc4 libglvnd`
    *   (Arch-based): `sudo pacman -S wget git python3`
2.  Navigate to your desired directory.
3.  Run `wget -q https://raw.githubusercontent.com/AUTOMATIC1111/stable-diffusion-webui/master/webui.sh` OR clone the repo using git.
4.  Run `./webui.sh`.

## Contributing

Contributions are welcome! See the [Contributing](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Contributing) guide for details.

## Documentation

Comprehensive documentation, including detailed feature explanations and troubleshooting guides, is available on the project's [wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki).