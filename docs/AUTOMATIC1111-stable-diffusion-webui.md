# Stable Diffusion Web UI: Unleash Your Creativity with AI-Powered Image Generation

**Create stunning images from text prompts with the Stable Diffusion web UI, a powerful and user-friendly interface.** Find the original repository here: [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui).

## Key Features

*   **Core Image Generation:**
    *   Original `txt2img` and `img2img` modes for text-to-image and image-to-image generation.
    *   Outpainting and inpainting capabilities for extending and editing images.
    *   Color Sketch feature for creating images from sketches.
    *   Prompt Matrix for exploring variations using different prompts.
    *   Highres Fix for generating high-resolution images without distortion.
*   **Advanced Image Enhancement:**
    *   Stable Diffusion Upscale for improving image resolution.
    *   GFPGAN and CodeFormer for face restoration and improvement.
    *   RealESRGAN, ESRGAN, SwinIR, Swin2SR, and LDSR upscalers for enhanced details.
    *   Tiling support for creating seamless textures.
*   **Prompting & Control:**
    *   Attention mechanism to focus on specific parts of your prompts.
    *   Negative prompts to specify what to exclude from the generated images.
    *   Styles for saving and applying prompt elements easily.
    *   Variations and seed resizing for iterative refinement.
    *   CLIP interrogator for generating prompts from images.
    *   No token limit for prompts.
    *   Composable-Diffusion for using multiple prompts at once.
*   **Workflow & Customization:**
    *   Batch processing for handling multiple images at once.
    *   X/Y/Z plot for creating 3D plots of image parameters.
    *   Seed resizing
    *   Hypernetworks, Loras, Embeddings
    *   Custom scripts for extending functionality.
    *   Integration with DeepDanbooru for anime prompts.
    *   Checkpoint Merger to merge different checkpoints.
    *   Comprehensive Settings Page for customizing the UI and behavior.
    *   API
*   **Additional Features:**
    *   Progress bar and live image generation preview.
    *   Generation parameters are saved with images.
    *   Reloading checkpoints on the fly.
    *   Support for [inpainting model](https://github.com/runwayml/stable-diffusion#inpainting-with-stable-diffusion).
    *   [Aesthetic Gradients](https://github.com/AUTOMATIC1111/stable-diffusion-webui-aesthetic-gradients)

## Installation and Running

Detailed installation instructions are available for:

*   [NVidia GPUs](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-NVidia-GPUs) (Recommended)
*   [AMD GPUs](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-AMD-GPUs)
*   [Intel CPUs and GPUs](https://github.com/openvinotoolkit/stable-diffusion-webui/wiki/Installation-on-Intel-Silicon)
*   [Ascend NPUs](https://github.com/wangshuai09/stable-diffusion-webui/wiki/Install-and-run-on-Ascend-NPUs)
*   [Apple Silicon](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Installation-on-Apple-Silicon)

You can also utilize online services for ease of use:

*   [List of Online Services](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Online-Services)

### Quick Installation (Windows with NVidia-GPUs)

1.  Download `sd.webui.zip` from the [v1.0.0-pre](https://github.com/AUTOMATIC1111/stable-diffusion-webui/releases/tag/v1.0.0-pre) release.
2.  Extract the contents.
3.  Run `update.bat`.
4.  Run `run.bat`.

### Automatic Installation (Windows)

1.  Install [Python 3.10.6](https://www.python.org/downloads/release/python-3106/) (with "Add Python to PATH" checked).
2.  Install [git](https://git-scm.com/download/win).
3.  Run `git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git`.
4.  Run `webui-user.bat`.

### Automatic Installation (Linux)

1.  Install dependencies (Debian-based: `sudo apt install wget git python3 python3-venv libgl1 libglib2.0-0`; Red Hat-based: `sudo dnf install wget git python3 gperftools-libs libglvnd-glx`).
2.  Run the following command from your desired installation directory: `wget -q https://raw.githubusercontent.com/AUTOMATIC1111/stable-diffusion-webui/master/webui.sh`.
3.  Alternatively, you can clone the repo: `git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui`.
4.  Run `webui.sh`.
5.  Consult `webui-user.sh` for configuration options.

## Contributing

Contribute to the project: [Contributing](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Contributing)

## Documentation

Access comprehensive documentation: [wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki).

## Credits

See `Settings -> Licenses` or `html/licenses.html` for a complete list of credits for the software and libraries.