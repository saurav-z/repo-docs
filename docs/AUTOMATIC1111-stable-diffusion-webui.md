# Stable Diffusion WebUI: Unleash Your Creativity with AI-Powered Image Generation

**Create stunning images from text prompts with the Stable Diffusion web UI, a powerful and user-friendly interface built with Gradio.**  [Explore the original repository](https://github.com/AUTOMATIC1111/stable-diffusion-webui) for further details.

![Stable Diffusion WebUI Screenshot](screenshot.png)

## Key Features:

*   **Core Functionality:**
    *   Original txt2img and img2img modes for diverse image generation.
    *   User-friendly interface powered by Gradio.
    *   One-click installation for easy setup (requires Python and Git).
*   **Advanced Editing & Control:**
    *   Outpainting and Inpainting for seamless image extensions and modifications.
    *   Color Sketch and Prompt Matrix for creative exploration.
    *   Attention mechanism to guide the model's focus (e.g., `a man in a ((tuxedo))`).
    *   Negative prompts to specify elements to avoid in the generated image.
    *   Prompt Editing for dynamic adjustments during generation.
    *   Styles for saving and applying prompt components.
    *   Variations and Seed resizing for iterative refinement.
*   **Upscaling & Enhancement:**
    *   Stable Diffusion Upscale for improved image resolution.
    *   Extras Tab: GFPGAN, CodeFormer, RealESRGAN, ESRGAN, SwinIR, Swin2SR, and LDSR for advanced image enhancement and face restoration.
*   **Workflow & Automation:**
    *   Loopback for iterative img2img processing.
    *   X/Y/Z plot for 3D parameter experimentation.
    *   Batch Processing for processing multiple images.
    *   Generation parameters saved with images for reproducibility.
    *   Read Generation Parameters button to load settings.
    *   Custom Scripts for community-developed extensions.
    *   Composable-Diffusion for using multiple prompts with weighting (e.g., `a cat :1.2 AND a dog AND a penguin :2.2`).
    *   Highres Fix for generating high-resolution images.
    *   Reloading checkpoints on the fly for iterative experimentation.
    *   Checkpoint Merger for creating new models.
*   **Advanced Features:**
    *   Textual Inversion for custom embeddings training on 8GB (or even 6GB)
    *   DeepDanbooru integration for anime prompts
    *   xformers optimization for faster generation on compatible GPUs.
    *   Training Tab with Hypernetworks and embeddings options
    *   API access for external applications.
    *   Support for dedicated inpainting model.
*   **User Interface & Enhancements:**
    *   Live prompt token length validation.
    *   Progress bar with estimated completion time.
    *   Integrated CLIP interrogator to guess prompts from images.
    *   Reorder UI elements from the settings screen.
    *   Eased resolution restriction
*   **Model Support:**
    *   Support for Stable Diffusion 2.0 and Alt-Diffusion.
    *   Load checkpoints in safetensors format.
    *   Support for Segmind Stable Diffusion.

## Installation and Running

Detailed installation instructions are available in the project's [wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki), with guides for:

*   NVidia GPUs (recommended)
*   AMD GPUs
*   Intel CPUs and GPUs
*   Apple Silicon
*   Ascend NPUs

Alternative installation options include online services (e.g., Google Colab).

### Quick Start - Windows (NVidia GPU, using release package)

1.  Download `sd.webui.zip` from [v1.0.0-pre](https://github.com/AUTOMATIC1111/stable-diffusion-webui/releases/tag/v1.0.0-pre) and extract.
2.  Run `update.bat`.
3.  Run `run.bat`.

### Automatic Installation - Windows

1.  Install Python 3.10.6 (and add to PATH) and git.
2.  Clone the repository: `git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git`.
3.  Run `webui-user.bat`.

### Automatic Installation - Linux

1.  Install dependencies (see detailed instructions in the wiki).
2.  Navigate to your desired installation directory and run:
    ```bash
    wget -q https://raw.githubusercontent.com/AUTOMATIC1111/stable-diffusion-webui/master/webui.sh
    ```
    Or clone the repo:
    ```bash
    git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui
    ```
3.  Run `webui.sh`.

### Installation - Apple Silicon

Refer to the [wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Installation-on-Apple-Silicon) for detailed instructions.

## Contributing

Contribute to the project by following the guidelines in the [Contributing](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Contributing) wiki page.

## Documentation

Comprehensive documentation is available on the project's [wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki).

## Credits

See `Settings -> Licenses` screen and `html/licenses.html` for licenses of borrowed code.  A non-exhaustive list of credits for various libraries, models, and ideas is provided in the original README.