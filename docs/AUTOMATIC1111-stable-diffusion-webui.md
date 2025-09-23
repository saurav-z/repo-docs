# Stable Diffusion Web UI: Unleash Your Creativity with AI-Powered Image Generation

**Generate stunning images from text prompts with the Stable Diffusion web UI, a user-friendly interface built with Gradio.** ([Original Repo](https://github.com/AUTOMATIC1111/stable-diffusion-webui))

This powerful tool puts the cutting edge of AI image generation at your fingertips, enabling you to create incredible visuals with ease.

## Key Features:

*   **Text-to-Image & Image-to-Image:** Generate images from text descriptions or modify existing images.
*   **One-Click Installation:** Simplify setup with easy-to-use install scripts.
*   **Advanced Image Editing:** Utilize outpainting, inpainting, color sketching, and more.
*   **Prompt Optimization:** Fine-tune image generation with prompt weighting, negative prompts, and styles.
*   **Upscaling & Enhancement:** Improve image quality with GFPGAN, CodeFormer, RealESRGAN, and other upscalers.
*   **Extensive Customization:** Explore features like X/Y/Z plots, textual inversion, custom scripts, and hypernetworks.
*   **Community-Driven:** Benefit from a wealth of community extensions and integrations.
*   **Comprehensive Feature Set:**
    *   Prompt Matrix
    *   Stable Diffusion Upscale
    *   Attention mechanisms to specify parts of text
    *   Loopback, run img2img processing multiple times
    *   X/Y/Z plot, a way to draw a 3 dimensional plot of images with different parameters
    *   Textual Inversion
    *   Extras tab with face restoration and upscaling tools
    *   Resizing aspect ratio options
    *   Sampling method selection
    *   Interrupt processing at any time
    *   4GB video card support (also reports of 2GB working)
    *   Correct seeds for batches
    *   Live prompt token length validation
    *   Generation parameters saved with image
    *   Read Generation Parameters Button, loads parameters in promptbox to UI
    *   Settings page
    *   Running arbitrary python code from UI (must run with `--allow-code` to enable)
    *   Mouseover hints for most UI elements
    *   Possible to change defaults/mix/max/step values for UI elements via text config
    *   Tiling support, a checkbox to create images that can be tiled like textures
    *   Progress bar and live image generation preview
    *   Negative prompt
    *   Styles
    *   Variations
    *   Seed resizing
    *   CLIP interrogator, a button that tries to guess prompt from an image
    *   Prompt Editing, a way to change prompt mid-generation
    *   Batch Processing, process a group of files using img2img
    *   Img2img Alternative, reverse Euler method of cross attention control
    *   Highres Fix
    *   Reloading checkpoints on the fly
    *   Checkpoint Merger
    *   Custom scripts with many extensions from community
    *   Composable-Diffusion
    *   No token limit for prompts
    *   DeepDanbooru integration
    *   xformers, major speed increase for select cards
    *   History tab
    *   Generate forever option
    *   Training tab (hypernetworks, embeddings)
    *   Clip skip
    *   Hypernetworks
    *   Loras
    *   A separate UI where you can choose, with preview, which embeddings, hypernetworks or Loras to add to your prompt
    *   Can select to load a different VAE from settings screen
    *   Estimated completion time in progress bar
    *   API
    *   Support for dedicated inpainting model
    *   Aesthetic Gradients, a way to generate images with a specific aesthetic by using clip images embeds
    *   Stable Diffusion 2.0 support
    *   Alt-Diffusion support
    *   Load checkpoints in safetensors format
    *   Eased resolution restriction
    *   Reorder elements in the UI from settings screen
    *   Segmind Stable Diffusion support

## Installation and Running

Detailed installation instructions are available in the project's wiki, covering various operating systems and hardware configurations:

*   **[NVidia GPUs](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-NVidia-GPUs)** (Recommended)
*   **[AMD GPUs](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-AMD-GPUs)**
*   **[Intel CPUs/GPUs](https://github.com/openvinotoolkit/stable-diffusion-webui/wiki/Installation-on-Intel-Silicon)** (external wiki page)
*   **[Ascend NPUs](https://github.com/wangshuai09/stable-diffusion-webui/wiki/Install-and-run-on-Ascend-NPUs)** (external wiki page)
*   **[Online Services](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Online-Services)** (e.g., Google Colab)

### Quick Start for Windows (NVidia GPUs)

1.  Download `sd.webui.zip` from [v1.0.0-pre](https://github.com/AUTOMATIC1111/stable-diffusion-webui/releases/tag/v1.0.0-pre) and extract.
2.  Run `update.bat`.
3.  Run `run.bat`.

### Automatic Installation (Windows)

1.  Install [Python 3.10.6](https://www.python.org/downloads/release/python-3106/) and Git.
2.  Clone the repository: `git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git`.
3.  Run `webui-user.bat`.

### Automatic Installation (Linux)

1.  Install dependencies (example for Debian-based systems): `sudo apt install wget git python3 python3-venv libgl1 libglib2.0-0`.
2.  Navigate to your desired directory and run: `wget -q https://raw.githubusercontent.com/AUTOMATIC1111/stable-diffusion-webui/master/webui.sh`
    or clone: `git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui`
3.  Run `./webui.sh`.

### Installation on Apple Silicon

Find the instructions [here](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Installation-on-Apple-Silicon).

## Contributing

Learn how to contribute to the project: [Contributing Guide](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Contributing)

## Documentation

Comprehensive documentation is available on the project's [wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki).

## Credits

This project is built upon the work of many contributors and libraries.  See `Settings -> Licenses` or `html/licenses.html` for detailed licensing information.

Key credits include:

*   Stable Diffusion (Stability AI)
*   k-diffusion
*   Spandrel (GFPGAN, CodeFormer, ESRGAN, SwinIR, Swin2SR)
*   LDSR
*   and many more (see the original README for a full list).