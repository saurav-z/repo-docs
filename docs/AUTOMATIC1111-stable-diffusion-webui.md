# Stable Diffusion Web UI: Unleash Your Creativity with AI-Powered Image Generation

**Create stunning images from text prompts with the Stable Diffusion web UI, a user-friendly interface built with Gradio.** ([Original Repository](https://github.com/AUTOMATIC1111/stable-diffusion-webui))

## Key Features

*   **Versatile Image Generation Modes:**
    *   txt2img and img2img for text-to-image and image-to-image transformations.
    *   Outpainting and Inpainting for extending and modifying images.
    *   Color Sketch to generate images from sketches.
    *   Prompt Matrix to explore multiple prompt combinations.
*   **Advanced Image Manipulation:**
    *   Stable Diffusion Upscale for enhancing image resolution.
    *   Attention mechanism to fine-tune specific text elements within prompts.
    *   Loopback for iterative img2img processing.
    *   X/Y/Z plot for visualizing image generation parameters.
*   **Enhanced Prompting & Control:**
    *   Negative prompt for specifying unwanted image elements.
    *   Styles for saving and applying prompt elements easily.
    *   Variations for generating slight image alterations.
    *   Seed resizing for generating images at different resolutions.
    *   CLIP interrogator for generating prompts from images.
    *   Prompt editing to modify prompts mid-generation.
    *   Composable Diffusion for combining multiple prompts.
    *   No token limit for prompts.
*   **Customization & Extensibility:**
    *   Extensive settings page.
    *   Support for custom scripts via community extensions.
    *   Training tab for hypernetworks and embeddings.
    *   Hypernetworks, Loras, and Embeddings for style and content customization.
    *   Integration of DeepDanbooru for anime prompts.
    *   Xformers for performance gains on select GPUs.
    *   Checkpoint merging
    *   Automatic loading of generation parameters from images.
*   **Image Enhancement Tools:**
    *   GFPGAN and CodeFormer for face restoration.
    *   RealESRGAN and ESRGAN for image upscaling.
    *   SwinIR, Swin2SR and LDSR for upscaling.
*   **Performance and Convenience:**
    *   Progress bar with live image preview.
    *   Support for 4GB and 2GB video cards.
    *   Correct seeds for batches.
    *   Live prompt token length validation.
    *   Checkpoint reloading on the fly.
    *   Estimated completion time in progress bar.
*   **Advanced Features:**
    *   Batch Processing
    *   Highres Fix
    *   API
    *   Support for Stable Diffusion 2.0 and Alt-Diffusion.
    *   Support for dedicated inpainting models.

## Installation and Running

Refer to the [installation instructions](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki) for detailed guidance on setting up the web UI on various platforms, including:

*   NVidia GPUs (Recommended)
*   AMD GPUs
*   Intel CPUs and GPUs
*   Ascend NPUs
*   Apple Silicon

You can also leverage online services such as Google Colab, using the [list of online services](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Online-Services).

### Quick Installation (Windows)

1.  Download `sd.webui.zip` from [v1.0.0-pre](https://github.com/AUTOMATIC1111/stable-diffusion-webui/releases/tag/v1.0.0-pre) and extract.
2.  Run `update.bat`.
3.  Run `run.bat`.

### Automatic Installation (Windows)

1.  Install Python 3.10.6, checking "Add Python to PATH".
2.  Install Git.
3.  Clone the repository: `git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git`.
4.  Run `webui-user.bat`.

### Automatic Installation (Linux)

1.  Install dependencies: (Debian-based: `sudo apt install wget git python3 python3-venv libgl1 libglib2.0-0`; Red Hat-based: `sudo dnf install wget git python3 gperftools-libs libglvnd-glx`; openSUSE-based: `sudo zypper install wget git python3 libtcmalloc4 libglvnd`; Arch-based: `sudo pacman -S wget git python3`).  Consider Python 3.10 or 3.11 if on a newer system.
2.  Navigate to your desired installation directory.
3.  Run: `wget -q https://raw.githubusercontent.com/AUTOMATIC1111/stable-diffusion-webui/master/webui.sh` or `git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui`.
4.  Run `webui.sh`.
5.  See `webui-user.sh` for configuration options.

### Installation on Apple Silicon

Find instructions [here](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Installation-on-Apple-Silicon).

## Contributing

See [Contributing](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Contributing) for information on how to contribute to the project.

## Documentation

Extensive documentation is available on the project's [wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki), including a [crawlable wiki](https://github-wiki-see.page/m/AUTOMATIC1111/stable-diffusion-webui/wiki) for search engine optimization.

## Credits

A comprehensive list of credits and licenses for borrowed code can be found in the `Settings -> Licenses` screen and the `html/licenses.html` file.