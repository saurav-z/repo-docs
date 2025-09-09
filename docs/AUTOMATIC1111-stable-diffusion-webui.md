# Stable Diffusion WebUI: Unleash Your Creativity with AI-Powered Image Generation

**Transform your imagination into stunning visuals with the Stable Diffusion web UI, a user-friendly interface for the groundbreaking Stable Diffusion image generation model.**  ([View the original repository](https://github.com/AUTOMATIC1111/stable-diffusion-webui))

## Key Features

*   **Core Image Generation:**
    *   txt2img and img2img modes for versatile image creation.
    *   Outpainting and Inpainting for extending and editing images.
    *   Color Sketch to generate images from sketches.
*   **Advanced Control & Customization:**
    *   Prompt Matrix for exploring multiple variations.
    *   Attention mechanism for fine-tuning prompt emphasis (e.g., `((tuxedo))` for increased focus).
    *   Negative prompt to specify unwanted elements.
    *   Styles: Save and easily apply prompt segments.
    *   Variations & Seed resizing to explore similar images.
    *   CLIP interrogator to generate prompts from images.
    *   Prompt editing to modify prompts mid-generation.
    *   Composable-Diffusion support for complex prompt combinations (using `AND`).
    *   No token limit for prompts.
*   **Enhancements & Upscaling:**
    *   Stable Diffusion Upscale for improved image quality.
    *   Extras tab with Face Restoration (GFPGAN, CodeFormer) and Upscaling (RealESRGAN, ESRGAN, SwinIR, Swin2SR, LDSR) tools.
    *   Highres Fix for creating high-resolution images with minimal distortion.
*   **Workflow & Efficiency:**
    *   One-click install and run script (requires Python and Git).
    *   Live generation preview and progress bar.
    *   Generation parameters are saved with images (PNG chunks/EXIF).
    *   Read Generation Parameters button for easy parameter restoration.
    *   Interrupt processing at any time.
    *   Support for 4GB video cards (and reports of 2GB working).
    *   Correct seeds for batches.
    *   Checkpoint Merger to combine checkpoints.
    *   Reloading checkpoints on the fly.
    *   Batch Processing of image files.
    *   Generate Forever option.
    *   Reorder UI elements from settings screen.
*   **Advanced Features:**
    *   Textual Inversion for training and using custom embeddings.
    *   X/Y/Z plot for parameter exploration.
    *   Training tab for hypernetworks, embeddings and LORAs.
    *   API for integration with other applications.
    *   DeepDanbooru integration for anime prompts.
    *   Support for dedicated inpainting models.
    *   Alt-Diffusion and Stable Diffusion 2.0 support.
    *   Load checkpoints in safetensors format.
    *   Eased resolution restrictions (multiples of 8).
*   **Community & Extensibility:**
    *   Custom scripts via extensions.
    *   Aesthetic Gradients support.
    *   UI for managing embeddings, hypernetworks, and LORAs.
*   **Hardware Support**
    *   AMD and NVidia GPU support.
    *   Intel CPU and GPU support.
    *   Ascend NPU support.

## Installation

Detailed installation instructions are available for:

*   [NVidia GPUs](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-NVidia-GPUs) (recommended)
*   [AMD GPUs](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-AMD-GPUs)
*   [Intel CPUs/GPUs](https://github.com/openvinotoolkit/stable-diffusion-webui/wiki/Installation-on-Intel-Silicon)
*   [Ascend NPUs](https://github.com/wangshuai09/stable-diffusion-webui/wiki/Install-and-run-on-Ascend-NPUs)
*   [Apple Silicon](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Installation-on-Apple-Silicon)

**Simplified Installation (Windows with NVidia GPUs - Release Package):**

1.  Download the `sd.webui.zip` release from [v1.0.0-pre](https://github.com/AUTOMATIC1111/stable-diffusion-webui/releases/tag/v1.0.0-pre)
2.  Extract the contents.
3.  Run `update.bat`.
4.  Run `run.bat`.

**Automatic Installation (Windows):**

1.  Install [Python 3.10.6](https://www.python.org/downloads/release/python-3106/) and Git.
2.  Clone the repository (e.g., `git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git`).
3.  Run `webui-user.bat`.

**Automatic Installation (Linux):**

1.  Install dependencies (example commands provided in original README).
2.  Navigate to your desired install directory.
3.  Run `wget -q https://raw.githubusercontent.com/AUTOMATIC1111/stable-diffusion-webui/master/webui.sh` or clone the repository.
4.  Run `webui.sh`.
5.  See `webui-user.sh` for configuration options.

## Contributing

[Contribute to the project here](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Contributing).

## Documentation

Access comprehensive documentation on the project's [wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki).
For search engine crawling, see the [crawlable wiki](https://github-wiki-see.page/m/AUTOMATIC1111/stable-diffusion-webui/wiki).

## Credits

The project utilizes various open-source components. Licenses and credits are available in `Settings -> Licenses` and `html/licenses.html`. A list of key contributors and their contributions can be found in the original README.