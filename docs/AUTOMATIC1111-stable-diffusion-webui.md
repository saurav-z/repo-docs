# Stable Diffusion Web UI: Unleash Your Creativity with AI-Powered Image Generation

**Create stunning images from text prompts with the Stable Diffusion web UI, a user-friendly interface built with Gradio.** [(Original Repository)](https://github.com/AUTOMATIC1111/stable-diffusion-webui)

This powerful web interface makes it easy to harness the potential of Stable Diffusion, a leading AI model for generating images.  Explore a wealth of features to bring your creative visions to life!

## Key Features

*   **Core Image Generation:**
    *   txt2img and img2img modes for creating images from text prompts or modifying existing images.
    *   Outpainting and Inpainting for expanding or editing images.
    *   Color Sketch to generate images based on color sketches.
*   **Advanced Features:**
    *   **Prompt Engineering:** Leverage attention mechanisms and negative prompts for precise control over image generation.
    *   **Interactive Control:**  Prompt Matrix, X/Y/Z plot for exploring different parameter combinations.
    *   **Upscaling and Enhancement:** Includes GFPGAN, CodeFormer, RealESRGAN, ESRGAN, SwinIR, Swin2SR, and LDSR for enhancing image quality.
    *   **Prompt Editing & Styles:**  Modify prompts mid-generation and save/apply prompt styles for efficiency.
    *   **Variations & Seed Control:** Generate image variations and control image seeds for fine-tuning.
    *   **Textual Inversion, Hypernetworks, LoRAs:** Fine-tune the model with your own custom styles and concepts.
    *   **Composable Diffusion:** Combine multiple prompts for complex image generation.
*   **User-Friendly Interface:**
    *   One-click install and run scripts for easy setup (Python and Git required).
    *   Live generation preview and progress bar.
    *   Generation parameters saved with images for reproducibility.
    *   Settings page and mouseover hints for ease of use.
    *   Optional extensions like History Tab and Aesthetic Gradients.
*   **Performance & Compatibility:**
    *   4GB video card support (with reports of 2GB working)
    *   xformers support for improved speed on select cards
    *   Multiple Sampling Methods and adjustable noise settings.
    *   Supports Stable Diffusion 2.0 and Alt-Diffusion.
    *   Supports Segmind Stable Diffusion

## Installation and Running

Comprehensive installation instructions are available in the project's [wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki). Choose the installation method best suited for your system:

*   **Recommended:** [NVidia](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-NVidia-GPUs) GPUs.
*   [AMD](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-AMD-GPUs) GPUs.
*   [Intel CPUs, Intel GPUs (both integrated and discrete)](https://github.com/openvinotoolkit/stable-diffusion-webui/wiki/Installation-on-Intel-Silicon) (external wiki page)
*   [Ascend NPUs](https://github.com/wangshuai09/stable-diffusion-webui/wiki/Install-and-run-on-Ascend-NPUs) (external wiki page)
*   **Online Services:** Utilize online services like Google Colab, (see [List of Online Services](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Online-Services)).
*   **Automatic Installation on Windows:**
    1.  Install [Python 3.10.6](https://www.python.org/downloads/release/python-3106/) and [git](https://git-scm.com/download/win).
    2.  Download the stable-diffusion-webui repository using git clone.
    3.  Run `webui-user.bat` from Windows Explorer.
*   **Automatic Installation on Linux:** Follow the steps outlined in the original README.
*   **Installation on Apple Silicon:** Find instructions [here](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Installation-on-Apple-Silicon).

## Contributing

Learn how to contribute to the project in the [Contributing](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Contributing) section of the wiki.

## Documentation

Find detailed documentation in the project's comprehensive [wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki). For search engine crawling:  [crawlable wiki](https://github-wiki-see.page/m/AUTOMATIC1111/stable-diffusion-webui/wiki).

## Credits

The project leverages contributions and code from various sources, including: Stable Diffusion, k-diffusion, Spandrel (GFPGAN, CodeFormer, ESRGAN, SwinIR, Swin2SR, LDSR), MiDaS, and many others. Licenses are available within the application's settings and in `html/licenses.html`.