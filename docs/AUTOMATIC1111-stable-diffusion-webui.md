# Stable Diffusion Web UI: Unleash Your Creativity with AI-Powered Image Generation

**Create stunning images from text prompts with the Stable Diffusion web UI, a user-friendly interface built with Gradio.** ([Original Repo](https://github.com/AUTOMATIC1111/stable-diffusion-webui))

**Key Features:**

*   **Versatile Generation Modes:** Utilize `txt2img` and `img2img` modes to generate images from text or transform existing images.
*   **One-Click Installation:** Quickly get started with a simplified install script (Python and Git required).
*   **Advanced Editing & Control:** Includes Outpainting, Inpainting, Color Sketch, Prompt Matrix, and prompt attention features.
*   **Powerful Upscaling & Enhancement:** Features Stable Diffusion Upscale and integrations with advanced upscalers like GFPGAN, CodeFormer, RealESRGAN, ESRGAN, SwinIR, Swin2SR, and LDSR.
*   **Flexible Parameter Control:** Adjust sampling methods, eta values, and noise settings for fine-tuned image generation.
*   **Interactive Features:** Enjoy live image previews, generation parameter saving, and easy restoration via drag-and-drop.
*   **Negative Prompts:** Refine your results by specifying what you *don't* want in the generated image.
*   **Customization & Extensibility:** Supports Styles, Variations, Seed resizing, custom scripts, and integrations with tools like Composable-Diffusion, DeepDanbooru, xformers and Loras.
*   **Comprehensive Training Capabilities:** Train hypernetworks, embeddings, and LoRAs within the UI.
*   **API Support:** Integrate with other applications through a robust API.
*   **Extensive Community Support:** Utilize a vast library of extensions and community-contributed features.

## Installation and Running

Detailed installation instructions are available for:

*   [NVidia GPUs](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-NVidia-GPUs) (recommended)
*   [AMD GPUs](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-AMD-GPUs)
*   [Intel CPUs, Intel GPUs (both integrated and discrete)](https://github.com/openvinotoolkit/stable-diffusion-webui/wiki/Installation-on-Intel-Silicon) (external wiki page)
*   [Ascend NPUs](https://github.com/wangshuai09/stable-diffusion-webui/wiki/Install-and-run-on-Ascend-NPUs) (external wiki page)
*   [Apple Silicon](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Installation-on-Apple-Silicon)

**Windows Installation (NVidia GPUs - Recommended):**

1.  Download `sd.webui.zip` from [v1.0.0-pre](https://github.com/AUTOMATIC1111/stable-diffusion-webui/releases/tag/v1.0.0-pre) and extract its contents.
2.  Run `update.bat`.
3.  Run `run.bat`.

**Automatic Installation (Windows):**

1.  Install [Python 3.10.6](https://www.python.org/downloads/release/python-3106/), ensuring "Add Python to PATH" is checked.
2.  Install [git](https://git-scm.com/download/win).
3.  Download the repository using `git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git`.
4.  Run `webui-user.bat` from Windows Explorer.

**Automatic Installation (Linux):**

1.  Install dependencies using the appropriate package manager for your distribution (examples provided in the original README).
2.  Navigate to your desired installation directory and run `wget -q https://raw.githubusercontent.com/AUTOMATIC1111/stable-diffusion-webui/master/webui.sh` or clone the repo.
3.  Run `webui.sh`.
4.  Configure options in `webui-user.sh`.

## Contributing

Contribute to the project by following the guidelines outlined in the [Contributing](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Contributing) section.

## Documentation

Comprehensive documentation is available on the project's [wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki). For search engine indexing, see the [crawlable wiki](https://github-wiki-see.page/m/AUTOMATIC1111/stable-diffusion-webui/wiki).

## Credits

This project leverages code from numerous contributors. Credit and licenses can be found in the `Settings -> Licenses` screen and in the `html/licenses.html` file.