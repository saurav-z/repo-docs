# Stable Diffusion Web UI: Unleash Your Creative Vision

**Transform your imagination into stunning visuals with the Stable Diffusion web UI, a user-friendly interface built with Gradio for generating images from text prompts.** [Explore the original repository](https://github.com/AUTOMATIC1111/stable-diffusion-webui).

## Key Features

*   **Versatile Generation Modes:** Utilize both `txt2img` (text-to-image) and `img2img` (image-to-image) modes.
*   **One-Click Installation:** Simplify setup with an easy-to-use install and run script (Python and Git are still required).
*   **Advanced Image Editing:** Leverage Outpainting, Inpainting, Color Sketch, and Prompt Matrix features.
*   **Attention Control:** Refine image generation by specifying parts of text that the model should pay more attention to, including advanced attention syntax.
*   **Iterative Refinement:** Use Loopback for iterative `img2img` processing and X/Y/Z plots for 3D parameter exploration.
*   **Textual Inversion:** Train custom embeddings with flexible naming and multiple vector options, supporting half-precision floating-point numbers.
*   **Enhanced Image Processing:** Integrate with tools like GFPGAN, CodeFormer, RealESRGAN, ESRGAN, SwinIR, Swin2SR, and LDSR for face restoration and upscaling.
*   **Flexible Aspect Ratio:** Easily resize images with various aspect ratio options.
*   **Sampling Control:** Select from a variety of sampling methods, including advanced noise setting options.
*   **Interrupt & Save:** Interrupt processing at any time and save generation parameters directly with the image.
*   **Hardware Support:** Compatible with 4GB video cards and supports correct seeds for batch processing.
*   **Interactive Prompting:** Benefit from Live prompt token length validation and drag-and-drop image/parameters support.
*   **Comprehensive Settings:** Access a settings page for customization and running arbitrary python code with the `--allow-code` flag.
*   **User-Friendly Interface:** Enjoy mouseover hints, configurable UI element defaults, and tiling support for seamless texture creation.
*   **Real-Time Feedback:** Monitor progress with a progress bar and live image generation preview.
*   **Negative Prompting:** Specify unwanted elements to exclude from generated images.
*   **Style & Variations:** Save prompt styles for easy reuse and generate variations of existing images.
*   **Advanced Tools:** Utilize Seed resizing, CLIP interrogator, Prompt Editing, Batch Processing, Highres Fix, and Checkpoint Merger.
*   **Extensibility:** Utilize Custom scripts and support for Composable-Diffusion.
*   **Advanced integration:** Includes DeepDanbooru integration, xformers support, History tab, Generate forever option, Training tab, Clip skip, Hypernetworks, Loras, and a separate UI for adding embeddings, hypernetworks, or Loras to your prompt.
*   **Additional Capabilities:** Includes support for dedicated inpainting models, Aesthetic Gradients, Stable Diffusion 2.0, Alt-Diffusion, and more.
*   **Improved Compatibility:** Support for safetensors format and eased resolution restrictions.
*   **Segmind Stable Diffusion support**
*   **Improved UI** Reorder elements in the UI from settings screen.

## Installation

Detailed installation instructions are available for:

*   [NVidia GPUs](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-NVidia-GPUs) (Recommended)
*   [AMD GPUs](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-AMD-GPUs)
*   [Intel CPUs, Intel GPUs](https://github.com/openvinotoolkit/stable-diffusion-webui/wiki/Installation-on-Intel-Silicon) (External Wiki)
*   [Ascend NPUs](https://github.com/wangshuai09/stable-diffusion-webui/wiki/Install-and-run-on-Ascend-NPUs) (External Wiki)

**Alternative:** [Online Services](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Online-Services) (e.g., Google Colab)

### Windows Installation (NVidia GPUs - Recommended)

1.  Download `sd.webui.zip` from [v1.0.0-pre](https://github.com/AUTOMATIC1111/stable-diffusion-webui/releases/tag/v1.0.0-pre) and extract.
2.  Run `update.bat`.
3.  Run `run.bat`.

### Automatic Installation on Windows

1.  Install [Python 3.10.6](https://www.python.org/downloads/release/python-3106/) with "Add Python to PATH" selected.
2.  Install [git](https://git-scm.com/download/win).
3.  Run `git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git`.
4.  Run `webui-user.bat` from Windows Explorer.

### Automatic Installation on Linux

1.  Install dependencies (Debian/Ubuntu, Red Hat, openSUSE, Arch) via the instructions in the original repo.
2.  Navigate to the desired installation directory.
3.  Run `wget -q https://raw.githubusercontent.com/AUTOMATIC1111/stable-diffusion-webui/master/webui.sh` or clone the repository.
4.  Run `webui.sh`.
5.  Customize with `webui-user.sh`.

### Installation on Apple Silicon

Find the specific instructions [here](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Installation-on-Apple-Silicon).

## Contributing

See the [Contributing guidelines](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Contributing) for information on how to contribute.

## Documentation

The documentation is hosted on the project's [wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki). For indexing purposes, a crawlable version of the wiki is available [here](https://github-wiki-see.page/m/AUTOMATIC1111/stable-diffusion-webui/wiki).

## Credits

A full list of credits, including licenses for borrowed code, is available in the `Settings -> Licenses` screen and the `html/licenses.html` file.