# Stable Diffusion Web UI: Unleash Your Creativity with AI-Powered Image Generation

[Link to Original Repository: AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)

This powerful web interface, built with Gradio, empowers you to generate stunning images from text prompts using Stable Diffusion. With a vast array of features and a user-friendly interface, you can bring your creative visions to life with ease.

## Key Features

*   **Text-to-Image & Image-to-Image Generation:** Generate images from text prompts or modify existing images.
*   **One-Click Installation:** Get up and running quickly with a simple installation script (Python and Git required).
*   **Advanced Image Editing:** Utilize features like outpainting, inpainting, color sketch, and prompt editing for precise control over your creations.
*   **Prompt Optimization:** Fine-tune your prompts with attention mechanisms, negative prompts, and style saving for optimal results.
*   **Upscaling and Enhancement:** Enhance image quality with integrated tools like GFPGAN, CodeFormer, RealESRGAN, ESRGAN, and more.
*   **Flexible Generation Parameters:** Save and load generation parameters, utilize seed resizing, and explore X/Y/Z plots for experimentation.
*   **Customization & Extensions:** Extend functionality with custom scripts, textual inversion, hypernetworks, LORAs, and a wide range of community-developed extensions.
*   **Batch Processing:** Efficiently process multiple images using the img2img feature.
*   **Performance and Speed:** Xformers support and other optimizations for faster image generation.
*   **API Integration:** Access the web UI's functionality through an API for integration into other applications.

## Installation and Running

### Prerequisites

*   Python (3.10.6 recommended, newer versions may have compatibility issues)
*   Git

Detailed installation instructions are available for:

*   NVIDIA GPUs ([Installation Guide](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-NVidia-GPUs))
*   AMD GPUs ([Installation Guide](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-AMD-GPUs))
*   Intel CPUs and GPUs ([Installation Guide](https://github.com/openvinotoolkit/stable-diffusion-webui/wiki/Installation-on-Intel-Silicon))
*   Apple Silicon ([Installation Guide](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Installation-on-Apple-Silicon))

Alternatively, use online services for easy access ([List of Online Services](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Online-Services)).

### Basic Installation Steps (Windows with NVIDIA GPU)

1.  Download the `sd.webui.zip` release package from the releases page.
2.  Run `update.bat`.
3.  Run `run.bat`.

### Automatic Installation (Windows)

1.  Install Python 3.10.6 and Git, ensuring Python is added to your PATH.
2.  Clone the repository: `git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git`
3.  Run `webui-user.bat`.

### Automatic Installation (Linux)

1.  Install dependencies (Debian/Ubuntu: `sudo apt install wget git python3 python3-venv libgl1 libglib2.0-0`; Red Hat: `sudo dnf install wget git python3 gperftools-libs libglvnd-glx`; Arch: `sudo pacman -S wget git python3`).
2.  Navigate to your desired install directory and run `wget -q https://raw.githubusercontent.com/AUTOMATIC1111/stable-diffusion-webui/master/webui.sh` or clone the repo with `git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui`.
3.  Run `./webui.sh`.

## Contributing

Contribute to the project through [these guidelines](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Contributing).

## Documentation

For in-depth information, refer to the project's [wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki).