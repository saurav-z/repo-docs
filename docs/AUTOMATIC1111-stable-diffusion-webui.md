# Stable Diffusion WebUI: Unleash Your Creativity with AI-Powered Image Generation

**Create stunning images from text prompts with the Stable Diffusion web UI, a powerful and user-friendly interface built with Gradio.** Explore a vast array of features, customizations, and extensions to bring your creative visions to life.  [Explore the original repo](https://github.com/AUTOMATIC1111/stable-diffusion-webui) for the latest updates and community contributions.

## Key Features:

*   **Core Functionality:**
    *   txt2img and img2img modes for generating images from text or transforming existing ones.
    *   Outpainting and inpainting capabilities to extend and refine images.
*   **Advanced Image Manipulation:**
    *   Prompt Matrix and X/Y/Z plot for exploring variations and parameter combinations.
    *   Attention mechanisms for fine-tuning prompt emphasis.
    *   Negative prompts to specify unwanted elements in the generated image.
    *   Loopback processing for iterative image refinement.
*   **Upscaling and Enhancement:**
    *   Integrated face restoration with GFPGAN and CodeFormer.
    *   Multiple upscaling options: RealESRGAN, ESRGAN, SwinIR, Swin2SR, and LDSR.
*   **Customization & Control:**
    *   Extensive settings for sampling methods, aspect ratios, and more.
    *   Support for textual inversion, hypernetworks, LoRAs, and embeddings to personalize your creations.
    *   Seed control and resizing options for image variations.
    *   Styles for saving and applying prompt elements.
    *   Tiling support for seamless texture generation.
*   **Workflow Enhancements:**
    *   Generation parameters are automatically saved with images.
    *   Drag-and-drop functionality for easy parameter loading.
    *   CLIP interrogator for reverse image prompting.
    *   Prompt editing during generation.
    *   Batch processing for efficient image generation workflows.
    *   Checkpoint merging for combining models.
*   **Advanced Features (Extensions & Integrations):**
    *   Custom scripts and community extensions.
    *   Composable Diffusion for complex prompts.
    *   DeepDanbooru integration for anime prompts.
    *   xformers for performance optimization.
    *   API for programmatic access.
    *   Support for Stable Diffusion 2.0 and Alt-Diffusion models.
    *   Segmind Stable Diffusion support.
*   **Training:**
    *   Training tab to train hypernetworks and embeddings
*   **And so much more:**  This WebUI is constantly being updated with new features!

## Installation

Detailed installation instructions are available in the project's [wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki), with specific guides for:

*   NVidia GPUs (Recommended)
*   AMD GPUs
*   Intel CPUs and GPUs
*   Apple Silicon
*   Ascend NPUs
*   And more

### Quick Start (Windows with NVidia GPU):

1.  Download the `sd.webui.zip` release package.
2.  Run `update.bat`.
3.  Run `run.bat`.

### Automatic Installation (Windows):

1.  Install Python 3.10.6 and Git.
2.  Clone the repository: `git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git`.
3.  Run `webui-user.bat`.

### Automatic Installation (Linux):

1.  Install dependencies using your package manager (e.g., `apt`, `dnf`).
2.  Run: `wget -q https://raw.githubusercontent.com/AUTOMATIC1111/stable-diffusion-webui/master/webui.sh` or clone the repo.
3.  Run `./webui.sh`.

## Contributing

Contribute to the project by following the guidelines outlined in the [Contributing](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Contributing) section of the wiki.

## Documentation

Comprehensive documentation, including detailed explanations of features and usage, is available on the project's [wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki).  For search engine indexing, see the [crawlable wiki](https://github-wiki-see.page/m/AUTOMATIC1111/stable-diffusion-webui/wiki).

## Credits

The Stable Diffusion WebUI leverages numerous open-source projects.  A detailed list of credits and licenses can be found within the application's settings (`Settings -> Licenses`) and the `html/licenses.html` file.