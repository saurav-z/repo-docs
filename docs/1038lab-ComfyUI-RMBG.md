<!-- SEO-optimized README for ComfyUI-RMBG -->

# ComfyUI-RMBG: Advanced Image Background Removal and Segmentation

**Effortlessly remove backgrounds and segment objects in your images with ComfyUI-RMBG, a powerful custom node for ComfyUI.**  ([Back to the original repository](https://github.com/1038lab/ComfyUI-RMBG))

## Key Features

*   **Comprehensive Background Removal:** Utilize advanced models like RMBG-2.0, INSPYRENET, BEN, and BEN2 for precise background removal.
*   **Precise Object Segmentation:** Leverage text prompts with the Segment node, using models like SAM and GroundingDINO to accurately isolate objects.
*   **Cutting-Edge SAM2 Segmentation:** Employ the latest SAM2 models for advanced, text-prompted segmentation with options for Tiny, Small, Base+ and Large models.
*   **Face, Clothes and Fashion Segmentation:** Use custom nodes to segment faces, clothes, and fashion elements with specialized models
*   **High-Resolution Support:** Process high-resolution images, up to 2048x2048 pixels.
*   **Enhanced Edge Detection:** Experience improved accuracy with real-time background replacement and enhanced edge detection capabilities.
*   **Flexible Background Options:** Choose from a variety of background colors or maintain transparency with the alpha channel.

## What's New

*   **v2.9.0** (2025/08/18): Added `SDMatte Matting` node
*   **v2.8.0** (2025/08/11): Added `SAM2Segment` node; Enhanced color widget support
*   **v2.7.1** (2025/08/06): Enhanced LoadImage nodes; Redesigned ImageStitch node; Fixed background color handling issues
*   **v2.6.0** (2025/07/15): Added `Kontext Refence latent Mask` node
*   **v2.5.2 - v2.5.0** (2025/07/15 - 2025/07/01): Added new nodes (MaskOverlay, ObjectRemover, ImageMaskResize), BiRefNet models and batch image support.
*   **v2.4.0** (2025/06/01): Added new nodes (CropObject, ImageCompare, ColorInput) and new Segment V2
*   **v2.3.2 - v2.3.0** (2025/05/15 - 2025/05/01): Added new nodes (IC-LoRA Concat, Image Crop) and new resizing options.
*   **v2.2.1 - v2.2.0** (2025/04/05): Added new nodes (Image Combiner, Image Stitch, Image/Mask Converter, Mask Enhancer, Mask Combiner, and Mask Extractor) and fixed compatibility issues.
*   **v2.1.1 - v2.1.0** (2025/03/21-2025/03/19): Enhanced compatibility with Transformers, i18n support.
*   **v2.0.0** (2025/03/13): Added Image and Mask Tools improved functionality.
*   **v1.9.3 - v1.9.0** (2025/02/24-2025/02/19): Code cleanup, Fast Foreground Color Estimation, BiRefNet model improvements.
*   **v1.8.0** (2025/02/07): Added new BiRefNet-HR model.
*   **v1.7.0** (2025/02/04): Added a new custom node for BEN2 model.
*   **v1.6.0** (2025/01/22): Added a new custom node for face parsing and segmentation
*   **v1.5.0** (2025/01/05): Added a new custom node for fashion segmentation.
*   **v1.4.0** (2025/01/02): Added intelligent clothes segmentation with 18 different categories
*   **v1.3.2 - v1.2.0** (2024/12/29 - 2024/11/29): Enhanced background handling, bug fixes and new Segment node.

## Installation

### 1. ComfyUI Manager (Recommended)

1.  Open ComfyUI Manager.
2.  Search for `ComfyUI-RMBG` and install.
3.  Install requirements.txt in the ComfyUI-RMBG folder:
    ```bash
    ./ComfyUI/python_embeded/python -m pip install -r requirements.txt
    ```

### 2. Manual Installation

1.  Navigate to your ComfyUI `custom_nodes` directory:
    ```bash
    cd ComfyUI/custom_nodes
    ```
2.  Clone the repository:
    ```bash
    git clone https://github.com/1038lab/ComfyUI-RMBG
    ```
3.  Install required packages:
    ```bash
    ./ComfyUI/python_embeded/python -m pip install -r requirements.txt
    ```

### 3. Comfy CLI

1.  Install Comfy CLI: `pip install comfy-cli`
2.  Install the node: `comfy node install ComfyUI-RMBG`
3.  Install requirements.txt in the ComfyUI-RMBG folder:
    ```bash
    ./ComfyUI/python_embeded/python -m pip install -r requirements.txt
    ```

## Model Downloads

*   **Automatic Download:** Models are automatically downloaded to the `ComfyUI/models/RMBG/` and relevant directories upon first use.
*   **Manual Download:** If needed, download models from the links below and place them in their respective folders within the `ComfyUI/models/` directory.

    *   RMBG-2.0: [Hugging Face](https://huggingface.co/1038lab/RMBG-2.0)
    *   INSPYRENET: [Hugging Face](https://huggingface.co/1038lab/inspyrenet)
    *   BEN: [Hugging Face](https://huggingface.co/1038lab/BEN)
    *   BEN2: [Hugging Face](https://huggingface.co/1038lab/BEN2)
    *   BiRefNet-HR: [Hugging Face](https://huggingface.co/1038lab/BiRefNet_HR)
    *   SAM: [Hugging Face](https://huggingface.co/1038lab/sam)
    *   SAM2: [Hugging Face](https://huggingface.co/1038lab/sam2)
    *   GroundingDINO: [Hugging Face](https://huggingface.co/1038lab/GroundingDINO)
    *   Clothes Segment: [Hugging Face](https://huggingface.co/1038lab/segformer_clothes)
    *   Fashion Segment: [Hugging Face](https://huggingface.co/1038lab/segformer_fashion)
    *   BiRefNet Models: [Hugging Face](https://huggingface.co/1038lab/BiRefNet)
    *   SDMatte: [Hugging Face](https://huggingface.co/1038lab/SDMatte)

## Usage

### RMBG Node

1.  Load the `RMBG (Remove Background)` node from the `🧪AILab/🧽RMBG` category.
2.  Connect an image to the input.
3.  Select a model from the dropdown.
4.  Adjust parameters (optional, see below).
5.  Get the processed image (with background) and the mask output.

### Optional Settings

| Setting                | Description                                                                                                                            | Tips                                                                                                                                       |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| **Sensitivity**        | Adjusts mask detection strength; higher values for stricter detection.                                                               | Start with 0.5, adjust for image complexity.                                                                                               |
| **Processing Resolution** | Controls image processing resolution (detail vs. memory).                                                                                | Choose between 256 and 2048, default is 1024. Higher resolution = better detail, more memory.                                              |
| **Mask Blur**          | Blurs mask edges to reduce jaggedness.                                                                                                   | Default 0, try 1-5 for smoother edges.                                                                                                     |
| **Mask Offset**        | Expands/shrinks mask boundary.                                                                                                            | Default 0, typically fine-tune between -10 and 10.                                                                                          |
| **Background**          | Select the output background color.                                                                                                    | Choose from Alpha (transparent), Black, White, Green, Blue, or Red.                                                                      |
| **Invert Output**      | Flip mask and image output.                                                                                                             | Invert both image and mask output.                                                                                                         |
| **Refine Foreground**  | Enable Fast Foreground Color Estimation for better transparency.                                                                        | Enable for better edge quality and transparency handling.                                                                                  |
| **Performance Optimization** | Properly setting options can enhance performance when processing multiple images.                                                     | If memory allows, consider increasing `process_res` and `mask_blur` values for better results, but be mindful of memory usage.                  |

### Segment Node

1.  Load the `Segment (RMBG)` node from the `🧪AILab/🧽RMBG` category.
2.  Connect an image to the input.
3.  Enter a text prompt (tag-style or natural language).
4.  Select SAM or GroundingDINO models.
5.  Adjust parameters (Threshold, Mask Blur, Mask Offset, Background Color) as needed.

## About Models

<details>
<summary><h2>Detailed Model Information</h2></summary>

### RMBG-2.0
Developed by BRIA AI, RMBG-2.0 utilizes the BiRefNet architecture for:
-   High accuracy in complex environments
-   Precise edge detection and preservation
-   Excellent handling of fine details
-   Support for multiple objects in a single image
-   Output Comparison
-   Output with background
-   Batch output for video

Trained on a diverse dataset of 15,000+ high-quality images.

### INSPYRENET
Specialized for human portrait segmentation:
-   Fast processing
-   Good edge detection
-   Ideal for portraits and human subjects

### BEN & BEN2
-   Robust on various image types
-   Good balance between speed and accuracy (BEN)
-   Improved accuracy, speed, and handling of complex scenes (BEN2)
-   Suitable for batch processing

### BIREFNET MODELS
-   BiRefNet-general purpose model (balanced performance)
-   BiRefNet_512x512 model (optimized for 512x512 resolution)
-   BiRefNet-portrait model (optimized for portrait/human matting)
-   BiRefNet-matting model (general purpose matting)
-   BiRefNet-HR model (high resolution up to 2560x2560)
-   BiRefNet-HR-matting model (high resolution matting)
-   BiRefNet_lite model (lightweight version for faster processing)
-   BiRefNet_lite-2K model (lightweight version for 2K resolution)

### SAM
State-of-the-art for object detection and segmentation:
-   High accuracy
-   Precise edge detection
-   Handles fine details
-   Supports multiple objects
-   Output Comparison
-   Output with background
-   Batch output for video

### SAM2
Latest segmentation model family:
-   Multiple sizes (Tiny, Small, Base+, Large)
-   Optimized inference
-   Automatic or manual model download

### GroundingDINO
Text-prompted object detection:
-   High accuracy
-   Precise edge detection
-   Handles fine details
-   Supports multiple objects
-   Output Comparison
-   Output with background
-   Batch output for video

### BiRefNet Models
- BiRefNet-general purpose model (balanced performance)
- BiRefNet_512x512 model (optimized for 512x512 resolution)
- BiRefNet-portrait model (optimized for portrait/human matting)
- BiRefNet-matting model (general purpose matting)
- BiRefNet-HR model (high resolution up to 2560x2560)
- BiRefNet-HR-matting model (high resolution matting)
- BiRefNet_lite model (lightweight version for faster processing)
- BiRefNet_lite-2K model (lightweight version for 2K resolution)
</details>

## Requirements

*   ComfyUI
*   Python 3.10+
*   Automatically installed packages:
    *   huggingface-hub>=0.19.0
    *   transparent-background>=1.1.2
    *   segment-anything>=1.0
    *   groundingdino-py>=0.4.0
    *   opencv-python>=4.7.0
    *   onnxruntime>=1.15.0
    *   onnxruntime-gpu>=1.15.0
    *   protobuf>=3.20.2,<6.0.0
    *   hydra-core>=1.3.0
    *   omegaconf>=2.3.0
    *   iopath>=0.1.9

### SDMatte models (manual download)

- Auto-download on first run to `models/RMBG/SDMatte/`
- If network restricted, place weights manually:
  - `models/RMBG/SDMatte/SDMatte.safetensors` (standard) or `SDMatte_plus.safetensors` (plus)
  - Components (config files) are auto-downloaded; if needed, mirror the structure from the Hugging Face repo to `models/RMBG/SDMatte/` (`scheduler/`, `text_encoder/`, `tokenizer/`, `unet/`, `vae/`)

## Troubleshooting

*   **401 error initializing GroundingDINO / missing models:** Delete `%USERPROFILE%\.cache\huggingface\token` and `%USERPROFILE%\.huggingface\token` if present. Ensure no `HF_TOKEN`/`HUGGINGFACE_TOKEN` environment variables are set. Re-run; public repos download anonymously (no login required).
*   **"Required input is missing: images" error:** Ensure image outputs are connected and upstream nodes ran successfully.

## Credits

*   RMBG-2.0: [https://huggingface.co/briaai/RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0)
*   INSPYRENET: [https://github.com/plemeri/InSPyReNet](https://github.com/plemeri/InSPyReNet)
*   BEN: [https://huggingface.co/PramaLLC/BEN](https://huggingface.co/PramaLLC/BEN)
*   BEN2: [https://huggingface.co/PramaLLC/BEN2](https://huggingface.co/PramaLLC/BEN2)
*   BiRefNet: [https://huggingface.co/ZhengPeng7](https://huggingface.co/ZhengPeng7)
*   SAM: [https://huggingface.co/facebook/sam-vit-base](https://huggingface.co/facebook/sam-vit-base)
*   GroundingDINO: [https://github.com/IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
*   Clothes Segment: [https://huggingface.co/mattmdjaga/segformer_b2_clothes](https://huggingface.co/mattmdjaga/segformer_b2_clothes)
*   SDMatte: [https://github.com/vivoCameraResearch/SDMatte](https://github.com/vivoCameraResearch/SDMatte)
*   Fashion Segment: [Hugging Face](https://huggingface.co/1038lab/segformer_fashion)
*   Created by: [AILab](https://github.com/1038lab)

## Star History

```html
<a href="https://www.star-history.com/#1038lab/comfyui-rmbg&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=1038lab/comfyui-rmbg&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=1038lab/comfyui-rmbg&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=1038lab/comfyui-rmbg&type=Date" />
 </picture>
</a>
```

## License

GPL-3.0 License