# ComfyUI-RMBG: Effortlessly Remove Backgrounds and Segment Images in ComfyUI

**Enhance your image editing workflow with ComfyUI-RMBG, a powerful custom node for ComfyUI that leverages advanced AI models to remove backgrounds and segment images with precision.**  [Explore the original repository](https://github.com/1038lab/ComfyUI-RMBG) for more details.

## Key Features:

*   **Advanced Background Removal:** Utilize a range of models, including RMBG-2.0, INSPYRENET, BEN, and BEN2, for accurate background removal.
*   **Precise Object Segmentation:** Segment objects using text prompts with SAM and GroundingDINO models, offering both tag-style and natural language inputs.
*   **SAM2 Segmentation:** Leverage the latest SAM2 models for text-prompted segmentation.
*   **Face, Fashion, and Clothes Segmentation:** Dedicated nodes for segmenting faces, clothing items, and fashion accessories.
*   **Flexible Output Options:**  Output images with transparent, black, white, green, blue, or red backgrounds.
*   **Real-time background replacement and edge detection**
*   **Image and Mask Tools:** Various nodes to combine, edit, and convert images and masks

## Recent Updates:

*   **v2.9.0 (2025/08/18):** Added `SDMatte Matting` node
*   **v2.8.0 (2025/08/11):** Added `SAM2Segment` node for text-prompted segmentation. Enhanced color widget support across all nodes.
*   **v2.7.1 (2025/08/06):** Enhanced LoadImage nodes, redesigned ImageStitch node and fixed background color handling issues.
*   **v2.6.0 (2025/07/15):** Added `Kontext Refence latent Mask` node
*   **v2.5.2 (2025/07/11):** 
*   **v2.5.1 (2025/07/07):** 
*   **v2.5.0 (2025/07/01):** Added `MaskOverlay`, `ObjectRemover`, `ImageMaskResize` new nodes. Added 2 BiRefNet models: `BiRefNet_lite-matting` and `BiRefNet_dynamic`. Added batch image support for `Segment_v1` and `Segment_V2` nodes
*   **v2.4.0 (2025/06/01):** Added `CropObject`, `ImageCompare`, `ColorInput` nodes and new Segment V2 (see update.md for details)
*   **v2.3.2 (2025/05/15):** 
*   **v2.3.1 (2025/05/02):** 
*   **v2.3.0 (2025/05/01):** Added new nodes: IC-LoRA Concat, Image Crop, resizing options for Load Image.
*   **v2.2.1 (2025/04/05):** 
*   **v2.2.0 (2025/04/05):** Added new nodes: Image Combiner, Image Stitch, Image/Mask Converter, Mask Enhancer, Mask Combiner, and Mask Extractor, fixed compatibility issues with transformers v4.49+, i18n translation errors, and mask image output to segment nodes.
*   **v2.1.1 (2025/03/21):** Enhanced compatibility with Transformers
*   **v2.1.0 (2025/03/19):** Integrated internationalization (i18n) support for multiple languages.
*   **v2.0.0 (2025/03/13):** Added Image and Mask Tools, enhanced code structure and documentation for better usability. Introduced a new category path: `🧪AILab/🛠️UTIL/🖼️IMAGE`.
*   **v1.9.3 (2025/02/24):** Clean up the code and fix the issue
*   **v1.9.2 (2025/02/21):**  Added new foreground refinement feature for better transparency handling, improved edge quality and detail preservation, and enhanced memory optimization.
*   **v1.9.1 (2025/02/20):** Changed repository for model management to the new repository and Reorganized models files structure for better maintainability.
*   **v1.9.0 (2025/02/19):** Enhanced BiRefNet model performance and stability, and improved memory management for large images
*   **v1.8.0 (2025/02/07):** Added a new custom node for BiRefNet-HR model and support high resolution image processing (up to 2048x2048)
*   **v1.7.0 (2025/02/04):** Added a new custom node for BEN2 model.
*   **v1.6.0 (2025/01/22):** Added a new custom node for face parsing and segmentation
*   **v1.5.0 (2025/01/05):** Added a new custom node for fashion segmentation.
*   **v1.4.0 (2025/01/02):** Added intelligent clothes segmentation.
*   **v1.3.2 (2024/12/29):** Enhanced background handling.
*   **v1.3.1 (2024/12/25):** Bug fixes.
*   **v1.3.0 (2024/12/23):** Added text-prompted object segmentation.
*   **v1.2.2 (2024/12/12):** Bug fixes.
*   **v1.2.1 (2024/12/02):** Bug fixes.
*   **v1.2.0 (2024/11/29):** Bug fixes.
*   **v1.1.0 (2024/11/21):** Bug fixes.

## Installation:

Choose your preferred installation method:

**1. Using ComfyUI Manager (Recommended):**

   *   Open ComfyUI Manager and search for `Comfyui-RMBG`.
   *   Click "Install".
   *   Install `requirements.txt` in the ComfyUI-RMBG folder:
        ```bash
        ./ComfyUI/python_embeded/python -m pip install -r requirements.txt
        ```
   *   Note: If you are having issues with your environment you can use ComfyUI's embedded python.

**2. Manual Installation:**

   *   Navigate to your ComfyUI `custom_nodes` directory:
        ```bash
        cd ComfyUI/custom_nodes
        ```
   *   Clone the repository:
        ```bash
        git clone https://github.com/1038lab/ComfyUI-RMBG
        ```
   *   Install dependencies:
        ```bash
        ./ComfyUI/python_embeded/python -m pip install -r requirements.txt
        ```

**3. Using Comfy CLI:**

   *   Ensure `pip install comfy-cli` is installed.
   *   Install ComfyUI using `comfy install` (if not already installed).
   *   Install ComfyUI-RMBG:
        ```bash
        comfy node install ComfyUI-RMBG
        ```
   *   Install dependencies:
        ```bash
        ./ComfyUI/python_embeded/python -m pip install -r requirements.txt
        ```

## Model Downloads:

*The models will auto-download on first use to `/ComfyUI/models/RMBG/` or `/ComfyUI/models/SAM/`. You can also manually download them from Hugging Face if needed.*

*   **RMBG-2.0:** [https://huggingface.co/1038lab/RMBG-2.0](https://huggingface.co/1038lab/RMBG-2.0)
*   **INSPYRENET:** [https://huggingface.co/1038lab/inspyrenet](https://huggingface.co/1038lab/inspyrenet)
*   **BEN:** [https://huggingface.co/1038lab/BEN](https://huggingface.co/1038lab/BEN)
*   **BEN2:** [https://huggingface.co/1038lab/BEN2](https://huggingface.co/1038lab/BEN2)
*   **BiRefNet-HR:** [https://huggingface.co/1038lab/BiRefNet_HR](https://huggingface.co/1038lab/BiRefNet_HR)
*   **SAM:** [https://huggingface.co/1038lab/sam](https://huggingface.co/1038lab/sam)
*   **SAM2:** [https://huggingface.co/1038lab/sam2](https://huggingface.co/1038lab/sam2)
*   **GroundingDINO:** [https://huggingface.co/1038lab/GroundingDINO](https://huggingface.co/1038lab/GroundingDINO)
*   **Clothes Segment:** [https://huggingface.co/1038lab/segformer_clothes](https://huggingface.co/1038lab/segformer_clothes)
*   **Fashion Segment:** [https://huggingface.co/1038lab/segformer_fashion](https://huggingface.co/1038lab/segformer_fashion)
*   **BiRefNet Models:** [https://huggingface.co/1038lab/BiRefNet](https://huggingface.co/1038lab/BiRefNet)
*   **SDMatte models:** [https://huggingface.co/1038lab/SDMatte](https://huggingface.co/1038lab/SDMatte)

## Usage Guide:

### RMBG Node

1.  Load the `RMBG (Remove Background)` node from the `🧪AILab/🧽RMBG` category.
2.  Connect an image to the input.
3.  Select a model.
4.  Get two outputs:
    *   `IMAGE`: Processed image with a transparent, black, white, green, blue, or red background.
    *   `MASK`: A binary mask of the foreground.

### Segment Node

1.  Load the `Segment (RMBG)` node from the `🧪AILab/🧽RMBG` category.
2.  Connect an image.
3.  Enter a text prompt (tag-style or natural language).
4.  Select the SAM or GroundingDINO models.
5.  Adjust parameters as needed.

## Optional Settings:

| Parameter            | Description                                                                   | Tips                                                                                              |
| -------------------- | ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| `Sensitivity`        | Adjusts mask detection strength.                                             | Higher values for stricter detection; adjust based on image complexity.                           |
| `Processing Resolution` | Controls image processing resolution.                                               | Choose between 256 and 2048; higher resolutions offer more detail but increase memory usage.       |
| `Mask Blur`          | Blurs mask edges to reduce jaggedness.                                       | Default is 0; try values between 1 and 5 for smoother edges.                                        |
| `Mask Offset`        | Expands or shrinks the mask boundary.                                           | Default is 0; typically fine-tune between -10 and 10.                                            |
| `Background`        | Choose output background color        | Alpha (transparent background) Black, White, Green, Blue, Red |
| `Invert Output`      | Flip mask and image output         | Invert both image and mask output |
| `Refine Foreground` | Use Fast Foreground Color Estimation to optimize transparent background        | Enable for better edge quality and transparency handling |
| `Performance Optimization` | Properly setting options can enhance performance when processing multiple images. | If memory allows, consider increasing `process_res` and `mask_blur` values for better results, but be mindful of memory usage. |

## About Models:

<details>
<summary>Click to expand and learn more about each model</summary>

### RMBG-2.0

*   Developed by BRIA AI, using BiRefNet architecture.
*   High accuracy, precise edge detection, and excellent handling of fine details.
*   Supports multiple objects and batch processing.

### INSPYRENET

*   Specialized in human portrait segmentation.
*   Fast processing and good edge detection.
*   Ideal for portrait photos and human subjects.

### BEN

*   Robust on various image types.
*   Good balance between speed and accuracy.

### BEN2

*   Improved accuracy and speed compared to BEN.
*   Better handling of complex scenes.

### BIREFNET MODELS
*   General purpose matting and high resolution models

### SAM

*   A powerful model for object detection and segmentation.
*   High accuracy, precise edge detection, excellent handling of fine details, and supports multiple objects.
*   Output with background.
*   Batch output for video.

### SAM2

*   The latest segmentation model family designed for efficient, high-quality text-prompted segmentation.
*   Multiple sizes: Tiny, Small, Base+, Large
*   Optimized inference with strong accuracy
*   Automatic download on first use; manual placement supported in `ComfyUI/models/sam2`

### GroundingDINO

*   Model for text-prompted object detection and segmentation.

### BiRefNet Models

*   BiRefNet-general purpose model (balanced performance)
*   BiRefNet_512x512 model (optimized for 512x512 resolution)
*   BiRefNet-portrait model (optimized for portrait/human matting)
*   BiRefNet-matting model (general purpose matting)
*   BiRefNet-HR model (high resolution up to 2560x2560)
*   BiRefNet-HR-matting model (high resolution matting)
*   BiRefNet_lite model (lightweight version for faster processing)
*   BiRefNet_lite-2K model (lightweight version for 2K resolution)
</details>

## Requirements:

*   ComfyUI
*   Python 3.10+
*   Required packages (automatically installed):
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
*   Auto-download on first run to `models/RMBG/SDMatte/`
*   If network restricted, place weights manually:
  * `models/RMBG/SDMatte/SDMatte.safetensors` (standard) or `SDMatte_plus.safetensors` (plus)
  * Components (config files) are auto-downloaded; if needed, mirror the structure from the Hugging Face repo to `models/RMBG/SDMatte/` (`scheduler/`, `text_encoder/`, `tokenizer/`, `unet/`, `vae/`)

## Troubleshooting:

*   **401 error when initializing GroundingDINO / missing `models/sam2`:**  Delete `%USERPROFILE%\.cache\huggingface\token` and `%USERPROFILE%\.huggingface\token` (if present). Ensure no `HF_TOKEN`/`HUGGINGFACE_TOKEN` environment variables are set. Re-run; public repos download anonymously.
*   **Preview shows "Required input is missing: images":** Ensure image outputs are connected and upstream nodes ran successfully.

## Credits:

*   RMBG-2.0: [https://huggingface.co/briaai/RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0)
*   INSPYRENET: [https://github.com/plemeri/InSPyReNet](https://github.com/plemeri/InSPyReNet)
*   BEN: [https://huggingface.co/PramaLLC/BEN](https://huggingface.co/PramaLLC/BEN)
*   BEN2: [https://huggingface.co/PramaLLC/BEN2](https://huggingface.co/PramaLLC/BEN2)
*   BiRefNet: [https://huggingface.co/ZhengPeng7](https://huggingface.co/ZhengPeng7)
*   SAM: [https://huggingface.co/facebook/sam-vit-base](https://huggingface.co/facebook/sam-vit-base)
*   GroundingDINO: [https://github.com/IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
*   Clothes Segment: [https://huggingface.co/mattmdjaga/segformer_b2_clothes](https://huggingface.co/mattmdjaga/segformer_b2_clothes)
*   SDMatte: [https://github.com/vivoCameraResearch/SDMatte](https://github.com/vivoCameraResearch/SDMatte)

*   Created by: [AILab](https://github.com/1038lab)

## Star History

<a href="https://www.star-history.com/#1038lab/comfyui-rmbg&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=1038lab/comfyui-rmbg&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=1038lab/comfyui-rmbg&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=1038lab/comfyui-rmbg&type=Date" />
 </picture>
</a>

**If you find ComfyUI-RMBG helpful, please consider giving the repository a star!**

## License

GPL-3.0 License