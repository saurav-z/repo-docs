# Enhance Your Images with ComfyUI-RMBG: AI-Powered Background Removal and Segmentation

Effortlessly remove backgrounds, segment objects, and refine images with the powerful ComfyUI-RMBG custom node, bringing advanced AI capabilities directly to your workflow.  [View the original repo on GitHub](https://github.com/1038lab/ComfyUI-RMBG).

## Key Features

*   **Advanced Background Removal:**
    *   Utilizes models like RMBG-2.0, INSPYRENET, BEN, BEN2, BiRefNet, SDMatte, SAM, and SAM2 for precise background removal.
    *   Offers real-time background replacement options.
    *   Includes enhanced edge detection for refined results.
    *   Batch processing support.

*   **Intelligent Object Segmentation:**
    *   Text-prompted object detection using tag-style or natural language inputs.
    *   Leverages SAM and GroundingDINO models for accurate segmentation.
    *   Adjustable parameters for thresholding, mask refinement, and background color.

*   **SAM2 Segmentation:**
    *   Text-prompted segmentation with the latest Facebook Research SAM2 (Tiny/Small/Base+/Large) models.
    *   Automatic model download, or manual placement support in `ComfyUI/models/sam2`.

*   **Specialized Segmentation:**
    *   Face parsing and segmentation with multiple facial feature categories.
    *   Fashion segmentation.
    *   Clothes segmentation.

## What's New

*   **v2.9.1 (2025/09/12):**  Update ComfyUI-RMBG to v2.9.1 ([update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v291-20250912)).
*   **v2.9.0 (2025/08/18):** Update ComfyUI-RMBG to v2.9.0 ([update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v290-20250818))
    *   Added `SDMatte Matting` node
*   **v2.8.0 (2025/08/11):** Update ComfyUI-RMBG to v2.8.0 ([update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v280-20250811))
    *   Added `SAM2Segment` node for text-prompted segmentation.
    *   Enhanced color widget support.
*   **v2.7.1 (2025/08/06):** Update ComfyUI-RMBG to v2.7.1 ([update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v271-20250806))
    *   Enhanced LoadImage node
    *   Redesigned ImageStitch node
    *   Fixed background color handling issues
*   **v2.6.0 (2025/07/15):** Update ComfyUI-RMBG to v2.6.0 ([update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v260-20250715))
    *   Added `Kontext Refence latent Mask` node
*   **v2.5.2, v2.5.1, v2.5.0 (2025/07/15, 07/11, 07/07):**
    *   Added new nodes: `MaskOverlay`, `ObjectRemover`, `ImageMaskResize`.
    *   Added 2 BiRefNet models.
    *   Added batch image support.
*   **v2.4.0 (2025/06/01):**
    *   Added `CropObject`, `ImageCompare`, `ColorInput` nodes and new Segment V2.
*   **v2.3.2, v2.3.1, v2.3.0 (2025/05/15, 05/02, 05/01):**
    *   Added new nodes: IC-LoRA Concat, Image Crop.
    *   Added resizing options.
*   **v2.2.1, v2.2.0 (2025/04/05):**
    *   Added new nodes: Image Combiner, Image Stitch, Image/Mask Converter, Mask Enhancer, Mask Combiner, and Mask Extractor.
    *   Fixed compatibility issues with transformers v4.49+.
    *   Fixed i18n translation errors.
    *   Added mask image output to segment nodes.
*   **v2.1.1, v2.1.0 (2025/03/21, 03/19):**
    *   Enhanced compatibility with Transformers.
    *   Integrated internationalization (i18n) support for multiple languages.
*   **v2.0.0 (2025/03/13):**
    *   Added Image and Mask Tools improved functionality.
    *   Introduced a new category path: `🧪AILab/🛠️UTIL/🖼️IMAGE`.
*   **Earlier Updates:**  See [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md) for a full changelog.

## Installation

Choose your preferred installation method:

### 1. ComfyUI Manager
Install directly from the ComfyUI Manager by searching for `Comfyui-RMBG` and installing. Ensure you install the requirements using the method described below after installation is complete.

### 2. Manual Cloning

1.  Navigate to your ComfyUI custom_nodes directory: `cd ComfyUI/custom_nodes`
2.  Clone the repository: `git clone https://github.com/1038lab/ComfyUI-RMBG`
3.  Install required packages (Important!):
    ```bash
    ./ComfyUI/python_embeded/python -m pip install -r requirements.txt
    ```
    or if you encounter an error:
    ```bash
    python -m pip install --no-user --no-cache-dir -r requirements.txt
    ```

### 3. Comfy CLI

1.  Install Comfy CLI: `pip install comfy-cli` (if you don't have it installed)
2.  Install the node: `comfy node install ComfyUI-RMBG`
3.  Install required packages (Important!):
    ```bash
    ./ComfyUI/python_embeded/python -m pip install -r requirements.txt
    ```
    or if you encounter an error:
    ```bash
    python -m pip install --no-user --no-cache-dir -r requirements.txt
    ```

## Model Downloads (Important!)

*   Models are automatically downloaded upon first use to the `ComfyUI/models/RMBG/` or `ComfyUI/models/SAM` or `ComfyUI/models/sam2` or `ComfyUI/models/grounding-dino` directories.
*   For manual download and placement, follow the specific instructions below (or see the model links in the troubleshooting section).

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
    *   **BiRefNet:** [https://huggingface.co/1038lab/BiRefNet](https://huggingface.co/1038lab/BiRefNet)
    *   **SDMatte models:** [https://huggingface.co/1038lab/SDMatte](https://huggingface.co/1038lab/SDMatte)

## Usage

### RMBG Node

1.  Load the `RMBG (Remove Background)` node from the `🧪AILab/🧽RMBG` category.
2.  Connect an image to the input.
3.  Select a model.
4.  Adjust optional parameters as needed (See below).
5.  Get two outputs: IMAGE (processed image) and MASK (foreground mask).

### Optional Settings :bulb: Tips

| Setting                  | Description                                                      | Tip                                                                                               |
| ------------------------ | ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| Sensitivity              | Mask detection strength.  Higher = stricter.                    | Adjust based on image complexity (default: 0.5).                                                 |
| Processing Resolution | Image processing resolution. Affects detail & memory.           |  Values: 256-2048 (default: 1024). Higher = more detail, more memory.             |
| Mask Blur                | Smooths mask edges.                                               |  Set between 1-5 for smoother edges (default: 0).                                                 |
| Mask Offset              | Expands/shrinks the mask.                                         |  Fine-tune between -10 and 10 (default: 0).                                                         |
| Background               | Output background color.                                         |  Select from Alpha, Black, White, Green, Blue, or Red.                                                |
| Invert Output            | Flip mask and image output.                                         |                                                                                   |
| Refine Foreground      | Use Fast Foreground Color Estimation to optimize transparent background. |  Enable for better edge quality and transparency handling.                                                                   |
| Performance Optimization | Improves when processing multiple images. |  If memory allows, consider increasing `process_res` and `mask_blur` values for better results, but be mindful of memory usage.    |

### Segment Node

1.  Load the `Segment (RMBG)` node from the `🧪AILab/🧽RMBG` category.
2.  Connect an image.
3.  Enter a text prompt (tag-style or natural language).
4.  Select SAM and/or GroundingDINO models.
5.  Adjust parameters: Threshold (0.25-0.55), Mask blur & offset, Background.

<details>
<summary><h2>About Models</h2></summary>

## RMBG-2.0
RMBG-2.0 is is developed by BRIA AI and uses the BiRefNet architecture which includes:
- High accuracy in complex environments
- Precise edge detection and preservation
- Excellent handling of fine details
- Support for multiple objects in a single image
- Output Comparison
- Output with background
- Batch output for video
The model is trained on a diverse dataset of over 15,000 high-quality images, ensuring:
- Balanced representation across different image types
- High accuracy in various scenarios
- Robust performance with complex backgrounds

## INSPYRENET
INSPYRENET is specialized in human portrait segmentation, offering:
- Fast processing speed
- Good edge detection capability
- Ideal for portrait photos and human subjects

## BEN
BEN is robust on various image types, offering:
- Good balance between speed and accuracy
- Effective on both simple and complex scenes
- Suitable for batch processing

## BEN2
BEN2 is a more advanced version of BEN, offering:
- Improved accuracy and speed
- Better handling of complex scenes
- Support for more image types
- Suitable for batch processing

## BIREFNET MODELS
BIREFNET is a powerful model for image segmentation, offering:
- BiRefNet-general purpose model (balanced performance)
- BiRefNet_512x512 model (optimized for 512x512 resolution)
- BiRefNet-portrait model (optimized for portrait/human matting)
- BiRefNet-matting model (general purpose matting)
- BiRefNet-HR model (high resolution up to 2560x2560)
- BiRefNet-HR-matting model (high resolution matting)
- BiRefNet_lite model (lightweight version for faster processing)
- BiRefNet_lite-2K model (lightweight version for 2K resolution)
  
## SAM
SAM is a powerful model for object detection and segmentation, offering:
- High accuracy in complex environments
- Precise edge detection and preservation
- Excellent handling of fine details
- Support for multiple objects in a single image
- Output Comparison
- Output with background
- Batch output for video

## SAM2
SAM2 is the latest segmentation model family designed for efficient, high-quality text-prompted segmentation:
- Multiple sizes: Tiny, Small, Base+, Large
- Optimized inference with strong accuracy
- Automatic download on first use; manual placement supported in `ComfyUI/models/sam2`

## GroundingDINO
GroundingDINO is a model for text-prompted object detection and segmentation, offering:
- High accuracy in complex environments
- Precise edge detection and preservation
- Excellent handling of fine details
- Support for multiple objects in a single image
- Output Comparison
- Output with background
- Batch output for video

## BiRefNet Models
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

### SDMatte Models (Manual Download)

*   Auto-download on first run to `models/RMBG/SDMatte/`.
*   If network restricted, manually place the weights:
    *   `models/RMBG/SDMatte/SDMatte.safetensors` (standard) or `SDMatte_plus.safetensors` (plus)
    *   Components (config files) are auto-downloaded; mirror the structure from the Hugging Face repo to `models/RMBG/SDMatte/` (`scheduler/`, `text_encoder/`, `tokenizer/`, `unet/`, `vae/`)

## Troubleshooting

*   **401 error when initializing GroundingDINO / missing models/sam2:**
    *   Delete `%USERPROFILE%\.cache\huggingface\token` and `%USERPROFILE%\.huggingface\token` (if present).
    *   Ensure no `HF_TOKEN`/`HUGGINGFACE_TOKEN` environment variables are set.
    *   Re-run; public repos download anonymously (no login required).

*   **Preview shows "Required input is missing: images":**
    *   Ensure image outputs are connected, and upstream nodes ran successfully.

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
*   Created by: [AILab](https://github.com/1038lab)

## Star History

<!-- STAR HISTORY -->
<a href="https://www.star-history.com/#1038lab/comfyui-rmbg&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=1038lab/comfyui-rmbg&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=1038lab/comfyui-rmbg&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=1038lab/comfyui-rmbg&type=Date" />
 </picture>
</a>
<!-- END STAR HISTORY -->

If you find this custom node useful, please give it a ⭐ on this repo!

## License

GPL-3.0 License