# ComfyUI-RMBG: Advanced Background Removal and Image Segmentation for ComfyUI

**Effortlessly remove backgrounds and segment objects with precision using the ComfyUI-RMBG custom node, powered by cutting-edge AI models.**  Explore the original repo [here](https://github.com/1038lab/ComfyUI-RMBG).

## Key Features:

*   **Background Removal:**
    *   Utilizes multiple models: RMBG-2.0, INSPYRENET, BEN, BEN2, BiRefNet-HR, etc.
    *   Offers versatile background options: Transparent, Black, White, Green, Blue, Red.
    *   Supports batch processing for efficiency.
*   **Object Segmentation:**
    *   Text-prompted object detection: Tag-style or natural language prompts.
    *   Employs SAM and GroundingDINO models for precise segmentation.
    *   Includes flexible parameter controls for customization.
*   **Specialized Segmentation:**
    *   Face segmentation with 19 facial feature categories
    *   Fashion and Accessories segmentation
    *   Clothes segmentation with 18 different categories.

## Updates: Stay up-to-date with the latest features!

**(Note: Refer to [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md) for detailed update information and version-specific details.)**

*   **v2.7.1 (2025/08/06):** ( [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v271-20250806) )
*   **v2.7.0 (2025/07/27):** ( [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v270-20250727) )
    *   Enhanced LoadImage nodes.
    *   Redesigned ImageStitch node.
    *   Fixed background color handling issues.
*   **v2.6.0 (2025/07/15):** ( [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v260-20250715) )
    *   Added `Kontext Refence latent Mask` node.
*   **v2.5.2 (2025/07/11):** ( [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v252-20250711) )
*   **v2.5.1 (2025/07/07):** ( [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v251-20250707) )
*   **v2.5.0 (2025/07/01):** ( [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v250-20250701) )
    *   Added `MaskOverlay`, `ObjectRemover`, `ImageMaskResize` new nodes.
    *   Added 2 BiRefNet models.
    *   Added batch image support for `Segment_v1` and `Segment_V2` nodes
*   **v2.4.0 (2025/06/01):** ( [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v240-20250601) )
    *   Added `CropObject`, `ImageCompare`, `ColorInput` nodes and new Segment V2
*   **v2.3.2 (2025/05/15):** ( [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v232-20250515) )
*   **v2.3.1 (2025/05/02):** ( [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v231-20250502) )
*   **v2.3.0 (2025/05/01):** ( [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v230-20250501) )
    *   Added new nodes: IC-LoRA Concat, Image Crop
    *   Added resizing options for Load Image: Longest Side, Shortest Side, Width, and Height, enhancing flexibility.
*   **v2.2.1 (2025/04/05):** ( [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v221-20250405) )
*   **v2.2.0 (2025/04/05):** ( [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v220-20250405) )
    *   Added new nodes: Image Combiner, Image Stitch, Image/Mask Converter, Mask Enhancer, Mask Combiner, and Mask Extractor
    *   Fixed compatibility issues with transformers v4.49+
    *   Fixed i18n translation errors
    *   Added mask image output to segment nodes
*   **v2.1.1 (2025/03/21):** ( [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v211-20250321) )
    *   Enhanced compatibility with Transformers
*   **v2.1.0 (2025/03/19):** ( [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v210-20250319) )
    *   Integrated internationalization (i18n) support for multiple languages.
    *   Improved user interface for dynamic language switching.
    *   Enhanced accessibility for non-English speaking users with fully translatable features.
*   **v2.0.0 (2025/03/13):** ( [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v200-20250313) )
    *   Added Image and Mask Tools improved functionality.
    *   Enhanced code structure and documentation for better usability.
    *   Introduced a new category path: `🧪AILab/🛠️UTIL/🖼️IMAGE`.
*   **v1.9.3 (2025/02/24):** ( [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v193-20250224) )
    *   Clean up the code and fix the issue.
*   **v1.9.2 (2025/02/21):** ( [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v192-20250221) )
    *   Added new foreground refinement feature for better transparency handling
    *   Improved edge quality and detail preservation
    *   Enhanced memory optimization
*   **v1.9.1 (2025/02/20):** ( [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v191-20250220) )
    *   Changed repository for model management to the new repository and Reorganized models files structure for better maintainability.
*   **v1.9.0 (2025/02/19):** ( [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v190-20250219) )
    *   Enhanced BiRefNet model performance and stability
    *   Improved memory management for large images
*   **v1.8.0 (2025/02/07):** ( [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v180-20250207) )
    *   Added a new custom node for BiRefNet-HR model.
    *   Support high resolution image processing (up to 2048x2048)
*   **v1.7.0 (2025/02/04):** ( [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v170-20250204) )
    *   Added a new custom node for BEN2 model.
*   **v1.6.0 (2025/01/22):** ( [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v160-20250122) )
    *   Added a new custom node for face parsing and segmentation
    *   Support for 19 facial feature categories (Skin, Nose, Eyes, Eyebrows, etc.)
    *   Precise facial feature extraction and segmentation
    *   Multiple feature selection for combined segmentation
    *   Same parameter controls as other RMBG nodes
*   **v1.5.0 (2025/01/05):** ( [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v150-20250105) )
    *   Added a new custom node for fashion segmentation.
*   **v1.4.0 (2025/01/02):** ( [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v140-20250102) )
    *   Added intelligent clothes segmentation with 18 different categories
    *   Support multiple item selection and combined segmentation
    *   Same parameter controls as other RMBG nodes
*   **v1.3.2 (2024/12/29):** ( [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v132-20241229) )
    *   Enhanced background handling to support RGBA output when "Alpha" is selected.
    *   Ensured RGB output for all other background color selections.
*   **v1.3.1 (2024/12/25):** ( [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v131-20241225) )
    *   Fixed an issue with mask processing when the model returns a list of masks.
    *   Improved handling of image formats to prevent processing errors.
*   **v1.3.0 (2024/12/23):** ( [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v140-20241222) )
    *   Added text-prompted object segmentation
    *   Support both tag-style ("cat, dog") and natural language ("a person wearing red jacket") prompts
    *   Multiple models: SAM (vit_h/l/b) and GroundingDINO (SwinT/B) (as always model file will be downloaded automatically when first time using the specific model)
    *   This update requires install requirements.txt
*   **v1.2.2 (2024/12/12):** ( [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v122-20241212) )
*   **v1.2.1 (2024/12/02):** ( [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.mdv121-20241202) )
*   **v1.2.0 (2024/11/29):** ( [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v120-20241129) )
*   **v1.1.0 (2024/11/21):** ( [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v110-20241121) )

## Installation:

### Method 1: Install via ComfyUI Manager
Search for `Comfyui-RMBG` and install directly within ComfyUI.

### Method 2: Clone the Repository
1.  Navigate to your ComfyUI's `custom_nodes` directory:
    ```bash
    cd ComfyUI/custom_nodes
    ```
2.  Clone the repository:
    ```bash
    git clone https://github.com/1038lab/ComfyUI-RMBG
    ```
3.  Install the required packages:
    ```bash
    ./ComfyUI/python_embeded/python -m pip install -r requirements.txt
    ```

### Method 3: Install via Comfy CLI
1.  Ensure `comfy-cli` is installed: `pip install comfy-cli`
2.  Install ComfyUI if you don't have it: `comfy install`
3.  Install ComfyUI-RMBG:
    ```bash
    comfy node install ComfyUI-RMBG
    ```
4.  Install requirements:
    ```bash
    ./ComfyUI/python_embeded/python -m pip install -r requirements.txt
    ```

### Model Downloads:

*   Models are automatically downloaded to the `ComfyUI/models/RMBG/` directory upon first use.
*   Alternatively, you can manually download models from the provided links below and place them in the appropriate folders:
    *   RMBG-2.0: [https://huggingface.co/1038lab/RMBG-2.0](https://huggingface.co/1038lab/RMBG-2.0) (in `/ComfyUI/models/RMBG/RMBG-2.0`)
    *   INSPYRENET: [https://huggingface.co/1038lab/inspyrenet](https://huggingface.co/1038lab/inspyrenet) (in `/ComfyUI/models/RMBG/INSPYRENET`)
    *   BEN: [https://huggingface.co/PramaLLC/BEN](https://huggingface.co/PramaLLC/BEN) (in `/ComfyUI/models/RMBG/BEN`)
    *   BEN2: [https://huggingface.co/PramaLLC/BEN2](https://huggingface.co/PramaLLC/BEN2) (in `/ComfyUI/models/RMBG/BEN2`)
    *   BiRefNet-HR: [https://huggingface.co/1038lab/BiRefNet_HR](https://huggingface.co/1038lab/BiRefNet_HR) (in `/ComfyUI/models/RMBG/BiRefNet-HR`)
    *   SAM: [https://huggingface.co/facebook/sam-vit-base](https://huggingface.co/facebook/sam-vit-base) (in `/ComfyUI/models/SAM`)
    *   GroundingDINO: [https://github.com/IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) (in `/ComfyUI/models/grounding-dino`)
    *   Clothes Segment: [https://huggingface.co/mattmdjaga/segformer_b2_clothes](https://huggingface.co/mattmdjaga/segformer_b2_clothes) (in `/ComfyUI/models/RMBG/segformer_clothes`)
    *   Fashion Segment: [https://huggingface.co/1038lab/segformer_fashion](https://huggingface.co/1038lab/segformer_fashion) (in `/ComfyUI/models/RMBG/segformer_fashion`)
    *   BiRefNet Models: [https://huggingface.co/1038lab/BiRefNet](https://huggingface.co/1038lab/BiRefNet) (in `/ComfyUI/models/RMBG/BiRefNet`)

## Usage:

### RMBG Node:

1.  Load the `RMBG (Remove Background)` node from the `🧪AILab/🧽RMBG` category.
2.  Connect an image to the input.
3.  Select a model from the dropdown menu.
4.  Adjust optional parameters as needed.
5.  Outputs:
    *   `IMAGE`: Processed image with the selected background (transparent, black, white, green, blue, or red).
    *   `MASK`: Binary mask of the foreground.

### Optional Settings: Tips for Optimal Results

| Optional Settings         | Description                                                                 | Tips                                                                                                                               |
| ------------------------- | --------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| **Sensitivity**           | Adjusts the mask detection strength. Higher values = stricter detection. | Default: 0.5.  Adjust based on image complexity; higher for complex images.                                                       |
| **Processing Resolution** | Controls image processing resolution, affecting detail and memory usage.    | Choose between 256 and 2048 (step 128), Default: 1024. Higher for better detail, but more memory usage.                              |
| **Mask Blur**             | Blurs mask edges, reducing jaggedness.                                     | Default: 0.  Try 1-5 for smoother edges.                                                                                              |
| **Mask Offset**           | Expands or shrinks the mask boundary.                                      | Default: 0.  Adjust between -10 and 10 for fine-tuning.                                                                          |
| **Background**            | Choose the output background color.                                         | Options: Alpha (transparent), Black, White, Green, Blue, Red.                                                                      |
| **Invert Output**         | Flips mask and image output.                                                | Inverts both image and mask output.                                                                                                |
| **Refine Foreground**     | Uses Fast Foreground Color Estimation for optimized transparency handling.  | Enable for better edge quality and transparency.                                                                                   |
| **Performance Optimization**| Setting options correctly can enhance processing multiple images. | Increase `process_res` and `mask_blur` values for better results if you have enough memory, but be mindful of memory usage.             |

### Segment Node:

1.  Load the `Segment (RMBG)` node from the `🧪AILab/🧽RMBG` category.
2.  Connect an image to the input.
3.  Enter a text prompt (tag-style like "cat, dog" or natural language like "a person wearing red jacket").
4.  Select SAM and GroundingDINO models.
5.  Adjust parameters:
    *   Threshold: 0.25-0.35 for broad detection, 0.45-0.55 for precision.
    *   Mask blur and offset for edge refinement.
    *   Background color options.

<details>
<summary><h2>About Models</h2></summary>

## RMBG-2.0
RMBG-2.0 is developed by BRIA AI and uses the BiRefNet architecture which includes:
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
*   Automatically installed packages:
    *   torch>=2.0.0
    *   torchvision>=0.15.0
    *   Pillow>=9.0.0
    *   numpy>=1.22.0
    *   huggingface-hub>=0.19.0
    *   tqdm>=4.65.0
    *   transformers>=4.35.0
    *   transparent-background>=1.2.4
    *   opencv-python>=4.7.0

## Credits

*   RMBG-2.0: [https://huggingface.co/briaai/RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0)
*   INSPYRENET: [https://github.com/plemeri/InSPyReNet](https://github.com/plemeri/InSPyReNet)
*   BEN: [https://huggingface.co/PramaLLC/BEN](https://huggingface.co/PramaLLC/BEN)
*   BEN2: [https://huggingface.co/PramaLLC/BEN2](https://huggingface.co/PramaLLC/BEN2)
*   BiRefNet: [https://huggingface.co/ZhengPeng7](https://huggingface.co/ZhengPeng7)
*   SAM: [https://huggingface.co/facebook/sam-vit-base](https://huggingface.co/facebook/sam-vit-base)
*   GroundingDINO: [https://github.com/IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
*   Clothes Segment: [https://huggingface.co/mattmdjaga/segformer_b2_clothes](https://huggingface.co/mattmdjaga/segformer_b2_clothes)

*   Created by: [AILab](https://github.com/1038lab)

## Star History

[![Star History](https://api.star-history.com/svg?repos=1038lab/comfyui-rmbg&type=Date)](https://star-history.com/#1038lab/comfyui-rmbg&Date)

**If you find this custom node helpful, please give it a ⭐ on this repository!** Your support is greatly appreciated!

## License
GPL-3.0 License