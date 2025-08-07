# ComfyUI-RMBG: Effortlessly Remove Backgrounds and Segment Images with Advanced AI

**Enhance your ComfyUI workflows with ComfyUI-RMBG, a powerful custom node offering advanced image background removal, object segmentation, and more!**  Access the original repository [here](https://github.com/1038lab/ComfyUI-RMBG).

## Key Features:

*   **Advanced Background Removal:**
    *   Utilizes multiple models, including RMBG-2.0, INSPYRENET, BEN, and BEN2, for precise background removal.
    *   Offers flexible background options: transparent, black, white, green, blue, or red.
    *   Supports batch processing for efficient workflow.

*   **Object Segmentation:**
    *   Segment objects using text prompts (tag-style or natural language) with Segment node.
    *   Utilizes SAM (Segment Anything Model) and GroundingDINO for high-precision segmentation.
    *   Provides flexible parameter controls for refining results.

*   **Facial and Fashion Segmentation:**
    *   Dedicated nodes for Face, Fashion, and Clothes segmentation for detailed content editing.
    *   Face Node supports 19 facial feature categories.
    *   Clothes Node supports 18 different clothing categories.

*   **Model Variety & Flexibility:**
    *   Includes BiRefNet models, including HR versions for high-resolution image processing.
    *   Offers various resolution and parameter options to optimize performance.

*   **Easy Installation:**
    *   Install via ComfyUI Manager, cloning the repository, or Comfy CLI.
    *   Includes automatic model downloading.
    *   Dependencies are automatically installed.

*   **Continuous Updates:**
    *   Regular updates with new models, features, and bug fixes.

## Recent Updates:

*   **v2.7.1 (2025/08/06):** ( [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v271-20250806) )
*   **v2.7.0 (2025/07/27):** ( [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v270-20250727) )
    *   Enhanced LoadImage node with three distinct options for versatile image loading, including URL support.
    *   Redesigned ImageStitch node compatible with ComfyUI's native functionalities.
    *   Addressed background color handling issues reported by users.
*   **v2.6.0 (2025/07/15):** ( [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v260-20250715) )
    *   Added `Kontext Refence latent Mask` node, Which uses a reference latent and mask for precise region conditioning.
*   **v2.5.2 (2025/07/11):** ( [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v252-20250711) )
*   **v2.5.1 (2025/07/07):** ( [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v251-20250707) )
*   **v2.5.0 (2025/07/01):** ( [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v250-20250701) )
    *   Added new nodes: `MaskOverlay`, `ObjectRemover`, `ImageMaskResize`.
    *   Added 2 BiRefNet models: `BiRefNet_lite-matting` and `BiRefNet_dynamic`.
    *   Added batch image support for `Segment_v1` and `Segment_V2` nodes.

*(See update.md files for comprehensive update details)*

## Installation:

1.  **ComfyUI Manager:** Search for and install "ComfyUI-RMBG" within the ComfyUI Manager.
2.  **Clone Repository:**
    ```bash
    cd ComfyUI/custom_nodes
    git clone https://github.com/1038lab/ComfyUI-RMBG
    ```
    Then install dependencies:
    ```bash
    ./ComfyUI/python_embeded/python -m pip install -r requirements.txt
    ```
3.  **Comfy CLI:**
    ```bash
    comfy node install ComfyUI-RMBG
    ```
    Then install dependencies:
    ```bash
    ./ComfyUI/python_embeded/python -m pip install -r requirements.txt
    ```

4.  **Model Download (Automatic):** Models are automatically downloaded upon first use.
5.  **Manual Model Download:** (If needed) Place model files in the specified `/ComfyUI/models/RMBG/` and `/ComfyUI/models/SAM/` or `/ComfyUI/models/grounding-dino/` directories.  Links to download are provided in the original README (see below).

## Basic Usage:

### RMBG Node
1.  Add an `RMBG (Remove Background)` node from the `🧪AILab/🧽RMBG` category.
2.  Connect an image to the input.
3.  Select a model.
4.  Adjust optional parameters like sensitivity, resolution, and background color.
5.  View the processed image and mask outputs.

### Segment Node
1.  Add a `Segment (RMBG)` node from the `🧪AILab/🧽RMBG` category.
2.  Connect an image to the input.
3.  Enter a text prompt.
4.  Select SAM and GroundingDINO models.
5.  Adjust the threshold, mask blur, and background color as needed.

## Optional Settings:

*   **Sensitivity:** Controls mask detection strength. Higher values = stricter.
*   **Processing Resolution:** Affects detail and memory usage.
*   **Mask Blur:** Smooths mask edges.
*   **Mask Offset:** Expands or shrinks the mask boundary.
*   **Background:** Choose output background color.
*   **Invert Output:** Flip mask and image output.
*   **Refine Foreground:** Improves transparency with Fast Foreground Color Estimation.

## Models: (More details in the original README, see below)

*   RMBG-2.0
*   INSPYRENET
*   BEN
*   BEN2
*   BiRefNet (Various versions)
*   SAM
*   GroundingDINO
*   Clothes Segment
*   Fashion Segment

## Requirements:

*   ComfyUI
*   Python 3.10+
*   Dependencies (automatically installed, see original README for the list).

## Credits:

*   Created by: [AILab](https://github.com/1038lab)
*   Refer to the original README for model credits and links.

## Star History
**(See original README for Star History chart)**

## License:
GPL-3.0 License

---
**(Original README content starts below - kept for detailed model information and download links)**

```

## About Models

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

## Requirements
- ComfyUI
- Python 3.10+
- Required packages (automatically installed):
  - torch>=2.0.0
  - torchvision>=0.15.0
  - Pillow>=9.0.0
  - numpy>=1.22.0
  - huggingface-hub>=0.19.0
  - tqdm>=4.65.0
  - transformers>=4.35.0
  - transparent-background>=1.2.4
  - opencv-python>=4.7.0

## Credits
- RMBG-2.0: https://huggingface.co/briaai/RMBG-2.0
- INSPYRENET: https://github.com/plemeri/InSPyReNet
- BEN: https://huggingface.co/PramaLLC/BEN
- BEN2: https://huggingface.co/PramaLLC/BEN2
- BiRefNet: https://huggingface.co/ZhengPeng7
- SAM: https://huggingface.co/facebook/sam-vit-base
- GroundingDINO: https://github.com/IDEA-Research/GroundingDINO
- Clothes Segment: https://huggingface.co/mattmdjaga/segformer_b2_clothes

- Created by: [AILab](https://github.com/1038lab)

## Star History

<a href="https://www.star-history.com/#1038lab/comfyui-rmbg&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=1038lab/comfyui-rmbg&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=1038lab/comfyui-rmbg&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=1038lab/comfyui-rmbg&type=Date" />
 </picture>
</a>

If this custom node helps you or you like my work, please give me ⭐ on this repo! It's a great encouragement for my efforts!

## License
GPL-3.0 License