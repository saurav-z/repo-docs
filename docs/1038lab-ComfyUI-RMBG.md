# ComfyUI-RMBG: Effortlessly Remove Backgrounds and Segment Images in ComfyUI

**Enhance your ComfyUI workflows with ComfyUI-RMBG, a powerful custom node enabling advanced image background removal, object segmentation, and precise feature extraction, all in one convenient package.**  [Visit the original repository for more details](https://github.com/1038lab/ComfyUI-RMBG).

## Key Features

*   **Advanced Background Removal:**
    *   Utilizes multiple models: RMBG-2.0, INSPYRENET, BEN, BEN2, BiRefNet and SDMatte models.
    *   Provides various background options (transparent, black, white, green, blue, red).
    *   Supports batch processing for efficient workflows.
    *   Refine Foreground using Fast Foreground Color Estimation.
*   **Precise Object Segmentation:**
    *   Text-prompted object detection using tag-style and natural language inputs.
    *   Integration with SAM and GroundingDINO models for high-precision segmentation.
    *   Fine-tune segmentation results with flexible parameter controls (Threshold, Mask blur, Offset).
*   **Cutting-Edge SAM2 Segmentation:**
    *   Leverages the latest SAM2 models (Tiny/Small/Base+/Large) for enhanced segmentation.
    *   Automatic model downloads for ease of use; manual download options are available.
*   **Versatile Feature Extraction:**
    *   Face segmentation node with support for 19 facial feature categories (Skin, Nose, Eyes, Eyebrows, etc.)
    *   Fashion segmentation node.
    *   Clothes segmentation node with 18 different categories
*   **Image Enhancement Tools:**
    *   Image and Mask Tools improved functionality.
    *   Added Image Combiner, Image Stitch, Image/Mask Converter, Mask Enhancer, Mask Combiner, and Mask Extractor

## Key Updates (Recent)

*   **v2.9.0 (2025/08/18):** Added `SDMatte Matting` node.
*   **v2.8.0 (2025/08/11):** Added `SAM2Segment` node for text-prompted segmentation using the latest Facebook Research SAM2 technology and enhanced color widget support.
*   **v2.7.1 (2025/08/06):** Enhanced LoadImage node and redesigned ImageStitch node
*   **v2.6.0 (2025/07/15):** Added `Kontext Refence latent Mask` node.
*   **v2.5.0 (2025/07/01):** Added `MaskOverlay`, `ObjectRemover`, `ImageMaskResize` new nodes, and new BiRefNet models, and batch image support for `Segment_v1` and `Segment_V2` nodes
*   **v2.4.0 (2025/06/01):** Added `CropObject`, `ImageCompare`, `ColorInput` nodes and new Segment V2
*   **v2.3.0 (2025/05/01):** Added IC-LoRA Concat, Image Crop nodes, and resizing options for Load Image.
*   **v2.2.0 (2025/04/05):** Added new nodes: Image Combiner, Image Stitch, Image/Mask Converter, Mask Enhancer, Mask Combiner, and Mask Extractor
*   **v2.1.0 (2025/03/19):** Integrated internationalization (i18n) support for multiple languages.
*   **v2.0.0 (2025/03/13):** Added Image and Mask Tools improved functionality.
*   **v1.9.2 (2025/02/21):** Added Fast Foreground Color Estimation.
*   **v1.9.0 (2025/02/19):** BiRefNet model improvements.
*   **v1.8.0 (2025/02/07):** Added BiRefNet-HR model.
*   **v1.7.0 (2025/02/04):** Added BEN2 model.
*   **v1.6.0 (2025/01/22):** Added Face Segment custom node.
*   **v1.5.0 (2025/01/05):** Added Fashion and accessories Segment custom node.
*   **v1.4.0 (2025/01/02):** Added Clothes Segment node.

## Installation

*   **Method 1. Install on ComfyUI-Manager:** Search `Comfyui-RMBG` and install.
*   **Method 2. Clone to custom\_nodes:** `cd ComfyUI/custom_nodes` and `git clone https://github.com/1038lab/ComfyUI-RMBG`.  Then, install the requirements with `./ComfyUI/python_embeded/python -m pip install -r requirements.txt`.
*   **Method 3: Install via Comfy CLI:** `comfy node install ComfyUI-RMBG` and then install the requirements with `./ComfyUI/python_embeded/python -m pip install -r requirements.txt`.

## Model Download (Manual - Optional, if auto-download fails)

*   The models will auto-download on first use.
*   Download models from the provided Hugging Face links (see "About Models" section).
*   Place the downloaded files in the corresponding folders within `ComfyUI/models/RMBG/` or `ComfyUI/models/SAM` or `ComfyUI/models/sam2` or `ComfyUI/models/grounding-dino`.

## Usage Guide

### RMBG Node

1.  Load `RMBG (Remove Background)` node from the `🧪AILab/🧽RMBG` category.
2.  Connect an image to the input.
3.  Select a model.
4.  Adjust parameters as needed (Optional).
5.  Get two outputs: Processed image and a mask.

### Segment Node

1.  Load `Segment (RMBG)` node from the `🧪AILab/🧽RMBG` category.
2.  Connect an image to the input.
3.  Enter a text prompt (tag-style or natural language).
4.  Select SAM and GroundingDINO models.
5.  Adjust parameters as needed.

## Parameter Tips

| Parameter            | Description                                                                                                                                       | Tip                                                                                                             |
| :------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------ | :-------------------------------------------------------------------------------------------------------------- |
| Sensitivity          | Controls mask detection strength.  Higher values = stricter detection.                                                                           | Default: 0.5. Adjust for image complexity.                                                                    |
| Processing Resolution | Controls image processing resolution. Affects detail and memory.                                                                               | 256-2048. Higher resolutions improve detail, increase memory use.                                            |
| Mask Blur            | Blurs mask edges.                                                                                                                                | Default: 0. Try 1-5 for smoother edges.                                                                        |
| Mask Offset          | Expands or shrinks mask boundaries.                                                                                                               | Default: 0.  Adjust between -10 and 10.                                                                      |
| Background           | Output background color | Alpha (transparent background) Black, White, Green, Blue, Red |
| Invert Output        | Flip mask and image output | Invert both image and mask output |
| Refine Foreground | Use Fast Foreground Color Estimation to optimize transparent background | Enable for better edge quality and transparency handling |

## About Models
See details in the original README, [linked above](https://github.com/1038lab/ComfyUI-RMBG)

## Requirements

*   ComfyUI
*   Python 3.10+
*   Required packages (automatically installed):  See original README for full list

## Troubleshooting

*   **401 error:**  Delete Hugging Face token (`%USERPROFILE%\.cache\huggingface\token`) and remove any environment variables.
*   **Missing images:** Ensure image outputs are connected and upstream nodes ran.

## Credits
See original README for full list.

## Star History

[![Star History](https://api.star-history.com/svg?repos=1038lab/comfyui-rmbg&type=Date)](https://star-history.com/#1038lab/comfyui-rmbg&Date)

## License
GPL-3.0 License