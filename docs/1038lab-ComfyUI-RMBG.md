# Enhance Your Images with ComfyUI-RMBG: Advanced Background Removal and Segmentation

Effortlessly remove backgrounds, segment objects, and refine images within ComfyUI using the powerful ComfyUI-RMBG custom node.  [View the original repository](https://github.com/1038lab/ComfyUI-RMBG).

## Key Features

*   **Advanced Background Removal:** Leverage a variety of models (RMBG-2.0, INSPYRENET, BEN, BEN2, BiRefNet, SDMatte) for precise background removal.
*   **Object & Subject Segmentation:** Segment objects, faces, clothing, and fashion elements using text prompts with SAM and GroundingDINO models.
*   **SAM2 Segmentation:** Utilize the latest SAM2 models (Tiny/Small/Base+/Large) for text-prompted segmentation.
*   **Real-time Background Replacement:** Seamlessly replace backgrounds with various color options.
*   **Enhanced Edge Detection:** Achieve superior accuracy with improved edge refinement and detail preservation.
*   **Model Flexibility:** Choose from a wide array of segmentation models, with automatic or manual model downloads.
*   **User-Friendly Interface:** Intuitive nodes and parameter controls for easy image manipulation.
*   **Regular Updates:** Stay up-to-date with the latest features, including new models and enhancements.

## What's New

*   **v2.9.0 (2025/08/18):** Added `SDMatte Matting` node.
*   **v2.8.0 (2025/08/11):** Added `SAM2Segment` node and enhanced color widget support.
*   **v2.7.1 (2025/08/06):** Enhanced LoadImage and ImageStitch nodes, and fixed background color handling issues.
*   **v2.6.0 (2025/07/15):** Added `Kontext Refence latent Mask` node.
*   **v2.5.2 (2025/07/11):** Bug fixes and improvements.
*   **v2.5.1 (2025/07/07):** Bug fixes.
*   **v2.5.0 (2025/07/01):** Added `MaskOverlay`, `ObjectRemover`, `ImageMaskResize` nodes and new BiRefNet models.
*   **v2.4.0 (2025/06/01):** Added `CropObject`, `ImageCompare`, `ColorInput` nodes and new Segment V2.
*   **v2.3.2 (2025/05/15):** Bug fixes.
*   **v2.3.1 (2025/05/02):** Bug fixes.
*   **v2.3.0 (2025/05/01):** Added IC-LoRA Concat, Image Crop nodes and resizing options.
*   **v2.2.1 (2025/04/05):** Bug fixes.
*   **v2.2.0 (2025/04/05):** Added Image Combiner, Image Stitch, Image/Mask Converter, Mask Enhancer, Mask Combiner, and Mask Extractor and mask image output to segment nodes.
*   **v2.1.1 (2025/03/21):** Enhanced compatibility with Transformers.
*   **v2.1.0 (2025/03/19):** Integrated internationalization (i18n) support for multiple languages.
*   **v2.0.0 (2025/03/13):** Added Image and Mask Tools, improved code structure and documentation.
*   **v1.9.3 (2025/02/24):** Clean up the code and fix the issue.
*   **v1.9.2 (2025/02/21):** Added Fast Foreground Color Estimation.
*   **v1.9.1 (2025/02/20):** Changed repository for model management.
*   **v1.9.0 (2025/02/19):** BiRefNet model improvements.
*   **v1.8.0 (2025/02/07):** Added new BiRefNet-HR model.
*   **v1.7.0 (2025/02/04):** Added new BEN2 model.
*   **v1.6.0 (2025/01/22):** Added new Face Segment custom node.
*   **v1.5.0 (2025/01/05):** Added new Fashion and accessories Segment custom node.
*   **v1.4.0 (2025/01/02):** Added new Clothes Segment node.
*   **v1.3.2 (2024/12/29):** Enhanced background handling.
*   **v1.3.1 (2024/12/25):** Bug fixes.
*   **v1.3.0 (2024/12/23):** Added new Segment node.
*   **v1.2.2 (2024/12/12):** Bug fixes.
*   **v1.2.1 (2024/12/02):** Bug fixes.
*   **v1.2.0 (2024/11/29):** General improvements and bug fixes.
*   **v1.1.0 (2024/11/21):** General improvements and bug fixes.

## Installation

Choose from the following methods:

*   **Method 1: Install via ComfyUI Manager** - Search and install "Comfyui-RMBG". Then install requirment.txt in the ComfyUI-RMBG folder.
*   **Method 2: Clone to custom_nodes Folder:**
    ```bash
    cd ComfyUI/custom_nodes
    git clone https://github.com/1038lab/ComfyUI-RMBG
    ```
    Then install requirment.txt in the ComfyUI-RMBG folder.
*   **Method 3: Install via Comfy CLI:**
    ```bash
    comfy node install ComfyUI-RMBG
    ```
    Then install requirment.txt in the ComfyUI-RMBG folder.

*   **Install Requirements:**
    ```bash
    ./ComfyUI/python_embeded/python -m pip install -r requirements.txt
    ```

## Model Download

*   Models will be downloaded automatically on first use to `ComfyUI/models/RMBG/` and  `ComfyUI/models/SAM/`, `ComfyUI/models/sam2`, `ComfyUI/models/grounding-dino`,
    `/ComfyUI/models/RMBG/segformer_clothes` and `/ComfyUI/models/RMBG/segformer_fashion`.
*   Manual Download: If needed, download models from the provided Hugging Face links and place them in the corresponding folders (e.g., `ComfyUI/models/RMBG/RMBG-2.0`, `ComfyUI/models/SAM/`).

## Usage Guide

### RMBG Node

1.  Load the `RMBG (Remove Background)` node from `🧪AILab/🧽RMBG`.
2.  Connect your image input.
3.  Select a background removal model.
4.  Adjust optional parameters (see below).
5.  Get the processed image and a foreground mask as outputs.

### Optional Settings Tips

| Optional Settings          | Description                                                                                                                                  | Tips                                                                                                                  |
| :------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------- |
| **Sensitivity**            | Adjusts mask detection strength.  Higher values mean stricter detection.                                                                      | Default: 0.5.  Increase for complex images.                                                                      |
| **Processing Resolution** | Sets the image processing resolution. Higher res = more detail & memory use.                                                                | Values from 256 to 2048 (step 128).  Default: 1024.                                                                |
| **Mask Blur**              | Blurs mask edges.                                                                                                                           | Default: 0.  Experiment with values from 1 to 5 for smoother edges.                                                 |
| **Mask Offset**            | Expands or shrinks the mask boundary.                                                                                                          | Default: 0.  Fine-tune between -10 and 10.                                                                            |
| **Background**             | Select output background color                                                                                                              | Alpha (transparent background) Black, White, Green, Blue, Red                                                      |
| **Invert Output**      | Flip mask and image output | Invert both image and mask output |
| **Refine Foreground**      |  Use Fast Foreground Color Estimation to optimize transparent background                                                                                                              | Enable for better edge quality and transparency handling |
| **Performance Optimization**      | Properly setting options can enhance performance when processing multiple images.                                                                                                              | If memory allows, consider increasing `process_res` and `mask_blur` values for better results, but be mindful of memory usage. |

### Basic Usage

1.  Load `RMBG (Remove Background)` node from the `🧪AILab/🧽RMBG` category
2.  Connect an image to the input
3.  Select a model from the dropdown menu
4.  select the parameters as needed (optional)
3.  Get two outputs:
   - IMAGE: Processed image with transparent, black, white, green, blue, or red background
   - MASK: Binary mask of the foreground

### Segment Node

1.  Load the `Segment (RMBG)` node from `🧪AILab/🧽RMBG`.
2.  Connect your image.
3.  Enter a text prompt (tag-style or natural language).
4.  Select SAM or GroundingDINO model.
5.  Adjust parameters:
    *   Threshold: 0.25-0.35 (broad), 0.45-0.55 (precise).
    *   Mask blur & offset for edge refinement.
    *   Background color options.

## Parameters

*   `sensitivity`: Sensitivity for background removal (0.0-1.0).
*   `process_res`: Processing resolution (512-2048, step 128).
*   `mask_blur`: Mask blur amount (0-64).
*   `mask_offset`: Adjust mask edges (-20 to 20).
*   `background`: Choose output background color
*   `invert_output`: Flip mask and image output
*   `optimize`: Toggle model optimization

## Models

This node supports a comprehensive selection of models (see the "About Models" section in the original README for in-depth information, which can be found at the bottom).

## Requirements

*   ComfyUI
*   Python 3.10+
*   Automatic package installation (see original README for specifics).

## Troubleshooting

*   Common issues and solutions are listed in the original README.

## Credits

*   This project utilizes models and code from various sources.  See the original README for a complete list of credits.

## Star History

[Include Star History Chart here]

Give this repo a ⭐ if you like the work!

## License

GPL-3.0 License