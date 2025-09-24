# ComfyUI-RMBG: Effortlessly Remove Backgrounds and Segment Images in ComfyUI

**Remove backgrounds, segment objects, and unlock advanced image editing capabilities with ComfyUI-RMBG, a powerful custom node for ComfyUI.** ([View on GitHub](https://github.com/1038lab/ComfyUI-RMBG))

## Key Features

*   **Advanced Background Removal:** Utilize various models like RMBG-2.0, INSPYRENET, BEN, and BEN2 for precise background removal.
*   **Object Segmentation:** Segment images using text prompts with SAM and GroundingDINO for versatile editing.
*   **SAM2 Segmentation:** Leverage the latest SAM2 models for text-prompted segmentation with various sizes.
*   **Real-Time Background Replacement:** Quickly swap out backgrounds for creative effects.
*   **Enhanced Edge Detection:** Improve accuracy with advanced edge detection features.
*   **Facial & Fashion Segmentation:**  Specialized nodes for precise face and fashion element segmentation.
*   **Flexible Output:**  Choose transparent, black, white, green, blue, or red backgrounds.
*   **Batch Processing:**  Process multiple images efficiently.
*   **Regular Updates:**  Benefit from ongoing improvements and new features.

## Latest Updates

*   **v2.9.1** (2025/09/12):  Update ComfyUI-RMBG to **v2.9.1** ([update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v291-20250912)).

    *   ... *(Follow links to see earlier updates)*
    
    ![v2.9.1](https://github.com/user-attachments/assets/9b6c3e6c-5866-4807-91ba-669eb7efc52b)

## Installation

Choose from the following methods to install ComfyUI-RMBG:

### Method 1: ComfyUI Manager

*   Install via the ComfyUI Manager. Search for `Comfyui-RMBG` and install.
*   Install the required packages:
    ```bash
    ./ComfyUI/python_embeded/python -m pip install -r requirements.txt
    ```

### Method 2: Manual Clone

*   Navigate to your ComfyUI custom nodes directory: `cd ComfyUI/custom_nodes`
*   Clone the repository: `git clone https://github.com/1038lab/ComfyUI-RMBG`
*   Install the required packages:
    ```bash
    ./ComfyUI/python_embeded/python -m pip install -r requirements.txt
    ```

### Method 3: Comfy CLI

*   Ensure `pip install comfy-cli` is installed.
*   If needed, install ComfyUI with `comfy install`.
*   Install ComfyUI-RMBG: `comfy node install ComfyUI-RMBG`
*   Install the required packages:
    ```bash
    ./ComfyUI/python_embeded/python -m pip install -r requirements.txt
    ```

### Model Downloads

*   Models are automatically downloaded to `ComfyUI/models/RMBG/` or `ComfyUI/models/SAM` the first time the nodes are used.
*   Alternatively, manually download models from the links provided in the original README, or in the About Models section below, and place them in the specified folders.

## Usage

### RMBG Node

1.  Load the `RMBG (Remove Background)` node from the `🧪AILab/🧽RMBG` category.
2.  Connect an image to the input.
3.  Select a model from the dropdown menu.
4.  Adjust parameters (optional) for sensitivity, processing resolution, mask blur, mask offset, background, and invert output.
5.  Outputs: Processed image and mask.

### Segment Node

1.  Load the `Segment (RMBG)` node from the `🧪AILab/🧽RMBG` category.
2.  Connect an image to the input.
3.  Enter a text prompt (tag-style or natural language).
4.  Select SAM or GroundingDINO models.
5.  Adjust parameters (threshold, mask blur, offset, background).

## Detailed Guide: RMBG Node & Segment Node (Detailed in the original README)

### :bulb: Tips for RMBG Node

| Optional Settings           | :memo: Description                                                               | :bulb: Tips                                                                                   |
| --------------------------- | --------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| **Sensitivity**             | Adjusts mask detection.  Higher values result in stricter detection.             | Default: 0.5.  Increase for complex images.                                                   |
| **Processing Resolution** | Controls processing resolution, affecting detail and memory usage.                | 256-2048 (default: 1024). Higher resolutions provide better detail but increase memory usage. |
| **Mask Blur**               | Controls mask edge blur, reducing jaggedness.                                   | Default: 0.  Try 1-5 for smoother edges.                                                        |
| **Mask Offset**             | Expand/shrink mask boundary.  (+) expands, (-) shrinks.                          | Default: 0. Fine-tune between -10 and 10.                                                       |
| **Background**              | Choose output background color.                                                   | Alpha (transparent), Black, White, Green, Blue, Red.                                           |
| **Invert Output**           | Flip mask and image output.                                                      | Invert both image and mask output.                                                             |
| **Refine Foreground**       | Use Fast Foreground Color Estimation to optimize transparent background.         | Enable for better edge quality and transparency handling.                                      |
| **Performance Optimization**| Enhance performance when processing multiple images.                              | Maximize `process_res` and `mask_blur` considering memory usage.                             |

### Parameters (RMBG Node)

*   `sensitivity`: Controls background removal sensitivity (0.0-1.0)
*   `process_res`: Processing resolution (512-2048, step 128)
*   `mask_blur`: Blur amount for the mask (0-64)
*   `mask_offset`: Adjust mask edges (-20 to 20)
*   `background`: Choose output background color
*   `invert_output`: Flip mask and image output
*   `optimize`: Toggle model optimization

### Parameters (Segment Node)

1.  Enter text prompt (tag-style or natural language)
2.  Select SAM and GroundingDINO models
3.  Adjust parameters as needed:
    -   Threshold: 0.25-0.35 for broad detection, 0.45-0.55 for precision
    -   Mask blur and offset for edge refinement
    -   Background color options

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
*   If network restricted, place weights manually:
    *   `models/RMBG/SDMatte/SDMatte.safetensors` (standard) or `SDMatte_plus.safetensors` (plus)
    *   Components (config files) are auto-downloaded; if needed, mirror the structure from the Hugging Face repo to `models/RMBG/SDMatte/` (`scheduler/`, `text_encoder/`, `tokenizer/`, `unet/`, `vae/`)

## Troubleshooting

*   **401 error with GroundingDINO / missing `models/sam2`:**
    *   Delete `%USERPROFILE%\.cache\huggingface\token` and `%USERPROFILE%\.huggingface\token`.
    *   Ensure no `HF_TOKEN`/`HUGGINGFACE_TOKEN` environment variables are set.
    *   Re-run (public repos download anonymously).
*   **Preview shows "Required input is missing: images":**
    *   Ensure image outputs are connected and upstream nodes ran successfully.

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

<a href="https://www.star-history.com/#1038lab/comfyui-rmbg&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=1038lab/comfyui-rmbg&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=1038lab/comfyui-rmbg&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=1038lab/comfyui-rmbg&type=Date" />
 </picture>
</a>

## License

GPL-3.0 License