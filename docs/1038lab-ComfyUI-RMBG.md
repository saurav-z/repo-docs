# ComfyUI-RMBG: Effortlessly Remove Backgrounds and Segment Images

Remove backgrounds, segment objects, and perform advanced image manipulation with ComfyUI-RMBG, a powerful custom node for ComfyUI.  [View the original repository](https://github.com/1038lab/ComfyUI-RMBG).

## Key Features

*   **Advanced Background Removal:**
    *   Utilizes a diverse range of models: RMBG-2.0, INSPYRENET, BEN, BEN2, BiRefNet, SDMatte.
    *   Offers various background options (transparent, black, white, custom).
    *   Supports batch processing for efficient workflows.
*   **Precise Object Segmentation:**
    *   Text-prompted object detection using SAM and GroundingDINO.
    *   Supports both tag-style ("cat, dog") and natural language prompts.
    *   Fine-grained control over segmentation parameters.
*   **SAM2 Segmentation:**
    *   Leverages the latest SAM2 models for cutting-edge text-prompted segmentation.
    *   Automatic model download for ease of use.
*   **Enhanced Image Manipulation:**
    *   Real-time background replacement capabilities.
    *   Improved edge detection for superior accuracy.
    *   Image stitching and merging functionalities.
    *   Mask manipulation tools (enhancement, extraction, combination).
*   **Fashion & Facial Feature Segmentation:**
    *   Dedicated nodes for fashion segmentation.
    *   Facial feature parsing and segmentation.
    *   Clothes segmentation with 18 different categories.

## Latest Updates
*   **v2.9.1** (2025/09/12): Update ComfyUI-RMBG
*   **v2.9.0** (2025/08/18): Added `SDMatte Matting` node
*   **v2.8.0** (2025/08/11): Added `SAM2Segment` node and Enhanced color widget support
*   **v2.7.1** (2025/08/06): Enhanced LoadImage nodes, redesigned ImageStitch node and bug fixes
*   **v2.6.0** (2025/07/15): Added `Kontext Refence latent Mask` node
*   **v2.5.2, v2.5.1, v2.5.0** (2025/07/15-2025/07/01): Added new nodes (MaskOverlay, ObjectRemover, ImageMaskResize, BiRefNet models) and more.
*   **v2.4.0** (2025/06/01): Added new nodes: CropObject, ImageCompare, ColorInput, Segment V2 and more.
*   **v2.3.2, v2.3.1, v2.3.0** (2025/05/15-2025/05/01): Added new nodes: IC-LoRA Concat, Image Crop and resizing options
*   **v2.2.1, v2.2.0** (2025/04/05): Added new nodes: Image Combiner, Image Stitch, Image/Mask Converter, Mask Enhancer, Mask Combiner, and Mask Extractor and more
*   **v2.1.1, v2.1.0** (2025/03/21-2025/03/19): Internationalization (i18n) support for multiple languages.
*   **v2.0.0** (2025/03/13): Added Image and Mask Tools, new category path: `🧪AILab/🛠️UTIL/🖼️IMAGE`.
*   **v1.9.3** (2025/02/24): Clean up the code and fix the issue
*   **v1.9.2** (2025/02/21): Added Fast Foreground Color Estimation
*   **v1.9.1** (2025/02/20): Changed repository for model management
*   **v1.9.0** (2025/02/19): BiRefNet model improvements
*   **v1.8.0** (2025/02/07): Added a new custom node for BiRefNet-HR model.
*   **v1.7.0** (2025/02/04): Added a new custom node for BEN2 model.
*   **v1.6.0** (2025/01/22): Added a new custom node for face parsing and segmentation
*   **v1.5.0** (2025/01/05): Added a new custom node for fashion segmentation.
*   **v1.4.0** (2025/01/02): Added intelligent clothes segmentation with 18 different categories
*   **v1.3.2** (2024/12/29): Enhanced background handling
*   **v1.3.1** (2024/12/25): Bug fixes
*   **v1.3.0** (2024/12/23): Added text-prompted object segmentation
*   **v1.2.2, v1.2.1, v1.2.0, v1.1.0** (2024/12/12-2024/11/21)
*   And more...

## Installation

Choose your preferred method:

1.  **ComfyUI Manager:** Search for `Comfyui-RMBG` and install directly through the ComfyUI Manager.  Remember to install the required packages after installing the node itself:
    ```bash
    ./ComfyUI/python_embeded/python -m pip install -r requirements.txt
    ```
2.  **Clone to Custom Nodes:**
    ```bash
    cd ComfyUI/custom_nodes
    git clone https://github.com/1038lab/ComfyUI-RMBG
    ```
    Install dependencies:
    ```bash
    ./ComfyUI/python_embeded/python -m pip install -r requirements.txt
    ```

3.  **Comfy CLI:** Ensure `pip install comfy-cli` is installed, then use:
    ```bash
    comfy node install ComfyUI-RMBG
    ```
    Install dependencies:
    ```bash
    ./ComfyUI/python_embeded/python -m pip install -r requirements.txt
    ```

4.  **Manual Model Download:** Models are auto-downloaded. If manual download is required, place model files in the respective directories:

    *   RMBG-2.0: `/ComfyUI/models/RMBG/RMBG-2.0`
    *   INSPYRENET: `/ComfyUI/models/RMBG/INSPYRENET`
    *   BEN: `/ComfyUI/models/RMBG/BEN`
    *   BEN2: `/ComfyUI/models/RMBG/BEN2`
    *   BiRefNet-HR: `/ComfyUI/models/RMBG/BiRefNet-HR`
    *   SAM: `/ComfyUI/models/SAM`
    *   SAM2: `/ComfyUI/models/sam2`
    *   GroundingDINO: `/ComfyUI/models/grounding-dino`
    *   Clothes Segment: `/ComfyUI/models/RMBG/segformer_clothes`
    *   Fashion Segment: `/ComfyUI/models/RMBG/segformer_fashion`
    *   BiRefNet models: `/ComfyUI/models/RMBG/BiRefNet`
    *   SDMatte: `/ComfyUI/models/RMBG/SDMatte`

## Usage

### RMBG Node

1.  Load the `🧪AILab/🧽RMBG` node.
2.  Connect an image.
3.  Select a model.
4.  Adjust parameters (optional).
5.  Outputs: Processed image (with a transparent, black, white, green, blue, or red background) and a foreground mask.

### Optional Settings

| Setting             | Description                                                           | Tips                                                                                                   |
| ------------------- | --------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| Sensitivity         | Mask detection strength.                                              | Adjust for image complexity.                                                                           |
| Processing Resolution | Controls detail and memory usage.                                        | 256-2048. Higher resolutions mean better detail, but require more memory.                         |
| Mask Blur           | Blurs mask edges.                                                      | 1-5 for smoother edges.                                                                                |
| Mask Offset         | Expands or shrinks the mask.                                          | Fine-tune between -10 and 10.                                                                             |
| Background          | Choose background color.                                           | Alpha (transparent background), Black, White, Green, Blue, Red                                         |
| Invert Output       | Flip image and mask output.                                          |                                                                                                        |
| Refine Foreground | Optimize transparent background with Fast Foreground Color Estimation     | Enable for edge quality and transparency handling                                                         |
| Performance Optimization | Enhance performance when processing multiple images |  If memory allows, consider increasing `process_res` and `mask_blur` values for better results, but be mindful of memory usage.  |

### Basic Usage

1.  Load `RMBG (Remove Background)` node from the `🧪AILab/🧽RMBG` category
2.  Connect an image to the input
3.  Select a model from the dropdown menu
4.  select the parameters as needed (optional)
3.  Get two outputs:
   - IMAGE: Processed image with transparent, black, white, green, blue, or red background
   - MASK: Binary mask of the foreground

### Parameters

*   `sensitivity`: Controls background removal sensitivity (0.0-1.0)
*   `process_res`: Processing resolution (512-2048, step 128)
*   `mask_blur`: Blur amount for the mask (0-64)
*   `mask_offset`: Adjust mask edges (-20 to 20)
*   `background`: Choose output background color
*   `invert_output`: Flip mask and image output
*   `optimize`: Toggle model optimization

### Segment Node

1.  Load the `🧪AILab/🧽RMBG` node.
2.  Connect an image.
3.  Enter a text prompt.
4.  Select SAM or GroundingDINO models.
5.  Adjust parameters as needed: Threshold, mask blur, offset, and background color.

<details>
<summary><h2>About Models</h2></summary>
<!-- Model Descriptions - Keep it concise -->

## RMBG-2.0
BRIA AI with BiRefNet architecture. Includes high accuracy, edge preservation, handling of fine details, and support for multiple objects. Trained on 15,000+ images for robust performance.

## INSPYRENET
Specialized in human portrait segmentation. Offers fast processing and good edge detection.

## BEN
Provides a balance between speed and accuracy for various image types.

## BEN2
Improved accuracy and speed over BEN, better handling of complex scenes.

## BIREFNET MODELS
BiRefNet (general purpose matting), BiRefNet_512x512 (optimized for 512x512), BiRefNet-portrait (optimized for portrait), BiRefNet-matting (general purpose matting), BiRefNet-HR (high resolution up to 2560x2560), BiRefNet-HR-matting (high resolution matting), BiRefNet_lite (lightweight version for faster processing), BiRefNet_lite-2K (lightweight version for 2K resolution)

## SAM
A powerful model for object detection and segmentation.

## SAM2
Latest segmentation model family. Optimized inference with strong accuracy.

## GroundingDINO
Model for text-prompted object detection and segmentation.

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
*   Automatically installed packages (see original README for list)

### SDMatte models (manual download)

*   Auto-downloaded to `models/RMBG/SDMatte/`.
*   Manual placement if network restricted. Place weights: `models/RMBG/SDMatte/SDMatte.safetensors` (or `SDMatte_plus.safetensors`). Also include config files (mirror Hugging Face repo structure).

## Troubleshooting

*   **401 error / missing `models/sam2`:** Delete token cache, ensure no environment variables, then re-run.
*   **"Required input is missing: images"**: Ensure image outputs are connected and upstream nodes executed successfully.

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

⭐ If you find this useful, please star the repo!  Your support encourages continued development.

## License
GPL-3.0 License