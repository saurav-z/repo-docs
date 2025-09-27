# ComfyUI-RMBG: Effortlessly Remove Backgrounds and Segment Images in ComfyUI

**Unlock the power of advanced image manipulation with ComfyUI-RMBG, a versatile custom node for ComfyUI that revolutionizes background removal, object segmentation, and more. [Visit the original repo for more details.](https://github.com/1038lab/ComfyUI-RMBG)**

## Key Features

*   **Advanced Background Removal:**
    *   Utilizes models like RMBG-2.0, INSPYRENET, BEN, BEN2, and BiRefNet for precise background removal.
    *   Offers various background options for versatile output.
    *   Supports batch processing for efficient workflow.
*   **Precise Object Segmentation:**
    *   Enables text-prompted object detection using tag-style or natural language inputs.
    *   Leverages SAM and GroundingDINO models for high-precision segmentation.
    *   Provides flexible parameter controls for refined results.
*   **Enhanced SAM2 Segmentation:**
    *   Integrates the latest SAM2 models (Tiny, Small, Base+, Large) for text-prompted segmentation.
    *   Features automatic model download with a manual placement option for flexibility.
*   **Innovative Real-Time Background Replacement:**
    *   New real-time background replacement capabilities for dynamic image editing.
*   **Improved Edge Detection:**
    *   Enhanced edge detection techniques for improved accuracy and detail preservation.
*   **Additional Features:**
    *   Includes nodes for image stitching, mask manipulation, and more.
    *   Offers a user-friendly interface for intuitive operation.
    *   Supports multiple image formats.

## News & Updates

*   **v2.9.1 (2025/09/12)** - See [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v291-20250912)
*   **v2.9.0 (2025/08/18)** - See [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v290-20250818) - Added `SDMatte Matting` node
*   **v2.8.0 (2025/08/11)** - See [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v280-20250811) - Added `SAM2Segment` node, Enhanced color widget support
*   **v2.7.1 (2025/08/06)** - See [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v271-20250806) - Enhanced LoadImage nodes, Redesigned ImageStitch node, Fixed background color issues
*   **v2.6.0 (2025/07/15)** - See [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v260-20250715) - Added `Kontext Refence latent Mask` node
*   **v2.5.2 (2025/07/11)** - See [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v252-20250711)
*   **v2.5.1 (2025/07/07)** - See [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v251-20250707)
*   **v2.5.0 (2025/07/01)** - See [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v250-20250701) - Added new nodes: `MaskOverlay`, `ObjectRemover`, `ImageMaskResize` and new BiRefNet models.
*   **v2.4.0 (2025/06/01)** - See [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v240-20250601) - Added `CropObject`, `ImageCompare`, `ColorInput` nodes.
*   **v2.3.2 (2025/05/15)** - See [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v232-20250515)
*   **v2.3.1 (2025/05/02)** - See [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v231-20250502)
*   **v2.3.0 (2025/05/01)** - See [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v230-20250501) - Added new nodes: IC-LoRA Concat, Image Crop and resizing options for Load Image
*   **v2.2.1 (2025/04/05)** - See [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v221-20250405)
*   **v2.2.0 (2025/04/05)** - See [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v220-20250405) - Added new nodes: Image Combiner, Image Stitch, Image/Mask Converter, Mask Enhancer, Mask Combiner, and Mask Extractor
*   **v2.1.1 (2025/03/21)** - See [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v211-20250321) - Enhanced compatibility with Transformers
*   **v2.1.0 (2025/03/19)** - See [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v210-20250319) - Integrated internationalization (i18n) support.
*   **v2.0.0 (2025/03/13)** - See [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v200-20250313) - Added Image and Mask Tools improved functionality.
*   **v1.9.3 (2025/02/24)** - See [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v193-20250224) - Clean up the code and fix the issue
*   **v1.9.2 (2025/02/21)** - See [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v192-20250221) - Fast Foreground Color Estimation
*   **v1.9.1 (2025/02/20)** - See [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v191-20250220) - Changed repository for model management
*   **v1.9.0 (2025/02/19)** - See [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v190-20250219) - BiRefNet model improvements
*   **v1.8.0 (2025/02/07)** - See [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v180-20250207) - new BiRefNet-HR model
*   **v1.7.0 (2025/02/04)** - See [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v170-20250204) - new BEN2 model
*   **v1.6.0 (2025/01/22)** - See [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v160-20250122) - new Face Segment custom node
*   **v1.5.0 (2025/01/05)** - See [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v150-20250105) - new Fashion and accessories Segment custom node
*   **v1.4.0 (2025/01/02)** - See [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v140-20250102) - new Clothes Segment node
*   **v1.3.2 (2024/12/29)** - See [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v132-20241229) - background handling
*   **v1.3.1 (2024/12/25)** - See [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v131-20241225) - bug fixes
*   **v1.3.0 (2024/12/23)** - See [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v140-20241222) - new Segment node
*   **v1.2.2 (2024/12/12)** - See [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v122-20241212)
*   **v1.2.1 (2024/12/02)** - See [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.mdv121-20241202)
*   **v1.2.0 (2024/11/29)** - See [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v120-20241129)
*   **v1.1.0 (2024/11/21)** - See [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v110-20241121)

## Installation

*   **Method 1: Install via ComfyUI Manager** - Search for `Comfyui-RMBG` and install directly through the ComfyUI Manager.

    *   Install `requirements.txt` in the ComfyUI-RMBG folder.
    ```bash
    ./ComfyUI/python_embeded/python -m pip install -r requirements.txt
    ```

*   **Method 2: Manual Clone**

    ```bash
    cd ComfyUI/custom_nodes
    git clone https://github.com/1038lab/ComfyUI-RMBG
    ```

    *   Install `requirements.txt` in the ComfyUI-RMBG folder.
    ```bash
    ./ComfyUI/python_embeded/python -m pip install -r requirements.txt
    ```

*   **Method 3: Install via Comfy CLI**

    ```bash
    comfy node install ComfyUI-RMBG
    ```

    *   Install `requirements.txt` in the ComfyUI-RMBG folder.
    ```bash
    ./ComfyUI/python_embeded/python -m pip install -r requirements.txt
    ```

*   **Model Download:**
    *   Models are automatically downloaded to `/ComfyUI/models/RMBG/` and `/ComfyUI/models/SAM/` on first use.
    *   Manual download options are also provided.

    *   Manually download the RMBG-2.0 model by visiting this [link](https://huggingface.co/1038lab/RMBG-2.0), then download the files and place them in the `/ComfyUI/models/RMBG/RMBG-2.0` folder.
    *   Manually download the INSPYRENET models by visiting the [link](https://huggingface.co/1038lab/inspyrenet), then download the files and place them in the `/ComfyUI/models/RMBG/INSPYRENET` folder.
    *   Manually download the BEN model by visiting the [link](https://huggingface.co/1038lab/BEN), then download the files and place them in the `/ComfyUI/models/RMBG/BEN` folder.
    *   Manually download the BEN2 model by visiting the [link](https://huggingface.co/1038lab/BEN2), then download the files and place them in the `/ComfyUI/models/RMBG/BEN2` folder.
    *   Manually download the BiRefNet-HR by visiting the [link](https://huggingface.co/1038lab/BiRefNet_HR), then download the files and place them in the `/ComfyUI/models/RMBG/BiRefNet-HR` folder.
    *   Manually download the SAM models by visiting the [link](https://huggingface.co/1038lab/sam), then download the files and place them in the `/ComfyUI/models/SAM` folder.
    *   Manually download the SAM2 models by visiting the [link](https://huggingface.co/1038lab/sam2), then download the files (e.g., `sam2.1_hiera_tiny.safetensors`, `sam2.1_hiera_small.safetensors`, `sam2.1_hiera_base_plus.safetensors`, `sam2.1_hiera_large.safetensors`) and place them in the `/ComfyUI/models/sam2` folder.
    *   Manually download the GroundingDINO models by visiting the [link](https://huggingface.co/1038lab/GroundingDINO), then download the files and place them in the `/ComfyUI/models/grounding-dino` folder.
    *   Manually download the Clothes Segment model by visiting the [link](https://huggingface.co/1038lab/segformer_clothes), then download the files and place them in the `/ComfyUI/models/RMBG/segformer_clothes` folder.
    *   Manually download the Fashion Segment model by visiting the [link](https://huggingface.co/1038lab/segformer_fashion), then download the files and place them in the `/ComfyUI/models/RMBG/segformer_fashion` folder.
    *   Manually download BiRefNet models by visiting the [link](https://huggingface.co/1038lab/BiRefNet), then download the files and place them in the `/ComfyUI/models/RMBG/BiRefNet` folder.
    *   Manually download SDMatte safetensors models by visiting the [link](https://huggingface.co/1038lab/SDMatte), then download the files and place them in the `/ComfyUI/models/RMBG/SDMatte` folder.

## Usage

### RMBG Node

1.  Load the `RMBG (Remove Background)` node from the `🧪AILab/🧽RMBG` category.
2.  Connect an image to the input.
3.  Select a model from the dropdown menu.
4.  Set parameters as needed (optional).
5.  Outputs:
    *   IMAGE: Processed image with a transparent, black, white, green, blue, or red background.
    *   MASK: Binary mask of the foreground.

### Optional Settings

| Optional Settings      | Description                                                              | Tips                                                                                       |
| ----------------------- | ------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------ |
| **Sensitivity**         | Adjusts mask detection strength. Higher values = stricter detection.       | Default: 0.5; adjust based on image complexity.                                            |
| **Processing Resolution** | Controls image processing resolution, impacting detail and memory usage. | Choose between 256 and 2048; default: 1024. Higher resolution improves detail.             |
| **Mask Blur**           | Controls the blur applied to mask edges.                               | Default: 0.  Try 1-5 for smoother edges.                                                  |
| **Mask Offset**         | Expands or shrinks mask boundary.                                         | Default: 0. Fine-tune between -10 and 10 based on the specific image.                     |
| **Background**          | Choose output background color.                                          | Alpha (transparent), Black, White, Green, Blue, Red.                                        |
| **Invert Output**       | Flip mask and image output.                                              | Invert both image and mask output.                                                         |
| **Refine Foreground**  | Use Fast Foreground Color Estimation for optimized transparent background. | Enable for improved edge quality and transparency handling.                               |
| **Performance Optimization** | Properly setting options can enhance performance when processing multiple images. | If memory allows, consider increasing `process_res` and `mask_blur` values for better results, but be mindful of memory usage. |

### Basic Usage

1.  Load the `RMBG (Remove Background)` node from the `🧪AILab/🧽RMBG` category.
2.  Connect an image to the input.
3.  Select a model from the dropdown menu.
4.  Adjust parameters as needed (optional).
5.  Obtain outputs: IMAGE (processed image), and MASK (foreground mask).

### Segment Node

1.  Load the `Segment (RMBG)` node from the `🧪AILab/🧽RMBG` category.
2.  Connect an image to the input.
3.  Enter a text prompt (tag-style or natural language).
4.  Select SAM and GroundingDINO models.
5.  Adjust parameters:
    *   Threshold: 0.25-0.35 (broad), 0.45-0.55 (precise).
    *   Mask blur and offset (edge refinement).
    *   Background color options.

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
*   Required packages (installed automatically):
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

*   **401 Error with GroundingDINO / Missing models/sam2:** Delete `%USERPROFILE%\.cache\huggingface\token` and ensure no `HF_TOKEN`/`HUGGINGFACE_TOKEN` environment variables are set.  Re-run; public repos download anonymously.
*   **Preview shows "Required input is missing: images":** Ensure image outputs are connected and upstream nodes ran successfully.

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