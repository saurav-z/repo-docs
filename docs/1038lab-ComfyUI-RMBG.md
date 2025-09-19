# ComfyUI-RMBG: Effortlessly Remove Backgrounds and Segment Images in ComfyUI

**Unleash the power of AI to precisely remove backgrounds, segment objects, and refine images within ComfyUI!**  Explore the full potential of image manipulation with this comprehensive custom node.  [See the original repo](https://github.com/1038lab/ComfyUI-RMBG) for more details.

## Key Features

*   **Advanced Background Removal:**  Utilize cutting-edge models like RMBG-2.0, INSPYRENET, BEN, and BEN2 for precise background removal.
*   **Object Segmentation:**  Segment images using text prompts with models like SAM and GroundingDINO, supporting both tag-style and natural language inputs.
*   **SAM2 Segmentation:** Leverage the latest SAM2 models (Tiny/Small/Base+/Large) for high-quality, text-prompted segmentation.
*   **Comprehensive Model Support:**  Access a wide array of models, including BiRefNet variants, SDMatte, and more.
*   **Real-time Background Replacement & Enhanced Edge Detection:** Enjoy a seamless image refinement experience with improved accuracy.
*   **New and Updated Nodes:** CropObject, ImageCompare, ColorInput nodes and new Segment V2.  Added TextPrompted object segmentation with multiple models SAM (vit\_h/l/b) and GroundingDINO (SwinT/B)
*   **Flexible Options:** Fine-tune results with adjustable sensitivity, resolution, mask blurring, and offset controls.

## Updates

*   **(2025/09/12) v2.9.1:** Update ComfyUI-RMBG to v2.9.1
*   **(2025/08/18) v2.9.0:** Added `SDMatte Matting` node
*   **(2025/08/11) v2.8.0:** Added `SAM2Segment` node, Enhanced color widget support
*   **(2025/08/06) v2.7.1:** Enhanced LoadImage, Redesigned ImageStitch node
*   **(2025/07/15) v2.6.0:** Added `Kontext Refence latent Mask` node
*   **(2025/07/01) v2.5.0:** Added `MaskOverlay`, `ObjectRemover`, `ImageMaskResize` new nodes. Added 2 BiRefNet models: `BiRefNet_lite-matting` and `BiRefNet_dynamic`
*   **(2025/06/01) v2.4.0:** Added `CropObject`, `ImageCompare`, `ColorInput` nodes and new Segment V2
*   **(2025/05/15) v2.3.2:** Update ComfyUI-RMBG to v2.3.2
*   **(2025/05/01) v2.3.0:** Added new nodes: IC-LoRA Concat, Image Crop, resizing options for Load Image
*   **(2025/04/05) v2.2.0:** Added new nodes: Image Combiner, Image Stitch, Image/Mask Converter, Mask Enhancer, Mask Combiner, and Mask Extractor
*   **(2025/03/19) v2.1.0:** Added internationalization (i18n) support for multiple languages.
*   **(2025/03/13) v2.0.0:** Added Image and Mask Tools improved functionality, new category path: `🧪AILab/🛠️UTIL/🖼️IMAGE`.
*   **(2025/02/21) v1.9.2:** Added new foreground refinement feature for better transparency handling
*   **(2025/02/19) v1.9.0:** Enhanced BiRefNet model performance and stability
*   **(2025/02/07) v1.8.0:** Added a new custom node for BiRefNet-HR model.
*   **(2025/02/04) v1.7.0:** Added a new custom node for BEN2 model.
*   **(2025/01/22) v1.6.0:** Added a new custom node for face parsing and segmentation
*   **(2025/01/05) v1.5.0:** Added a new custom node for fashion segmentation.
*   **(2025/01/02) v1.4.0:** Added intelligent clothes segmentation with 18 different categories
*   **(2024/12/23) v1.3.0:** Added text-prompted object segmentation
*   **(2024/12/12) v1.2.2:** Update Comfyui-RMBG ComfyUI Custom Node to v1.2.2
*   **(2024/12/02) v1.2.1:** Update Comfyui-RMBG ComfyUI Custom Node to v1.2.1
*   **(2024/11/29) v1.2.0:** Update Comfyui-RMBG ComfyUI Custom Node to v1.2.0
*   **(2024/11/21) v1.1.0:** Update Comfyui-RMBG ComfyUI Custom Node to v1.1.0

## Installation

Choose your preferred installation method:

### Method 1: ComfyUI Manager

*   Install via ComfyUI Manager by searching for `Comfyui-RMBG` and installing directly.
*   Install requirements using the ComfyUI embedded Python:
    ```bash
    ./ComfyUI/python_embeded/python -m pip install -r requirements.txt
    ```

### Method 2: Manual Cloning

1.  Navigate to your ComfyUI custom\_nodes directory:

    ```bash
    cd ComfyUI/custom_nodes
    ```
2.  Clone the repository:

    ```bash
    git clone https://github.com/1038lab/ComfyUI-RMBG
    ```
3.  Install dependencies (using the embedded Python):

    ```bash
    ./ComfyUI/python_embeded/python -m pip install -r requirements.txt
    ```

### Method 3: Comfy CLI

1.  Ensure `pip install comfy-cli` is installed.
2.  Install ComfyUI if you don't have it.
3.  Install the ComfyUI-RMBG:

    ```bash
    comfy node install ComfyUI-RMBG
    ```
4.  Install requirements:

    ```bash
    ./ComfyUI/python_embeded/python -m pip install -r requirements.txt
    ```

### Model Downloads

*   Models are auto-downloaded on first use and located in the appropriate folder under `ComfyUI/models/RMBG/` (or `/SAM`, `/grounding-dino`, etc.).  If you have any download issues, models can be manually downloaded.

*   **RMBG-2.0:** [RMBG-2.0](https://huggingface.co/1038lab/RMBG-2.0) in `/ComfyUI/models/RMBG/RMBG-2.0`
*   **INSPYRENET:** [INSPYRENET](https://huggingface.co/1038lab/inspyrenet) in `/ComfyUI/models/RMBG/INSPYRENET`
*   **BEN:** [BEN](https://huggingface.co/1038lab/BEN) in `/ComfyUI/models/RMBG/BEN`
*   **BEN2:** [BEN2](https://huggingface.co/1038lab/BEN2) in `/ComfyUI/models/RMBG/BEN2`
*   **BiRefNet-HR:** [BiRefNet\_HR](https://huggingface.co/1038lab/BiRefNet_HR) in `/ComfyUI/models/RMBG/BiRefNet-HR`
*   **SAM:** [SAM](https://huggingface.co/1038lab/sam) in `/ComfyUI/models/SAM`
*   **SAM2:** [SAM2](https://huggingface.co/1038lab/sam2) in `/ComfyUI/models/sam2`
*   **GroundingDINO:** [GroundingDINO](https://huggingface.co/1038lab/GroundingDINO) in `/ComfyUI/models/grounding-dino`
*   **Clothes Segment:** [segformer\_clothes](https://huggingface.co/1038lab/segformer_clothes) in `/ComfyUI/models/RMBG/segformer_clothes`
*   **Fashion Segment:** [segformer\_fashion](https://huggingface.co/1038lab/segformer_fashion) in `/ComfyUI/models/RMBG/segformer_fashion`
*   **BiRefNet:** [BiRefNet](https://huggingface.co/1038lab/BiRefNet) in `/ComfyUI/models/RMBG/BiRefNet`
*   **SDMatte:** [SDMatte](https://huggingface.co/1038lab/SDMatte) in `/ComfyUI/models/RMBG/SDMatte`

## Usage

### RMBG Node

*   Load the `RMBG (Remove Background)` node from the `🧪AILab/🧽RMBG` category.
*   Connect an image to the input.
*   Select a model from the dropdown menu.
*   Adjust optional parameters as needed.
*   Outputs: Processed image and a binary mask.

### Optional Settings :bulb: Tips

| Optional Settings       | :memo: Description                                                           | :bulb: Tips                                                                                   |
|-------------------------|-----------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| **Sensitivity**         | Adjusts mask detection strength.  Higher values = stricter detection.        | Default: 0.5.  Adjust for image complexity.                                                    |
| **Processing Resolution**| Controls processing resolution for detail and memory usage.                | Between 256-2048, Default: 1024. Higher = more detail, more memory.                           |
| **Mask Blur**           | Blurs mask edges, reducing jaggedness.                                      | Default: 0. Try 1-5 for smoother edges.                                                        |
| **Mask Offset**         | Expands or shrinks mask boundary.  Positive = expand, negative = shrink.   | Default: 0. Fine-tune between -10 and 10.                                                       |
| **Background**          | Choose output background color | Alpha (transparent background) Black, White, Green, Blue, Red |
| **Invert Output**       | Flip mask and image output | Invert both image and mask output |
| **Refine Foreground**   | Use Fast Foreground Color Estimation to optimize transparent background | Enable for better edge quality and transparency handling |
| **Performance Optimization** | Properly setting options can enhance performance when processing multiple images. | If memory allows, consider increasing `process_res` and `mask_blur` values for better results, but be mindful of memory usage. |

### Basic Usage

1.  Load `RMBG (Remove Background)` from `🧪AILab/🧽RMBG`.
2.  Connect an image to the input.
3.  Select a model.
4.  Adjust parameters (optional).
5.  Outputs: Processed image (with background) and a mask.

### Parameters

*   `sensitivity`: Background removal sensitivity (0.0-1.0)
*   `process_res`: Processing resolution (512-2048, step 128)
*   `mask_blur`: Blur amount (0-64)
*   `mask_offset`: Adjust mask edges (-20 to 20)
*   `background`: Background color choice
*   `invert_output`: Flip mask and image output
*   `optimize`: Toggle model optimization

### Segment Node

1.  Load `Segment (RMBG)` from `🧪AILab/🧽RMBG`.
2.  Connect an image.
3.  Enter a text prompt (tag-style or natural language).
4.  Select SAM or GroundingDINO models.
5.  Adjust parameters as needed: threshold, mask blur, offset, and background color.

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
*   Automatically installed packages: `huggingface-hub>=0.19.0`, `transparent-background>=1.1.2`, `segment-anything>=1.0`, `groundingdino-py>=0.4.0`, `opencv-python>=4.7.0`, `onnxruntime>=1.15.0`, `onnxruntime-gpu>=1.15.0`, `protobuf>=3.20.2,<6.0.0`, `hydra-core>=1.3.0`, `omegaconf>=2.3.0`, `iopath>=0.1.9`

### SDMatte models (manual download)

*   Auto-download on first run to `models/RMBG/SDMatte/`
*   Manual download:
    *   `models/RMBG/SDMatte/SDMatte.safetensors` (standard) or `SDMatte_plus.safetensors` (plus)
    *   Components (config files) auto-downloaded; mirror the Hugging Face repo structure to `models/RMBG/SDMatte/` (`scheduler/`, `text_encoder/`, `tokenizer/`, `unet/`, `vae/`)

## Troubleshooting

*   **401 Error:** Delete `%USERPROFILE%\.cache\huggingface\token` (and `.huggingface\token` if present). Ensure no `HF_TOKEN`/`HUGGINGFACE_TOKEN` env vars are set. Then re-run. Public repos download anonymously (no login required).
*   **"Required input is missing: images":** Ensure image outputs are connected and upstream nodes ran successfully.

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

⭐ If you find this custom node helpful, please star the repository! ⭐

## License

GPL-3.0 License