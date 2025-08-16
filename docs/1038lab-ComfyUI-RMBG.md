# ComfyUI-RMBG: Effortlessly Remove Backgrounds and Segment Images in ComfyUI

**Enhance your ComfyUI workflows with ComfyUI-RMBG, a powerful custom node for precise image background removal, object segmentation, and advanced image manipulation.**  [See the original repository here](https://github.com/1038lab/ComfyUI-RMBG).

## Key Features

*   **Advanced Background Removal:** Remove backgrounds with precision using multiple models like RMBG-2.0, INSPYRENET, BEN, and BEN2.
    *   Support for various output background options (transparent, black, white, etc.)
    *   Batch processing support for efficient workflow
*   **Text-Prompted Object Segmentation:** Accurately segment objects using text prompts with SAM and GroundingDINO models.
    *   Supports both tag-style ("cat, dog") and natural language prompts.
    *   Flexible parameter controls for refined segmentation.
*   **SAM2 Segmentation:** Utilize the latest SAM2 models for precise, text-prompted segmentation.
    *   Includes Tiny, Small, Base+, and Large SAM2 models.
    *   Automatic model download, with manual placement support.
*   **Face, Fashion, and Clothes Segmentation:** Dedicated nodes for specialized segmentation tasks.
    *   Face segmentation with 19 facial feature categories.
    *   Fashion segmentation for clothing and accessories.
    *   Clothes segmentation with 18 different categories.
*   **Enhanced Image Tools:** Improved functionality with new nodes for Image Combiner, Image Stitch, Mask Enhancer, and more.
*   **Real-Time Background Replacement and Edge Detection:** Enhanced edge detection, improved accuracy, and real-time background replacement.

## Updates

Stay up-to-date with the latest features and improvements:

*   **v2.8.0 (2025/08/11):** Added `SAM2Segment` node for text-prompted segmentation with the latest Facebook Research SAM2 technology and enhanced color widget support across all nodes.
*   **v2.7.1 (2025/08/06):** Enhanced LoadImage into three distinct nodes, redesigned ImageStitch node, and fixed background color handling issues.
*   **v2.6.0 (2025/07/15):** Added `Kontext Refence latent Mask` node.
*   **v2.5.2 (2025/07/11)**
*   **v2.5.1 (2025/07/07)**
*   **v2.5.0 (2025/07/01):** Added `MaskOverlay`, `ObjectRemover`, `ImageMaskResize` new nodes, added 2 BiRefNet models, and added batch image support.
*   **v2.4.0 (2025/06/01):** Added `CropObject`, `ImageCompare`, `ColorInput` nodes and new Segment V2.
*   **v2.3.2 (2025/05/15)**
*   **v2.3.1 (2025/05/02)**
*   **v2.3.0 (2025/05/01):** Added new nodes: IC-LoRA Concat, Image Crop and added resizing options for Load Image.
*   **v2.2.1 & v2.2.0 (2025/04/05):** Added new nodes: Image Combiner, Image Stitch, Image/Mask Converter, Mask Enhancer, Mask Combiner, and Mask Extractor; Fixed compatibility issues and i18n translation errors.
*   **v2.1.1 (2025/03/21):** Enhanced compatibility with Transformers.
*   **v2.1.0 (2025/03/19):** Integrated internationalization (i18n) support for multiple languages.
*   **v2.0.0 (2025/03/13):** Added Image and Mask Tools.
*   **v1.9.3 (2025/02/24):** Clean up the code and fix the issue
*   **v1.9.2 (2025/02/21):** Fast Foreground Color Estimation
*   **v1.9.1 (2025/02/20)**
*   **v1.9.0 (2025/02/19):** BiRefNet model improvements
*   **v1.8.0 (2025/02/07):** New BiRefNet-HR model
*   **v1.7.0 (2025/02/04):** New BEN2 model
*   **v1.6.0 (2025/01/22):** New Face Segment custom node
*   **v1.5.0 (2025/01/05):** New Fashion and accessories Segment custom node
*   **v1.4.0 (2025/01/02):** New Clothes Segment node
*   **v1.3.2 (2024/12/29):** Enhanced background handling
*   **v1.3.1 (2024/12/25):** Bug fixes
*   **v1.3.0 (2024/12/23):** New Segment node
*   **v1.2.2 (2024/12/12)**
*   **v1.2.1 (2024/12/02)**
*   **v1.2.0 (2024/11/29)**
*   **v1.1.0 (2024/11/21)**

## Installation

Choose your preferred method:

### 1. ComfyUI Manager Installation

Search for `Comfyui-RMBG` in the ComfyUI Manager and install. Then, install the requirements.txt in the ComfyUI-RMBG folder.

```bash
./ComfyUI/python_embeded/python -m pip install -r requirements.txt
```

> [!TIP]
> If you cannot install dependencies with the system Python, use ComfyUI's embedded Python.

### 2. Manual Clone

Clone the repository to your ComfyUI `custom_nodes` folder:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/1038lab/ComfyUI-RMBG
```

Install the requirements.txt in the ComfyUI-RMBG folder

```bash
./ComfyUI/python_embeded/python -m pip install -r requirements.txt
```

### 3. Comfy CLI

Install via Comfy CLI, if you have comfy-cli and ComfyUI installed.

```bash
comfy node install ComfyUI-RMBG
```

Install the requirements.txt in the ComfyUI-RMBG folder

```bash
./ComfyUI/python_embeded/python -m pip install -r requirements.txt
```

## 4. Model Downloads

The models are automatically downloaded to `ComfyUI/models/RMBG/` upon first use.  You can also manually download models from the following links and place them in their respective folders within `ComfyUI/models/RMBG/` or `ComfyUI/models/SAM/` :

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

## Usage

### RMBG Node

*   Load the `RMBG (Remove Background)` node from the `🧪AILab/🧽RMBG` category.
*   Connect an image to the input.
*   Select a model from the dropdown menu.
*   Adjust parameters as needed.
*   Outputs: Processed image and a mask.

### Segment Node

*   Load the `Segment (RMBG)` node from the `🧪AILab/🧽RMBG` category.
*   Connect an image to the input.
*   Enter a text prompt (tag-style or natural language).
*   Select SAM and GroundingDINO models.
*   Adjust parameters (Threshold, Mask blur, Mask offset, and Background color).

## Detailed Model Information

<details>
<summary><h2>About Models</h2></summary>

**(Expanded Model Descriptions Here)**

## RMBG-2.0

*   Developed by BRIA AI.
*   Uses the BiRefNet architecture.
*   Key features:
    *   High accuracy in complex environments
    *   Precise edge detection and preservation
    *   Excellent handling of fine details
    *   Support for multiple objects in a single image
    *   Output Comparison
    *   Output with background
    *   Batch output for video
*   Trained on a diverse dataset of over 15,000 high-quality images.

## INSPYRENET

*   Specialized in human portrait segmentation.
*   Key features:
    *   Fast processing speed
    *   Good edge detection capability
    *   Ideal for portrait photos and human subjects

## BEN

*   Robust on various image types.
*   Key features:
    *   Good balance between speed and accuracy
    *   Effective on both simple and complex scenes
    *   Suitable for batch processing

## BEN2

*   More advanced version of BEN.
*   Key features:
    *   Improved accuracy and speed
    *   Better handling of complex scenes
    *   Support for more image types
    *   Suitable for batch processing

## BIREFNET MODELS

*   Powerful model for image segmentation.
*   Key features include:
    *   BiRefNet-general purpose model (balanced performance)
    *   BiRefNet_512x512 model (optimized for 512x512 resolution)
    *   BiRefNet-portrait model (optimized for portrait/human matting)
    *   BiRefNet-matting model (general purpose matting)
    *   BiRefNet-HR model (high resolution up to 2560x2560)
    *   BiRefNet-HR-matting model (high resolution matting)
    *   BiRefNet_lite model (lightweight version for faster processing)
    *   BiRefNet_lite-2K model (lightweight version for 2K resolution)

## SAM

*   Powerful model for object detection and segmentation.
*   Key features:
    *   High accuracy in complex environments
    *   Precise edge detection and preservation
    *   Excellent handling of fine details
    *   Support for multiple objects in a single image
    *   Output Comparison
    *   Output with background
    *   Batch output for video

## SAM2

*   Latest segmentation model family
*   Key Features:
    *   Multiple sizes: Tiny, Small, Base+, Large
    *   Optimized inference with strong accuracy
    *   Automatic download on first use; manual placement supported in `ComfyUI/models/sam2`

## GroundingDINO

*   Model for text-prompted object detection and segmentation.
*   Key features:
    *   High accuracy in complex environments
    *   Precise edge detection and preservation
    *   Excellent handling of fine details
    *   Support for multiple objects in a single image
    *   Output Comparison
    *   Output with background
    *   Batch output for video

## BiRefNet Models

*   BiRefNet-general purpose model (balanced performance)
*   BiRefNet_512x512 model (optimized for 512x512 resolution)
*   BiRefNet-portrait model (optimized for portrait/human matting)
*   BiRefNet-matting model (general purpose matting)
*   BiRefNet-HR model (high resolution up to 2560x2560)
*   BiRefNet-HR-matting model (high resolution matting)
*   BiRefNet_lite model (lightweight version for faster processing)
*   BiRefNet_lite-2K model (lightweight version for 2K resolution)

</details>

## Usage Tips

| Parameter            | Description                                                                   | Tip                                                                                               |
|----------------------|-------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| **Sensitivity**      | Adjusts mask detection strength. Higher values = stricter detection.           | Default: 0.5. Adjust based on image complexity; use higher values for more complex images.     |
| **Processing Resolution** | Controls image processing resolution, affecting detail and memory usage.      | Choose: 256-2048 (step 128). Higher resolutions yield better detail but increase memory usage. |
| **Mask Blur**        | Controls mask edge blurring, reducing jaggedness.                               | Default: 0. Try values between 1-5 for smoother edges.                                             |
| **Mask Offset**      | Expands/shrinks the mask boundary. Positive values expand, negative values shrink. | Default: 0. Fine-tune between -10 and 10.                                                       |
| **Background**      | Selects the output background color.                                            | Alpha, Black, White, Green, Blue, Red.                                                                    |
| **Invert Output**      | Flips mask and image output.                                                | Invert both image and mask output.                                                                    |
| **Refine Foreground** | Uses Fast Foreground Color Estimation.                                                                        | Enable for better edge quality and transparency handling |
| **Performance Optimization** | Properly setting options can enhance performance when processing multiple images.                                                                       | If memory allows, consider increasing `process_res` and `mask_blur` values for better results, but be mindful of memory usage. |

## Requirements

*   ComfyUI
*   Python 3.10+
*   Required packages (installed automatically):  Check `requirements.txt` for specific versions.
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

## Credits

*   RMBG-2.0: [https://huggingface.co/briaai/RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0)
*   INSPYRENET: [https://github.com/plemeri/InSPyReNet](https://github.com/plemeri/InSPyReNet)
*   BEN: [https://huggingface.co/PramaLLC/BEN](https://huggingface.co/PramaLLC/BEN)
*   BEN2: [https://huggingface.co/PramaLLC/BEN2](https://huggingface.co/PramaLLC/BEN2)
*   BiRefNet: [https://huggingface.co/ZhengPeng7](https://huggingface.co/ZhengPeng7)
*   SAM: [https://huggingface.co/facebook/sam-vit-base](https://huggingface.co/facebook/sam-vit-base)
*   GroundingDINO: [https://github.com/IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
*   Clothes Segment: [https://huggingface.co/mattmdjaga/segformer_clothes](https://huggingface.co/mattmdjaga/segformer_clothes)

*   Created by: [AILab](https://github.com/1038lab)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=1038lab/comfyui-rmbg&type=Date)](https://star-history.com/#1038lab/comfyui-rmbg&Date)

If this custom node helps you, please consider giving the repo a ⭐!

## License

GPL-3.0 License