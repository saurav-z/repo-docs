# ComfyUI-RMBG: Advanced Image Background Removal and Segmentation for ComfyUI

**Unlock unparalleled image editing capabilities with ComfyUI-RMBG, the ultimate ComfyUI custom node for precise background removal, object segmentation, and more.**  Access the original repo here:  [https://github.com/1038lab/ComfyUI-RMBG](https://github.com/1038lab/ComfyUI-RMBG)

## Key Features

*   **Multiple Models:** Utilize state-of-the-art models including RMBG-2.0, INSPYRENET, BEN, BEN2, BiRefNet-HR, SAM, and GroundingDINO for versatile image processing.
*   **Background Removal:** Easily remove backgrounds from images with multiple background color options (transparent, black, white, green, blue, red) and various model options.
*   **Object Segmentation:** Segment specific objects using text prompts, supporting tag-style and natural language inputs, powered by SAM and GroundingDINO.
*   **Facial Feature Segmentation:** Accurately segment faces into 19 categories, including skin, eyes, nose, and eyebrows.
*   **Clothes & Fashion Segmentation:** Isolate clothing and accessories with dedicated models and multiple category selection.
*   **Image Processing Tools:** Includes nodes for image combining, stitching, conversion, masking, and enhancement.
*   **Batch Processing Support:** Streamline your workflow by processing multiple images simultaneously.
*   **Flexible Parameter Controls:** Fine-tune results with adjustable sensitivity, processing resolution, mask blur, and offset settings.
*   **Easy Installation:**  Install via ComfyUI Manager, cloning the repository, or using the Comfy CLI.

## Recent Updates
*   **v2.7.1** (2025/08/06): [View Update Details](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v271-20250806)
    *   Enhanced LoadImage into three distinct nodes.
    *   Redesigned ImageStitch node.
    *   Fixed background color handling issues.
*   **v2.7.0** (2025/07/27): [View Update Details](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v270-20250727)
    *   Enhanced LoadImage into three distinct nodes.
    *   Redesigned ImageStitch node.
    *   Fixed background color handling issues.
*   **v2.6.0** (2025/07/15): [View Update Details](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v260-20250715)
    *   Added `Kontext Refence latent Mask` node.
*   **v2.5.2, v2.5.1, v2.5.0, v2.4.0, v2.3.2, v2.3.1, v2.3.0, v2.2.1, v2.2.0, v2.1.1, v2.1.0, v2.0.0, v1.9.3, v1.9.2, v1.9.1, v1.9.0, v1.8.0, v1.7.0, v1.6.0, v1.5.0, v1.4.0, v1.3.2, v1.3.1, v1.3.0, v1.2.2, v1.2.1, v1.2.0, v1.1.0** Added new features, improved performance and new models.

## Installation

Choose from the following methods:

*   **ComfyUI Manager:** Search for `Comfyui-RMBG` and install.
*   **Manual Clone:** Clone the repository into your ComfyUI `custom_nodes` folder and install requirements.
*   **Comfy CLI:** Use `comfy node install ComfyUI-RMBG`.

**Important:** After installation, ensure you install the required packages in the ComfyUI-RMBG folder by using the command:
  ```bash
  ./ComfyUI/python_embeded/python -m pip install -r requirements.txt
  ```

### Model Downloads
The models are automatically downloaded upon first use.  However, you can manually download models from the links provided in the  [About Models](#about-models) section of this document. Place the downloaded model files into the corresponding folders within your `ComfyUI/models/RMBG/` directory.

## Usage

### RMBG Node

1.  Load the `RMBG (Remove Background)` node from the `🧪AILab/🧽RMBG` category.
2.  Connect an image to the input.
3.  Select a model from the dropdown.
4.  Adjust parameters as needed (optional).

### Segment Node

1.  Load the `Segment (RMBG)` node from the `🧪AILab/🧽RMBG` category.
2.  Connect an image to the input.
3.  Enter a text prompt (tag-style or natural language).
4.  Select SAM or GroundingDINO models.
5.  Adjust parameters.

## Optional Settings

| Setting                 | Description                                                                 | Tips                                                                                                                      |
| :---------------------- | :-------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------ |
| **Sensitivity**         | Adjusts mask detection strength. Higher values for stricter detection.       | Default: 0.5. Adjust based on image complexity.                                                                           |
| **Processing Resolution** | Controls image processing resolution.                                      | Values: 256-2048 (step 128). Higher resolution = better detail, higher memory usage.                                     |
| **Mask Blur**           | Blurs mask edges for smoother results.                                         | Default: 0. Recommended: 1-5.                                                                                             |
| **Mask Offset**         | Expands or shrinks mask boundaries.                                          | Default: 0. Fine-tune between -10 and 10.                                                                                 |
| **Background**          | Select the desired output background color.                                 | Choose from: Alpha, Black, White, Green, Blue, Red.                                                                        |
| **Invert Output**       | Flips mask and image output.                                                 | Inverts both the image and mask.                                                                                           |
| **Refine Foreground**   | Uses Fast Foreground Color Estimation to optimize transparency.              | Enable for improved edge quality and transparency.                                                                        |
| **Performance Optimization** | Fine-tune settings for multiple image processing. | Consider increasing `process_res` and `mask_blur` with enough memory, but be mindful of memory usage. |

<details>
<summary><h2>About Models</h2></summary>
**RMBG-2.0**
  *   Developed by BRIA AI using the BiRefNet architecture.
  *   Key features: High accuracy, precise edge detection, fine detail handling, support for multiple objects, batch processing.
  *   Trained on a dataset of over 15,000 high-quality images.

**INSPYRENET**
  *   Specialized in human portrait segmentation.
  *   Fast processing with good edge detection.

**BEN & BEN2**
  *   Robust on various image types.
  *   BEN2 offers improved accuracy and speed.

**BiRefNet Models**
  *   A suite of models for image segmentation: general purpose, optimized for specific resolutions, portrait/human matting, and high-resolution (up to 2560x2560).

**SAM**
  *   Segment Anything Model for high accuracy object detection and segmentation.

**GroundingDINO**
  *   Text-prompted object detection and segmentation.

**Clothes Segment**
  *   Segment objects into clothes with 18 different categories.

**Fashion Segment**
  *   Segment objects by Fashion and Accessories.
</details>

## Requirements

*   ComfyUI
*   Python 3.10+
*   Required Packages (installed automatically): `torch>=2.0.0`, `torchvision>=0.15.0`, `Pillow>=9.0.0`, `numpy>=1.22.0`, `huggingface-hub>=0.19.0`, `tqdm>=4.65.0`, `transformers>=4.35.0`, `transparent-background>=1.2.4`, `opencv-python>=4.7.0`

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

## License

GPL-3.0 License