# ComfyUI-RMBG: Advanced Image Background Removal and Segmentation

**Effortlessly remove backgrounds and precisely segment images within ComfyUI using a variety of powerful models, including SAM2 for the latest segmentation technology.**  [View the original repository on GitHub](https://github.com/1038lab/ComfyUI-RMBG).

## Key Features

*   **Background Removal:**
    *   Utilizes models like RMBG-2.0, INSPYRENET, BEN, and BEN2 for accurate background removal.
    *   Offers various background color options (Alpha, Black, White, Green, Blue, Red).
    *   Batch processing support for efficient workflow.
*   **Object Segmentation:**
    *   Segment objects using text prompts with the `Segment` node.
    *   Supports tag-style and natural language input.
    *   Includes SAM models for high-precision segmentation.
    *   Customize with flexible parameter controls.
*   **SAM2 Segmentation:**
    *   Leverages the latest SAM2 technology for text-prompted segmentation.
    *   Includes SAM2 models (Tiny, Small, Base+, Large).
    *   Automatic model download for ease of use.
*   **Enhanced Image Processing:**
    *   Improved edge detection and detail preservation.
    *   Optimized performance and memory management.
    *   Real-time background replacement capabilities.
*   **Model Support:**
    *   Extensive model support, including BiRefNet, SAM, and GroundingDINO.
    *   Dedicated nodes for Clothes, Fashion, and Face Segmentation
*   **User-Friendly Interface:**
    *   Easy installation via ComfyUI Manager or manual methods.
    *   Comprehensive documentation and usage tips.
    *   Regular updates and new features.

## What's New
*   **v2.8.0** (2025/08/11)
    *   Added `SAM2Segment` node for text-prompted segmentation with the latest Facebook Research SAM2 technology.
    *   Enhanced color widget support across all nodes
*   **v2.7.1** (2025/08/06)
    *   Enhanced LoadImage into three distinct nodes to meet different needs, all supporting direct image loading from local paths or URLs
    *   Completely redesigned ImageStitch node compatible with ComfyUI's native functionality
    *   Fixed background color handling issues reported by users
*   **v2.6.0** (2025/07/15)
    *   Added `Kontext Refence latent Mask` node, Which uses a reference latent and mask for precise region conditioning.

**(See the full update history in the original README for all version details.)**

## Installation

Choose your preferred installation method:

*   **ComfyUI Manager:** Search for `ComfyUI-RMBG` and install directly.
*   **Clone Repository:**
    1.  Navigate to your ComfyUI custom\_nodes directory: `cd ComfyUI/custom_nodes`
    2.  Clone the repository: `git clone https://github.com/1038lab/ComfyUI-RMBG`
    3.  Install dependencies:  `./ComfyUI/python_embeded/python -m pip install -r requirements.txt` (or use your ComfyUI's embedded Python)
*   **Comfy CLI:**  Use the command `comfy node install ComfyUI-RMBG`, then install requirements.

## Model Downloads

Models are automatically downloaded upon first use.  You can also manually download them and place them in the appropriate folders.  Links to model locations are listed in the original README.

## Usage

### RMBG Node

1.  Load the `RMBG (Remove Background)` node from the `🧪AILab/🧽RMBG` category.
2.  Connect an image input.
3.  Select a model.
4.  Adjust parameters as needed.
5.  Outputs: Processed image (with a transparent, black, white, green, blue, or red background) and a foreground mask.

### Segment Node

1.  Load the `Segment (RMBG)` node from the `🧪AILab/🧽RMBG` category.
2.  Connect an image input.
3.  Enter a text prompt (tag-style or natural language).
4.  Select SAM or GroundingDINO models.
5.  Adjust parameters (threshold, mask blur, mask offset, background).

## Optional Settings for Optimal Results

| Setting              | Description                                                                        | Tip                                                                                                     |
| :------------------- | :--------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------ |
| Sensitivity          | Controls mask detection strength.                                               | Adjust based on image complexity.  Higher values for more complex images.                               |
| Processing Resolution | Controls image processing resolution.                                             | Balance detail and memory usage.  Higher resolutions = more detail, but more memory.                    |
| Mask Blur            | Blurs mask edges.                                                                  | Set between 1-5 for smoother edges.                                                                    |
| Mask Offset          | Expands/shrinks the mask boundary.                                                | Adjust between -10 and 10 for fine-tuning.                                                             |
| Background           | Choose output background color.                                                     | Select "Alpha" for a transparent background.                                                             |
| Invert Output        | Inverts the mask and image output.                                                 | If you want the background, use the invert.                                                             |
| Refine Foreground    | Improves transparency handling.                                                    | Enable for better edge quality.                                                                         |
| Optimize  | Optimize the model if you have a GPU                                          | Consider increasing `process_res` and `mask_blur` values for better results, but be mindful of memory usage. |

## About Models
**(Expanded information about the models, including RMBG-2.0, INSPYRENET, BEN, BEN2, BiRefNet, SAM, SAM2, GroundingDINO, can be found in the original README.)**

## Requirements

*   ComfyUI
*   Python 3.10+
*   Install dependencies:  `./ComfyUI/python_embeded/python -m pip install -r requirements.txt`

## Credits

*   Created by [AILab](https://github.com/1038lab)
*   (And links to the models, which are in the original README)

## Star History

**(The Star History graph is included in the original README.)**

## License

GPL-3.0 License