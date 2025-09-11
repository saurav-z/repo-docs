# ComfyUI-RMBG: Effortlessly Remove Backgrounds and Segment Images

Tired of manual background removal?  ComfyUI-RMBG is a powerful custom node for ComfyUI, offering advanced image background removal, object segmentation, and more, all within an intuitive workflow. ([Original Repo](https://github.com/1038lab/ComfyUI-RMBG))

## Key Features

*   **Advanced Background Removal:** Utilize a variety of models, including RMBG-2.0, INSPYRENET, BEN, BEN2, BiRefNet models, and SDMatte models for precise background removal.
*   **Precise Object Segmentation:** Segment objects, faces, clothing, and fashion elements with text-prompted segmentation using SAM, SAM2, and GroundingDINO.
*   **Real-Time Background Replacement:** Instantly replace backgrounds with customizable options.
*   **Enhanced Edge Detection:** Achieve higher accuracy with improved edge detection.
*   **Flexible Image Processing:** Supports batch processing, adjustable parameters for sensitivity and resolution, and a range of background color options (including transparency).
*   **Multiple Model Support:** Access a variety of segmentation models, including Segment Anything (SAM) and Grounding DINO.
*   **Face, Clothes and Fashion Segmentation:** Dedicated models for facial feature extraction, clothing, and fashion item segmentation.

## Recent Updates

*   **v2.9.0 (2025/08/18):** Added `SDMatte Matting` node.
*   **v2.8.0 (2025/08/11):** Added `SAM2Segment` node for text-prompted segmentation using SAM2 technology and enhanced color widget support.
*   **v2.7.1 (2025/08/06):** Enhanced image loading, redesigned image stitch node, and fixed background color handling.
*   **v2.6.0 (2025/07/15):** Added `Kontext Refence latent Mask` node.
*   (See the original README for a complete update log)

## Installation

### Method 1. Install via ComfyUI Manager

Search for `Comfyui-RMBG` in the ComfyUI Manager and install.  Then install requirements.txt in the ComfyUI-RMBG folder.

  ```bash
  ./ComfyUI/python_embeded/python -m pip install -r requirements.txt
  ```

### Method 2. Clone the Repository

1.  Navigate to your ComfyUI custom\_nodes directory:  `cd ComfyUI/custom_nodes`
2.  Clone the repository: `git clone https://github.com/1038lab/ComfyUI-RMBG`
3.  Install the requirements: `./ComfyUI/python_embeded/python -m pip install -r requirements.txt`

### Method 3: Install via Comfy CLI

1.  Ensure `pip install comfy-cli` is installed.
2.  Install ComfyUI: `comfy install` (if not already installed)
3.  Install the ComfyUI-RMBG: `comfy node install ComfyUI-RMBG`
4.  Install the requirements: `./ComfyUI/python_embeded/python -m pip install -r requirements.txt`

### Model Downloads

*   Models are automatically downloaded to the correct directory the first time you use a custom node.
*   Manual download options are available. See the original README for model-specific download links and instructions.

## Usage

### RMBG Node

1.  Load the `RMBG (Remove Background)` node from the `🧪AILab/🧽RMBG` category.
2.  Connect an image to the input.
3.  Select a model.
4.  (Optional) Adjust parameters such as sensitivity, processing resolution, mask blur, and background color.
5.  Output: Processed image with transparent, black, white, green, blue, or red background + mask.

### Segment Node

1.  Load the `Segment (RMBG)` node from the `🧪AILab/🧽RMBG` category.
2.  Connect an image to the input.
3.  Enter a text prompt (tag-style or natural language).
4.  Select a model (SAM, GroundingDINO).
5.  Adjust parameters (threshold, mask blur, offset, background color).

## Troubleshooting

*   Refer to the original README for troubleshooting tips.

## Credits

*   See the original README for model authors and contributors.
*   Created by: [AILab](https://github.com/1038lab)

## Star History
[Include Star History here, as in the original README]

## License

GPL-3.0 License