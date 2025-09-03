# Effortlessly Remove Backgrounds and Segment Images with ComfyUI-RMBG

ComfyUI-RMBG is a powerful custom node for ComfyUI, enabling advanced image background removal, object segmentation, and more.  [Check it out on GitHub!](https://github.com/1038lab/ComfyUI-RMBG)

## Key Features

*   **Background Removal:**
    *   Supports multiple models: RMBG-2.0, INSPYRENET, BEN, BEN2, BiRefNet, SDMatte.
    *   Flexible background options (transparent, color).
    *   Batch processing.
*   **Object Segmentation:**
    *   Text-prompted object detection using SAM and GroundingDINO.
    *   Support both tag-style and natural language inputs
    *   SAM2 Segmentation: Text-prompted segmentation with the latest Facebook Research SAM2 technology.
*   **Advanced Segmentation Models:**
    *   Face, Fashion, and Clothes segmentation nodes.
*   **Enhanced Features:**
    *   Improved edge detection for greater accuracy.
    *   Real-time background replacement.
    *   Mask overlay and masking tools.
*   **User-Friendly:**
    *   Easy installation via ComfyUI Manager.
    *   Automatic model downloads.
    *   Comprehensive documentation and usage examples.
    *   i18n support for multiple languages.

## Latest Updates

*   **v2.9.0** (2025/08/18): Added `SDMatte Matting` node
*   **v2.8.0** (2025/08/11): Added `SAM2Segment` node and enhanced color widget support.
*   **v2.7.1** (2025/08/06): Enhanced LoadImage node and redesigned ImageStitch node.
*   **(See the full history in the original README for even more updates!)**

## Installation

Choose your preferred method for easy installation:

### Method 1: ComfyUI Manager

1.  Open ComfyUI Manager.
2.  Search for "ComfyUI-RMBG".
3.  Click "Install".
4.  Restart ComfyUI.

### Method 2: Clone the Repository

1.  Navigate to your ComfyUI custom\_nodes folder: `cd ComfyUI/custom_nodes`
2.  Clone the repository: `git clone https://github.com/1038lab/ComfyUI-RMBG`
3.  Install requirements in the ComfyUI-RMBG folder: `./ComfyUI/python_embeded/python -m pip install -r requirements.txt`
4.  Restart ComfyUI.

### Method 3: Comfy CLI

1.  Ensure `pip install comfy-cli` is installed.
2.  Install ComfyUI if needed `comfy install`
3.  Install the ComfyUI-RMBG: `comfy node install ComfyUI-RMBG`
4.  Install requirements in the ComfyUI-RMBG folder: `./ComfyUI/python_embeded/python -m pip install -r requirements.txt`
5.  Restart ComfyUI.

### Model Downloads

*   Models download automatically when first used.
*   Manual Download:  See the original README for links and placement instructions.

## Usage

### RMBG Node (Background Removal)

1.  Load the `🧪AILab/🧽RMBG` node.
2.  Connect an image.
3.  Select a model.
4.  Adjust parameters (optional):  Sensitivity, Processing Resolution, Mask Blur, Mask Offset, Background, Invert Output, Refine Foreground, and Performance Optimization.
5.  Outputs: Processed image & Mask.

### Segment Node (Object Segmentation)

1.  Load the `🧪AILab/🧽RMBG` node.
2.  Connect an image.
3.  Enter a text prompt (tags or natural language).
4.  Select models (SAM and GroundingDINO).
5.  Adjust parameters: Threshold, Mask Blur, Mask Offset, Background Color.

## Model Details

*   **RMBG-2.0:** Excellent for complex scenes with detailed edge detection and preservation
*   **INSPYRENET:** Best for portrait segmentation.
*   **BEN & BEN2:** General-purpose models offering speed and accuracy.
*   **BiRefNet:** Powerful model with many variations including high-resolution.
*   **SAM & SAM2:** Powerful segmentation models.
*   **GroundingDINO:** Text-prompted object detection with high accuracy.
*   **(Full details about each model can be found in the original README's "About Models" section.)**

## Troubleshooting

*   Refer to the original README for common issues and solutions.

## Credits

*   See the original README for links to the creators of the models used.
*   Created by [AILab](https://github.com/1038lab).

## Star History

<a href="https://www.star-history.com/#1038lab/comfyui-rmbg&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=1038lab/comfyui-rmbg&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=1038lab/comfyui-rmbg&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=1038lab/comfyui-rmbg&type=Date" />
 </picture>
</a>

## License

*   GPL-3.0 License