# ComfyUI-RMBG: Effortlessly Remove Backgrounds and Segment Images in ComfyUI

**Unlock advanced image editing capabilities with ComfyUI-RMBG, a powerful custom node for precise background removal, object segmentation, and more.** [View on GitHub](https://github.com/1038lab/ComfyUI-RMBG)

## Key Features

*   **Advanced Background Removal:** Utilize multiple models (RMBG-2.0, INSPYRENET, BEN, BEN2) for accurate and versatile background removal.
*   **Precise Object Segmentation:** Segment objects using text prompts with SAM and GroundingDINO models, supporting both tag-style and natural language inputs.
*   **SAM2 Segmentation:** Leverage the latest SAM2 models (Tiny/Small/Base+/Large) for high-quality, text-prompted segmentation.
*   **Real-time Background Replacement:** Easily replace backgrounds with custom colors or images.
*   **Enhanced Edge Detection:** Refine mask edges for improved accuracy and smoother results.
*   **Face, Clothes, and Fashion Segmentation:** Dedicated nodes for detailed facial feature, clothing, and fashion element segmentation.
*   **High-Resolution Support:** Process images up to 2048x2048 with selected models.
*   **Batch Processing:** Process multiple images at once, streamlining your workflow.
*   **User-Friendly Interface:** Intuitive controls and optional settings for customization.
*   **Regular Updates:** Stay up-to-date with the latest advancements and models.

## Recent Updates

*   **v2.9.0 (2025/08/18):** Added `SDMatte Matting` node.
*   **v2.8.0 (2025/08/11):** Added `SAM2Segment` node for text-prompted segmentation with the latest Facebook Research SAM2 technology and enhanced color widget support across all nodes.
*   **v2.7.1 (2025/08/06):** Enhanced LoadImage and ImageStitch nodes
*   **(See the original README for a detailed changelog)**

## Installation

### Method 1. Install via ComfyUI Manager
   Search `Comfyui-RMBG` and install. Install requirements using the following command in the ComfyUI-RMBG folder:
  ```bash
  ./ComfyUI/python_embeded/python -m pip install -r requirements.txt
  ```

### Method 2. Clone the repository
   Clone the repository to your ComfyUI `custom_nodes` folder:

  ```bash
  cd ComfyUI/custom_nodes
  git clone https://github.com/1038lab/ComfyUI-RMBG
  ```

  Install requirements using the following command in the ComfyUI-RMBG folder:

  ```bash
  ./ComfyUI/python_embeded/python -m pip install -r requirements.txt
  ```
### Method 3: Install via Comfy CLI
  Ensure `pip install comfy-cli` is installed.
  Installing ComfyUI `comfy install` (if you don't have ComfyUI Installed)
  install the ComfyUI-RMBG, use the following command:
  ```bash
  comfy node install ComfyUI-RMBG
  ```
  install requirment.txt in the ComfyUI-RMBG folder
  ```bash
  ./ComfyUI/python_embeded/python -m pip install -r requirements.txt
  ```
### Model Downloads
   - Models will be automatically downloaded. 
   - Manual download options are available (see the original README for specific model download links and instructions).

## Usage

### RMBG Node
   1.  Load the `RMBG (Remove Background)` node from the `🧪AILab/🧽RMBG` category.
   2.  Connect an image to the input.
   3.  Select a model from the dropdown.
   4.  Adjust optional parameters as needed.
   5.  Outputs an image with a transparent, black, white, green, blue, or red background, and the binary mask.

### Segment Node
   1.  Load the `Segment (RMBG)` node from the `🧪AILab/🧽RMBG` category.
   2.  Connect an image to the input.
   3.  Enter a text prompt (tag-style or natural language).
   4.  Select SAM and GroundingDINO models.
   5.  Adjust parameters as needed (threshold, mask blur, offset).

## About Models (See the original README for detailed information)

*   **RMBG-2.0**
*   **INSPYRENET**
*   **BEN**
*   **BEN2**
*   **BiRefNet Models**
*   **SAM**
*   **SAM2**
*   **GroundingDINO**
*   **Clothes Segment**
*   **SDMatte**
*   **Fashion Segment**

## Requirements

*   ComfyUI
*   Python 3.10+
*   Dependencies are automatically installed during installation.

## Troubleshooting (See the original README for detailed information)

## Credits

*   [AILab](https://github.com/1038lab)
*   (See the original README for model creators)

## Star History

```html
<a href="https://www.star-history.com/#1038lab/comfyui-rmbg&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=1038lab/comfyui-rmbg&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=1038lab/comfyui-rmbg&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=1038lab/comfyui-rmbg&type=Date" />
 </picture>
</a>
```

## License

GPL-3.0 License