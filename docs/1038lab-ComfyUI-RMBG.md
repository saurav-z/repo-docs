# Enhance Your Images with Advanced Background Removal and Segmentation in ComfyUI!

Tired of tedious background removal? **ComfyUI-RMBG** offers a suite of cutting-edge custom nodes for ComfyUI, providing unparalleled image editing capabilities. Visit the original repo for more details: [https://github.com/1038lab/ComfyUI-RMBG](https://github.com/1038lab/ComfyUI-RMBG).

## Key Features

*   **Multiple Models:** Leverage state-of-the-art models including RMBG-2.0, INSPYRENET, BEN, BEN2, BiRefNet-HR, SAM, and GroundingDINO for superior background removal and segmentation.
*   **Precise Segmentation:** Accurately segment objects, faces, clothes, and fashion items using text prompts or model-based analysis.
*   **Flexible Background Options:** Choose transparent backgrounds (Alpha), or solid colors including black, white, green, blue, and red.
*   **Batch Processing:** Process multiple images efficiently.
*   **User-Friendly Interface:** Easily integrate the nodes into your ComfyUI workflows.
*   **Regular Updates:** Benefit from frequent updates with new features and model improvements.

## What's New? (Recent Updates)

*   **v2.7.1 (2025/08/06):** Bug fixes and improvements.
*   **v2.7.0 (2025/07/27):** Enhanced LoadImage nodes, redesigned ImageStitch node, and background color handling fixes.
    *   ![v2.7.0_ImageStitch](https://github.com/user-attachments/assets/3f31fe25-a453-4f86-bf3d-dc12a8affd39)
*   **v2.6.0 (2025/07/15):** Added `Kontext Refence latent Mask` node.
    *   ![ReferenceLatentMaskr](https://github.com/user-attachments/assets/756641b7-0833-4fe0-b32f-2b848a14574e)

*(See the original README for more detailed update information and image examples.)*

## Installation

Choose one of these easy installation methods:

*   **ComfyUI Manager:** Install directly from the ComfyUI Manager by searching for `Comfyui-RMBG`.  Then install the requirements.txt file.
*   **Manual Clone:**
    1.  Navigate to your ComfyUI custom nodes folder: `cd ComfyUI/custom_nodes`
    2.  Clone the repository: `git clone https://github.com/1038lab/ComfyUI-RMBG`
    3.  Install requirements:  `./ComfyUI/python_embeded/python -m pip install -r requirements.txt`
*   **Comfy CLI:** Use the command `comfy node install ComfyUI-RMBG` and install the requirements.txt file.

*   **Model Downloads:**
    *   The models will download automatically when used. However, manual download is also possible following the links provided in the original README.

## Usage

### RMBG Node (Background Removal)

1.  Load the `RMBG (Remove Background)` node from the `🧪AILab/🧽RMBG` category.
2.  Connect an image to the input.
3.  Select a model.
4.  Adjust parameters as needed (Sensitivity, Processing Resolution, Mask Blur, Mask Offset, Background Color, Invert Output, Refine Foreground).
5.  Get two outputs: processed image and a mask of the foreground.

    ![RMBG](https://github.com/user-attachments/assets/cd0eb92e-8f2e-4ae4-95f1-899a6d83cab6)

### Segment Node (Object Segmentation)

1.  Load the `Segment (RMBG)` node from the `🧪AILab/🧽RMBG` category.
2.  Connect an image.
3.  Enter a text prompt (tag-style or natural language).
4.  Select SAM or GroundingDINO models.
5.  Adjust threshold, mask blur, offset, and background color.

    ![RMBG Demo](https://github.com/user-attachments/assets/f3ffa3c4-5a21-4c0c-a078-b4ffe681c4c4)

## Models

ComfyUI-RMBG supports a variety of models, including:

*   **Background Removal:** RMBG-2.0, INSPYRENET, BEN, BEN2
*   **Segmentation:** SAM (vit\_h/l/b), GroundingDINO (SwinT/B), BiRefNet.
*   **Specialized Segmentation:** Face, Fashion, and Clothes segmentation.

*(See the original README for detailed descriptions of each model)*

## Requirements

*   ComfyUI
*   Python 3.10+
*   Dependencies will be automatically installed.

## Credits

*   Developed by [AILab](https://github.com/1038lab).
*   This project uses models and code from various sources. (See the original README for detailed credits.)

## Star History

<a href="https://www.star-history.com/#1038lab/comfyui-rmbg&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=1038lab/comfyui-rmbg&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=1038lab/comfyui-rmbg&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=1038lab/comfyui-rmbg&type=Date" />
 </picture>
</a>

**If you find this custom node helpful, please consider giving it a ⭐ on GitHub!**

## License

GPL-3.0 License