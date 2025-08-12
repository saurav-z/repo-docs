# ComfyUI-RMBG: Effortlessly Remove Backgrounds and Segment Images in ComfyUI

**Enhance your image editing workflow with ComfyUI-RMBG, a powerful custom node offering advanced background removal, object segmentation, and precise control within ComfyUI.  [Check out the original repository here](https://github.com/1038lab/ComfyUI-RMBG)!**

## Key Features

*   **Background Removal:**
    *   Multiple models: RMBG-2.0, INSPYRENET, BEN, BEN2, BiRefNet.
    *   Versatile background options (transparent, solid colors).
    *   Batch processing support for efficient workflows.
*   **Object Segmentation:**
    *   Text-prompted object detection using tags or natural language.
    *   High-precision segmentation with SAM and SAM2.
    *   Customizable parameters for fine-tuning results.
    *   Flexible parameter controls.
*   **SAM2 Segmentation:**
    *   Leverages the latest SAM2 models (Tiny/Small/Base+/Large).
    *   Automated model download and manual installation options.
*   **Real-time Background Replacement:** Enhance your image by automatically changing the background in real-time.
*   **Edge Detection:** Enhanced edge detection that gives you the most accurate results for both foreground and background separation.

## What's New

Stay up-to-date with the latest features and improvements:

*   **v2.8.0 (2025/08/11):**
    *   Added `SAM2Segment` node for text-prompted segmentation with the latest Facebook Research SAM2 technology.
    *   Enhanced color widget support across all nodes.

*   **[See the full update history here](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md) for more updates.**

## Installation

Choose your preferred installation method:

1.  **ComfyUI Manager:** Search for `Comfyui-RMBG` and install directly through the ComfyUI Manager.
2.  **Manual Clone:**
    ```bash
    cd ComfyUI/custom_nodes
    git clone https://github.com/1038lab/ComfyUI-RMBG
    ```
3.  **Comfy CLI:**
    ```bash
    comfy node install ComfyUI-RMBG
    ```

    *   **Important:** After installation, ensure you install the required packages:
        ```bash
        ./ComfyUI/python_embeded/python -m pip install -r requirements.txt
        ```
        (or use your system's Python if the embedded one doesn't work)

4.  **Model Downloads:** The models are automatically downloaded upon first use.  If manual download is needed, see the model links in the [original README](https://github.com/1038lab/ComfyUI-RMBG).

## Usage

### 1. RMBG Node (Background Removal)

*   Load the `RMBG (Remove Background)` node from the `🧪AILab/🧽RMBG` category.
*   Connect an image to the input.
*   Select a model from the dropdown menu.
*   Adjust optional parameters for fine-tuning (see below).
*   Outputs: Processed image (with background removal) and a mask.

### 2. Segment Node (Object Segmentation)

*   Load the `Segment (RMBG)` node from the `🧪AILab/🧽RMBG` category.
*   Connect an image to the input.
*   Enter a text prompt (tag-style or natural language).
*   Select the desired model (SAM, GroundingDINO, SAM2).
*   Adjust parameters as needed for optimal results.

### Optional Settings: Enhance Your Results

| Setting                  | Description                                                                | Tip                                                                                                  |
| ------------------------ | -------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| Sensitivity              | Controls mask detection strength.  Higher = stricter.                        | Adjust based on image complexity.  Default: 0.5.                                                   |
| Processing Resolution    | Controls detail and memory usage.                                           | Range: 256-2048.  Default: 1024.  Higher = more detail, more memory.                             |
| Mask Blur                | Blurs mask edges.                                                           | Default: 0.  Experiment with 1-5 for smoother edges.                                                |
| Mask Offset              | Expands or shrinks mask boundary.                                           | Default: 0.  Fine-tune between -10 and 10.                                                           |
| Background               | Choose output background color.                                          | Select from Alpha (transparent), Black, White, Green, Blue, or Red.                                  |
| Invert Output            | Flips mask and image output.                                               | Inverts both image and mask output.                                                                   |
| Refine Foreground        | Optimize transparent backgrounds                                              | Enable for better edge quality and transparency handling                                         |
| Performance Optimization | Improve speed and memory                                                 | Increase `process_res` and `mask_blur` values for better results, while keeping memory usage in mind. |

## Models

ComfyUI-RMBG supports a variety of models for different use cases.  See the [original README](https://github.com/1038lab/ComfyUI-RMBG) for detailed information and links to download individual models.

## Requirements

*   ComfyUI
*   Python 3.10+
*   Required packages (installed automatically during setup).

## Credits

*   Developed by [AILab](https://github.com/1038lab).
*   See the [original README](https://github.com/1038lab/ComfyUI-RMBG) for detailed model credits.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=1038lab/comfyui-rmbg&type=Date)](https://star-history.com/#1038lab/comfyui-rmbg&Date)

Show your support and give the repository a ⭐!

## License

GPL-3.0 License