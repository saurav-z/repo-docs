# Enhance Your Images with Advanced Background Removal and Segmentation: ComfyUI-RMBG

Tired of clunky background removal? **ComfyUI-RMBG** offers a powerful suite of tools within ComfyUI for precise image editing. ([View on GitHub](https://github.com/1038lab/ComfyUI-RMBG))

## Key Features:

*   **Advanced Background Removal:** Utilizes models like RMBG-2.0, INSPYRENET, BEN, BEN2, and BiRefNet for superior background removal.
*   **Precise Object Segmentation:** Enables segmentation of objects, faces, clothing, and fashion elements using models such as SAM, SAM2, and GroundingDINO.
*   **Real-time Background Replacement:** Seamlessly replace backgrounds for instant visual transformations.
*   **Enhanced Edge Detection:** Improve accuracy with superior edge detection capabilities.
*   **Versatile Model Support:** Offers a wide range of models, including SDMatte, for various image processing needs.
*   **SAM2 Segmentation:** Uses latest SAM2 models to perform segmentation, with easy model downloads.
*   **Multiple Output Options:** Produces both processed images and masks for advanced control.

## Latest Updates:

*   **v2.9.0:** Added `SDMatte Matting` node. ([update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v290-20250818))
*   **v2.8.0:** Added `SAM2Segment` node for text-prompted segmentation, enhanced color widget support. ([update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v280-20250811))
*   **v2.7.1:** Improved image loading, redesigned ImageStitch node, background color fixes. ([update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v271-20250806))

(See full list of updates with screenshots in the original README.)

## Installation:

### Method 1: ComfyUI Manager

1.  Search for `Comfyui-RMBG` in the ComfyUI Manager.
2.  Install `requirements.txt` after installation.

### Method 2: Manual Clone

1.  Navigate to your ComfyUI `custom_nodes` directory: `cd ComfyUI/custom_nodes`
2.  Clone the repository: `git clone https://github.com/1038lab/ComfyUI-RMBG`
3.  Install dependencies within the ComfyUI environment: `./ComfyUI/python_embeded/python -m pip install -r requirements.txt`

### Method 3: Comfy CLI

1.  Install `comfy-cli`: `pip install comfy-cli`
2.  Install ComfyUI if you don't have it installed: `comfy install`
3.  Install ComfyUI-RMBG: `comfy node install ComfyUI-RMBG`
4.  Install dependencies: `./ComfyUI/python_embeded/python -m pip install -r requirements.txt`

### Model Downloads:

*   Models download automatically on first use. If not, manually download models from the provided links (see original README for details) and place them in the respective `ComfyUI/models/RMBG/` or `ComfyUI/models/SAM` directories.

## Usage:

### RMBG Node

1.  Load the `RMBG (Remove Background)` node from the `🧪AILab/🧽RMBG` category.
2.  Connect an image input.
3.  Select a model from the dropdown menu.
4.  Adjust parameters (optional, see below)
5.  Get two outputs: Processed Image and Mask.

### Segment Node

1.  Load the `Segment (RMBG)` node from the `🧪AILab/🧽RMBG` category.
2.  Connect an image input.
3.  Enter a text prompt (tag-style or natural language).
4.  Select SAM or GroundingDINO models.
5.  Adjust parameters as needed (threshold, mask blur, offset, background color).

### Optional Settings and Tips

**[See the original README for detailed optional settings and tips.]**

## About Models:

**[See the original README for details on the models (RMBG-2.0, INSPYRENET, BEN, BEN2, BiRefNet, SAM, SAM2, GroundingDINO).]**

## Requirements:

*   ComfyUI
*   Python 3.10+
*   Automatically installed packages include:
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
*   **SDMatte models:** Auto-download to `models/RMBG/SDMatte/`. Manual placement is also possible.  (See original README)

## Troubleshooting:

**[See the original README for troubleshooting tips.]**

## Credits:

*   **[Links to model sources and creators are listed in the original README.]**
*   Created by: [AILab](https://github.com/1038lab)

## Star History

**[See the original README for the Star History chart.]**

## License

GPL-3.0 License