# ComfyUI-RMBG: Advanced Image Background Removal and Segmentation

**Unlock powerful image editing capabilities with ComfyUI-RMBG, a custom node that effortlessly removes backgrounds and precisely segments objects using cutting-edge AI models.** [(View on GitHub)](https://github.com/1038lab/ComfyUI-RMBG)

## Key Features

*   **Advanced Background Removal:**
    *   Utilizes models like RMBG-2.0, INSPYRENET, BEN, BEN2, and BiRefNet for accurate background removal.
    *   Offers various background color options (transparent, black, white, green, blue, red).
    *   Includes real-time background replacement for seamless integration.
    *   Batch processing support for efficient workflow.
*   **Precise Object Segmentation:**
    *   Text-prompted object detection using SAM, SAM2, and GroundingDINO.
    *   Supports both tag-style and natural language prompts for flexible object selection.
    *   Provides adjustable parameters for refining segmentation accuracy.
*   **Enhanced Edge Detection:**
    *   Improved edge quality and detail preservation.
    *   Optimized edge detection for sharper results.
*   **Comprehensive Model Support:**
    *   Includes a wide array of pre-trained models for diverse use cases.
    *   Automatic and manual model download options for easy setup.
*   **User-Friendly Interface:**
    *   Intuitive node design for seamless integration with ComfyUI workflows.
    *   Adjustable parameters (sensitivity, processing resolution, mask blur, etc.) for fine-tuning results.
*   **Recent Updates:**
    *   **v2.8.0 (2025/08/11):** Added `SAM2Segment` for text-prompted segmentation with the latest Facebook Research SAM2 technology and enhanced color widget support across all nodes
    *   **v2.7.1 (2025/08/06):** Enhanced LoadImage into three distinct nodes, redesigned ImageStitch node, and fixed background color handling issues
    *   Regular updates with new features and model support.

## Installation

Choose your preferred installation method:

*   **ComfyUI Manager:** Search for `Comfyui-RMBG` in the ComfyUI Manager and install. Remember to install the requirements.txt file.

    ```bash
    ./ComfyUI/python_embeded/python -m pip install -r requirements.txt
    ```

*   **Manual Installation:**
    ```bash
    cd ComfyUI/custom_nodes
    git clone https://github.com/1038lab/ComfyUI-RMBG
    ```
    Install the requirements.txt file.

    ```bash
    ./ComfyUI/python_embeded/python -m pip install -r requirements.txt
    ```

*   **Comfy CLI:**
    ```bash
    comfy node install ComfyUI-RMBG
    ```
    Install the requirements.txt file.

    ```bash
    ./ComfyUI/python_embeded/python -m pip install -r requirements.txt
    ```

## Usage

### 1. RMBG Node (Background Removal)

1.  Load the `RMBG (Remove Background)` node from the `🧪AILab/🧽RMBG` category.
2.  Connect an image to the input.
3.  Select a model from the dropdown menu.
4.  Adjust parameters as needed (optional).
5.  Outputs: Processed image and a binary mask.

### 2. Segment Node (Object Segmentation)

1.  Load the `Segment (RMBG)` node from the `🧪AILab/🧽RMBG` category.
2.  Connect an image to the input.
3.  Enter a text prompt (tag-style or natural language).
4.  Select models (SAM, GroundingDINO).
5.  Adjust parameters for precision.

## Model Details

*   Detailed information on the available models (RMBG-2.0, INSPYRENET, BEN, BEN2, BiRefNet, SAM, SAM2, and GroundingDINO) can be found in the original README (linked above).

## Requirements

*   ComfyUI
*   Python 3.10+
*   Dependencies: (Automatically installed during installation)
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

*   Links to the original model creators are included in the original README.

## Star History

[![Star History](https://api.star-history.com/svg?repos=1038lab/comfyui-rmbg&type=Date)](https://star-history.com/#1038lab/comfyui-rmbg&Date)

## License

GPL-3.0 License