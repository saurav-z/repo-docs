# ComfyUI-RMBG: Effortlessly Remove Backgrounds and Segment Images in ComfyUI

**Enhance your image editing workflow with ComfyUI-RMBG, a powerful custom node for precise background removal, object segmentation, and advanced image manipulation.**  See the original repo here: [https://github.com/1038lab/ComfyUI-RMBG](https://github.com/1038lab/ComfyUI-RMBG)

## Key Features

*   **Background Removal:**
    *   Multiple models: RMBG-2.0, INSPYRENET, BEN, BEN2, BiRefNet.
    *   Flexible background options (transparent, color).
    *   Batch processing support.
    *   Fast Foreground Color Estimation for improved transparency.
*   **Object Segmentation:**
    *   Text-prompted object detection using SAM2 and GroundingDINO.
    *   Supports both tag-style and natural language prompts.
    *   Adjustable parameters for fine-tuning results.
*   **Advanced Segmentation Tools:**
    *   Face segmentation with 19 feature categories.
    *   Fashion and clothing segmentation.
*   **Enhanced Image Tools:**
    *   Image Combining, Stitching, Mask Enhancement, Mask Extraction
    *   Resize Images
    *   Convert image formats

## What's New

*   **v2.9.1:** Updates and fixes.
*   **v2.9.0:** Added `SDMatte Matting` node.
*   **v2.8.0:** Added `SAM2Segment` node. Enhanced color widget support.
*   **v2.7.1:** Enhanced LoadImage nodes. Redesigned ImageStitch node.
*   **v2.6.0:** Added `Kontext Refence latent Mask` node.
*   **v2.5.0:** Added `MaskOverlay`, `ObjectRemover`, `ImageMaskResize` new nodes. Added BiRefNet and batch image support.
*   **v2.4.0:** Added CropObject, ImageCompare, and ColorInput nodes
*   **(See full update history in the original README for versions prior to 2.4.0.)**

## Installation

### Method 1:  ComfyUI Manager (Recommended)

1.  Open ComfyUI Manager.
2.  Search for `ComfyUI-RMBG` and install.
3.  Install the requirements:

    ```bash
    ./ComfyUI/python_embeded/python -m pip install -r requirements.txt
    ```

### Method 2: Manual Clone

1.  Navigate to your ComfyUI `custom_nodes` directory:

    ```bash
    cd ComfyUI/custom_nodes
    ```

2.  Clone the repository:

    ```bash
    git clone https://github.com/1038lab/ComfyUI-RMBG
    ```

3.  Install the requirements:

    ```bash
    ./ComfyUI/python_embeded/python -m pip install -r requirements.txt
    ```

### Method 3: Comfy CLI

1.  Ensure you have `comfy-cli` installed: `pip install comfy-cli`
2.  Install ComfyUI if you haven't: `comfy install`
3.  Install the ComfyUI-RMBG custom node:

    ```bash
    comfy node install ComfyUI-RMBG
    ```

4.  Install the requirements:

    ```bash
    ./ComfyUI/python_embeded/python -m pip install -r requirements.txt
    ```

## Model Downloads

*   **Automatic Download:** Models are automatically downloaded to `ComfyUI/models/RMBG/` when first used.
*   **Manual Download:**
    *   Download the required models from the provided Hugging Face links (see original README for links)
    *   Place the model files in the corresponding folders within your `ComfyUI/models/RMBG/` or `ComfyUI/models/SAM/` or `ComfyUI/models/grounding-dino/` directory.

## Usage

### RMBG Node (Background Removal)

1.  Add the `RMBG (Remove Background)` node from the `🧪AILab/🧽RMBG` category.
2.  Connect an image to the input.
3.  Select a model.
4.  Adjust the optional parameters.
5.  Outputs: Processed image (with selected background) and mask.

### Optional Settings:

| Parameter              | Description                                                                      | Tip                                                                               |
| :--------------------- | :------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------- |
| **Sensitivity**       | Adjusts mask detection sensitivity. Higher value = stricter.                      | Default 0.5. Increase for complex images.                                          |
| **Processing Resolution**| Controls the processing resolution, detail, and memory usage.                   | 256-2048, Default 1024.  Higher resolution = better detail, more memory.          |
| **Mask Blur**          | Blurs mask edges.                                                                | Default 0. Try 1-5 for smoother edges.                                             |
| **Mask Offset**        | Expands/shrinks mask boundary.                                                    | Default 0.  Fine-tune between -10 and 10.                                          |
| **Background**        | Choose output background color                                                      | Alpha (transparent), Black, White, Green, Blue, Red                                    |
| **Invert Output**      | Flip mask and image output                                                      |                                                                            |
| **Refine Foreground** | Use Fast Foreground Color Estimation to optimize transparent background         | Enable for better edge quality and transparency handling                                   |
| **Performance Optimization**| Improve performance when processing multiple images |  Increase `process_res` and `mask_blur` if memory allows.                     |

### Segment Node (Object Segmentation)

1.  Add the `Segment (RMBG)` node from the `🧪AILab/🧽RMBG` category.
2.  Connect an image to the input.
3.  Enter a text prompt (tag-style or natural language).
4.  Select a model (SAM or GroundingDINO).
5.  Adjust parameters: Threshold, Mask blur, Offset, Background Color.

## About Models (Summarized)

*   **RMBG-2.0:** Developed by BRIA AI, BiRefNet architecture. High accuracy, edge detection, multiple objects, and batch processing.
*   **INSPYRENET:**  Human portrait segmentation. Fast processing.
*   **BEN/BEN2:**  Balanced speed and accuracy.
*   **BiRefNet Models:** Various BiRefNet models (general, portrait, matting, HR, lite) for different needs.
*   **SAM/SAM2:** Text-prompted segmentation. Includes various sizes.
*   **GroundingDINO:** Text-prompted object detection and segmentation.
*   **Clothes/Fashion Segment:** Models for clothing and fashion segmentation.
*   **SDMatte:** Models for matting.

## Requirements

*   ComfyUI
*   Python 3.10+
*   Automatically installed packages (see original README)

## Troubleshooting (Concise)

*   **401 error (GroundingDINO/SAM2 init):**  Delete `%USERPROFILE%\.cache\huggingface\token` and ensure no `HF_TOKEN`/`HUGGINGFACE_TOKEN` env vars.
*   **Missing "images" in preview:** Ensure image outputs are connected and upstream nodes ran successfully.

## Credits

*   See original README for model creators, and the [AILab](https://github.com/1038lab) at [https://github.com/1038lab](https://github.com/1038lab)

## Star History (See original README for chart)

## License

GPL-3.0 License