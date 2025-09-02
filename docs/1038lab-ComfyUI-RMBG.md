# Enhance Your Images with Precision: ComfyUI-RMBG for Advanced Background Removal and Segmentation

**ComfyUI-RMBG** is your go-to custom node for ComfyUI, offering cutting-edge image background removal and object segmentation capabilities.  [Visit the original repository](https://github.com/1038lab/ComfyUI-RMBG) for more details and to contribute.

## Key Features

*   **Versatile Background Removal:** Choose from multiple models including RMBG-2.0, INSPYRENET, BEN, BEN2, BiRefNet, and SDMatte, providing flexibility for various image types.
*   **Advanced Object Segmentation:** Leverage text prompts with SAM and GroundingDINO to precisely segment objects using tag-style or natural language descriptions.
*   **SAM2 Segmentation:** Utilize the latest SAM2 models (Tiny/Small/Base+/Large) for high-quality, text-prompted segmentation.
*   **Real-Time Background Replacement:**  Easily replace backgrounds with a variety of color options, including transparency.
*   **Enhanced Edge Detection:** Refine your results with improved edge handling, ensuring clean and accurate segmentation.
*   **Face and Fashion Segmentation:**  Dedicated nodes for precise segmentation of facial features and clothing/fashion elements.
*   **Batch Processing Support:** Efficiently process multiple images at once.
*   **User-Friendly Interface:**  Easy-to-use nodes within ComfyUI for seamless integration into your workflows.
*   **Up-to-Date and Optimized:** Constantly updated with new models and features.

## Updates
*   **v2.9.0** (2025/08/18): Added `SDMatte Matting` node
*   **v2.8.0** (2025/08/11): Added `SAM2Segment` node and enhanced color widget support.
*   **v2.7.1** (2025/08/06): Enhanced image loading and ImageStitch node.
*   ... (See original README for previous updates.)

## Installation

**Choose one of the following installation methods:**

1.  **ComfyUI Manager:** Search for "Comfyui-RMBG" within the ComfyUI Manager and install.  Remember to install the required packages:
    ```bash
    ./ComfyUI/python_embeded/python -m pip install -r requirements.txt
    ```

2.  **Manual Clone:**
    ```bash
    cd ComfyUI/custom_nodes
    git clone https://github.com/1038lab/ComfyUI-RMBG
    ```
    Then, install the requirements:
    ```bash
    ./ComfyUI/python_embeded/python -m pip install -r requirements.txt
    ```

3.  **Comfy CLI:**
    ```bash
    comfy node install ComfyUI-RMBG
    ```
      Then, install the requirements:
    ```bash
    ./ComfyUI/python_embeded/python -m pip install -r requirements.txt
    ```

## Model Downloads

*   Models are automatically downloaded on first use. However, you can also manually download them and place them in the appropriate folders (see the original README for specific model download links).
    *   `ComfyUI/models/RMBG/RMBG-2.0`
    *   `ComfyUI/models/RMBG/INSPYRENET`
    *   `ComfyUI/models/RMBG/BEN`
    *   `ComfyUI/models/RMBG/BEN2`
    *   `ComfyUI/models/RMBG/BiRefNet-HR`
    *   `ComfyUI/models/SAM`
    *   `ComfyUI/models/sam2`
    *   `ComfyUI/models/grounding-dino`
    *   `ComfyUI/models/RMBG/segformer_clothes`
    *   `ComfyUI/models/RMBG/segformer_fashion`
    *   `ComfyUI/models/RMBG/BiRefNet`
    *   `ComfyUI/models/RMBG/SDMatte`

## Usage

### RMBG Node

1.  Load the `RMBG (Remove Background)` node from the `🧪AILab/🧽RMBG` category.
2.  Connect an image to the input.
3.  Select a model from the dropdown menu.
4.  Adjust parameters as needed (optional):

    *   **Sensitivity:** Adjust background removal sensitivity (0.0-1.0).
    *   **Processing Resolution:** Controls image detail and memory usage (256-2048).
    *   **Mask Blur:** Smooths mask edges (0-64).
    *   **Mask Offset:** Expands or shrinks mask boundaries (-20 to 20).
    *   **Background:** Choose output background color (Alpha, Black, White, Green, Blue, Red).
    *   **Invert Output:** Flips mask and image output.
    *   **Refine Foreground:** Use Fast Foreground Color Estimation to optimize transparent background.
    *   **Performance Optimization:** Properly setting options can enhance performance when processing multiple images.

5.  Get two outputs:
    *   IMAGE: Processed image with the selected background.
    *   MASK: Binary mask of the foreground.

### Segment Node

1.  Load `Segment (RMBG)` from `🧪AILab/🧽RMBG`.
2.  Connect an image.
3.  Enter a text prompt (tag-style or natural language).
4.  Select SAM or GroundingDINO models.
5.  Adjust parameters as needed (Threshold, Mask blur, Offset, Background color).

## Troubleshooting
*   401 error when initializing GroundingDINO / missing `models/sam2`:
    - Delete `%USERPROFILE%\.cache\huggingface\token` (and `%USERPROFILE%\.huggingface\token` if present)
    - Ensure no `HF_TOKEN`/`HUGGINGFACE_TOKEN` env vars are set
    - Re-run; public repos download anonymously (no login required)
*   Preview shows "Required input is missing: images":
    - Ensure image outputs are connected and upstream nodes ran successfully

## Credits
*   RMBG-2.0: https://huggingface.co/briaai/RMBG-2.0
*   ... (See original README for full credits)

## Star History
[![Star History Chart](https://api.star-history.com/svg?repos=1038lab/comfyui-rmbg&type=Date)](https://star-history.com/#1038lab/comfyui-rmbg&Date)

**Show your support!** If this custom node enhances your workflow, please consider giving the repository a ⭐.

## License
GPL-3.0 License