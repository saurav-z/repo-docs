# Enhance Your Images with ComfyUI-RMBG: Advanced Background Removal and Segmentation

Tired of complex image editing? **ComfyUI-RMBG** is your all-in-one solution for effortless background removal, object segmentation, and precise image manipulation within ComfyUI. [**Get it now!**](https://github.com/1038lab/ComfyUI-RMBG)

**Key Features:**

*   **Advanced Background Removal:**
    *   Utilizes models like RMBG-2.0, INSPYRENET, BEN, and BiRefNet for superior background removal.
    *   Offers versatile background options, including transparent, black, white, green, blue, and red.
    *   Supports batch processing for efficient workflow.

*   **Precise Object Segmentation:**
    *   Text-prompted object detection using both tag-style and natural language prompts.
    *   Powered by SAM2, SAM, and GroundingDINO models for highly accurate segmentation.
    *   Flexible parameter controls for fine-tuning results.

*   **SAM2 Segmentation:**
    *   Leverages the latest SAM2 models (Tiny/Small/Base+/Large) for text-prompted segmentation.
    *   Automatic model download with a manual placement option, ensuring ease of use.

*   **Comprehensive Image Tools:**
    *   Includes nodes for image combining, stitching, and format conversion.
    *   Features mask enhancement, combination, and extraction tools for advanced control.

**Recent Updates:**
*   **v2.8.0** (2025/08/11): Added `SAM2Segment` node, enhanced color widget support.
*   **v2.7.1** (2025/08/06): Enhanced LoadImage nodes, completely redesigned ImageStitch node.
*   **v2.6.0** (2025/07/15): Added `Kontext Refence latent Mask` node
*   **v2.5.2-v2.0.0** (2025/07/11-2025/03/13): Added new nodes, enhanced code structure and documentation for better usability.
*   **(and many more - see original README for a complete list of updates)**

**Installation:**

Choose your preferred installation method:

*   **ComfyUI Manager:** Search for `Comfyui-RMBG` and install.
    ```bash
    ./ComfyUI/python_embeded/python -m pip install -r requirements.txt
    ```

*   **Manual Clone:**
    ```bash
    cd ComfyUI/custom_nodes
    git clone https://github.com/1038lab/ComfyUI-RMBG
    ./ComfyUI/python_embeded/python -m pip install -r requirements.txt
    ```

*   **Comfy CLI:**
    ```bash
    comfy node install ComfyUI-RMBG
    ./ComfyUI/python_embeded/python -m pip install -r requirements.txt
    ```

**Model Downloads:**
*   Models will be downloaded automatically upon first use.
*   For manual downloads, refer to the original README for direct download links and placement instructions.

**Usage:**

1.  **RMBG Node (Background Removal):** Load, connect an image, select a model, and choose your settings. Get a processed image with a transparent or solid background, plus a foreground mask.
2.  **Segment Node (Object Segmentation):** Connect an image, provide a text prompt, and select your segmentation model for accurate object detection.

**Explore the features in the original README. [View Original README](https://github.com/1038lab/ComfyUI-RMBG) for complete details, optional settings, model information, and usage examples.**