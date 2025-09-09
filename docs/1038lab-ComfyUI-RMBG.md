# Remove Backgrounds and Segment Images with Ease using ComfyUI-RMBG

Effortlessly remove backgrounds, segment objects, and refine images within ComfyUI using advanced models with [ComfyUI-RMBG](https://github.com/1038lab/ComfyUI-RMBG).

**Key Features:**

*   **Background Removal (RMBG Node):**
    *   Multiple models: RMBG-2.0, INSPYRENET, BEN, BEN2, SDMatte.
    *   Flexible background options (transparent, color).
    *   Batch processing support for efficient workflow.
    *   Fast Foreground Color Estimation to optimize transparent background
*   **Object Segmentation (Segment Node):**
    *   Text-prompted object detection using SAM2 and GroundingDINO.
    *   Support for tag-style and natural language prompts.
    *   Precise segmentation with adjustable parameters.
*   **Advanced Segmentation:**
    *   Face segmentation for facial feature extraction.
    *   Fashion and clothes segmentation with 18 different categories.
    *   Image/Mask Converter and Image/Mask Enhancer Nodes
*   **Real-time Background Replacement and Enhanced Edge Detection**

## Updates

*   **Recent Updates:** Stay informed with the latest features and improvements.
    *   **v2.9.0 (2025/08/18):** Added `SDMatte Matting` node
    *   **v2.8.0 (2025/08/11):** Added `SAM2Segment` node for text-prompted segmentation with the latest Facebook Research SAM2 technology.

    *(See full update history in the [original README](https://github.com/1038lab/ComfyUI-RMBG))*

## Installation

Choose your preferred method:

1.  **ComfyUI Manager:** Search and install "ComfyUI-RMBG" directly within ComfyUI.
2.  **Clone:**
    ```bash
    cd ComfyUI/custom_nodes
    git clone https://github.com/1038lab/ComfyUI-RMBG
    cd ComfyUI-RMBG
    ./ComfyUI/python_embeded/python -m pip install -r requirements.txt
    ```
3.  **Comfy CLI:** `comfy node install ComfyUI-RMBG`
    ```bash
    cd ComfyUI-RMBG
    ./ComfyUI/python_embeded/python -m pip install -r requirements.txt
    ```

    *(Ensure you install `requirements.txt` after installation)*

## Model Downloads

Models are automatically downloaded to `ComfyUI/models/RMBG/` or `ComfyUI/models/SAM/` and `ComfyUI/models/sam2/` and `ComfyUI/models/grounding-dino/` the first time a node uses them. If you have network restrictions, manually download the models from the provided links in the original README.

## Usage

1.  Load the `RMBG (Remove Background)` or `Segment (RMBG)` node from the `🧪AILab/🧽RMBG` category.
2.  Connect an image to the input.
3.  Select a model (RMBG Node) or enter a text prompt (Segment Node).
4.  Adjust parameters (sensitivity, resolution, blur, etc.) as needed.
5.  View the processed image and mask outputs.

### RMBG Node:

*   **Parameters**: sensitivity, process_res, mask_blur, mask_offset, background, invert_output, optimize
*   **Basic Usage**: Load the `RMBG (Remove Background)` node, connect an image, select a model and parameters.

### Segment Node:

*   **Basic Usage**: Load `Segment (RMBG)` node, connect an image, enter text prompt (tag-style or natural language), select SAM and GroundingDINO models.

## Troubleshooting

*   **Missing `models/sam2`:** Delete `%USERPROFILE%\.cache\huggingface\token` and `%USERPROFILE%\.huggingface\token` and rerun.
*   **"Required input is missing: images":**  Ensure image outputs are connected and upstream nodes ran successfully.

## Credits

*   [AILab](https://github.com/1038lab)

**(Refer to original README for a complete list of credits.)**

## Star History

<a href="https://www.star-history.com/#1038lab/comfyui-rmbg&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=1038lab/comfyui-rmbg&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=1038lab/comfyui-rmbg&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=1038lab/comfyui-rmbg&type=Date" />
 </picture>
</a>

## License

GPL-3.0 License