# ComfyUI-RMBG: Effortlessly Remove Backgrounds and Segment Images in ComfyUI

**Enhance your image editing workflow with ComfyUI-RMBG, a powerful custom node for ComfyUI that provides advanced background removal, object segmentation, and more!** ✨ ([View the original repository](https://github.com/1038lab/ComfyUI-RMBG))

## Key Features

*   **Diverse Models:** Utilize a wide array of models for accurate background removal and segmentation, including RMBG-2.0, INSPYRENET, BEN, BEN2, BiRefNet, SDMatte, SAM, SAM2, and GroundingDINO.
*   **Real-time Background Replacement:** Easily replace backgrounds with transparent, black, white, green, blue or red options.
*   **Advanced Edge Detection:** Improve accuracy with enhanced edge detection and mask refinement.
*   **Text-Prompted Segmentation:** Employ text prompts to isolate specific objects using the Segment node.
*   **Face, Clothes, and Fashion Segmentation:** Specialized nodes for precise segmentation of facial features, clothing items, and fashion elements.
*   **SAM2 Integration:** Leverage the latest Facebook Research SAM2 technology for text-prompted segmentation.
*   **Batch Processing:** Efficiently process multiple images with batch processing support.
*   **Flexible Parameters:** Fine-tune results with adjustable sensitivity, resolution, mask blur, and mask offset settings.
*   **Easy Installation:** Install easily through ComfyUI Manager, Comfy CLI or by cloning the repository.

## Recent Updates

*   **v2.9.0:** Added `SDMatte Matting` node.
*   **v2.8.0:** Added `SAM2Segment` node for text-prompted segmentation with the latest Facebook Research SAM2 technology.
*   **v2.7.1:** Enhanced LoadImage nodes and fixed background color handling.
*   **v2.6.0:** Added `Kontext Refence latent Mask` node.
*   **v2.5.2:** Bug fixes.
*   **v2.5.1:** Bug fixes.
*   **v2.5.0:** Added `MaskOverlay`, `ObjectRemover`, `ImageMaskResize` new nodes and more.
*   **v2.4.0:** Added `CropObject`, `ImageCompare`, `ColorInput` nodes.
*   **v2.3.2:** Bug fixes.
*   **v2.3.1:** Bug fixes.
*   **v2.3.0:** Added new nodes: IC-LoRA Concat, Image Crop and more.
*   **v2.2.1:** Bug fixes.
*   **v2.2.0:** Added Image Combiner, Image Stitch and more.
*   **v2.1.1:** Enhanced compatibility with Transformers.
*   **v2.1.0:** Added internationalization (i18n) support for multiple languages.
*   **v2.0.0:** Added Image and Mask Tools improved functionality.

## Installation

Choose your preferred installation method:

1.  **ComfyUI Manager:** Search for "Comfyui-RMBG" in the ComfyUI Manager and install. Install the requirements.txt.
2.  **Clone Repository:**
    ```bash
    cd ComfyUI/custom_nodes
    git clone https://github.com/1038lab/ComfyUI-RMBG
    ```
    Install the requirements.txt.
3.  **Comfy CLI:**
    ```bash
    comfy node install ComfyUI-RMBG
    ```
    Install the requirements.txt.

    ```bash
    ./ComfyUI/python_embeded/python -m pip install -r requirements.txt
    ```

### Model Downloads

*   The models will be automatically downloaded to the appropriate folders (`ComfyUI/models/RMBG/`, `ComfyUI/models/SAM/`, etc.) when you first use the custom nodes.
*   If you prefer manual download, find the links in the original README under "Manually download the models:".

## Usage

### RMBG Node

1.  Load the `RMBG (Remove Background)` node from the `🧪AILab/🧽RMBG` category.
2.  Connect an image to the input.
3.  Select a model from the dropdown menu.
4.  Adjust parameters as needed (optional).
5.  Get two outputs:
    *   IMAGE: Processed image with transparent, black, white, green, blue, or red background.
    *   MASK: Binary mask of the foreground.

### Segment Node

1.  Load the `Segment (RMBG)` node from the `🧪AILab/🧽RMBG` category.
2.  Connect an image to the input.
3.  Enter a text prompt (tag-style or natural language).
4.  Select SAM or GroundingDINO models.
5.  Adjust parameters as needed (Threshold, Mask blur, Mask offset, and Background).

## Troubleshooting (short)

*   **401 error initializing GroundingDINO/missing models:** Delete the hugging face cache and try again.
*   **"Required input is missing" preview:** Ensure image outputs are connected.

## Credits

*   Created by: [AILab](https://github.com/1038lab)
*   (Credits to the model creators listed in original README - RMBG-2.0, INSPYRENET, BEN, BEN2, BiRefNet, SAM, GroundingDINO, SDMatte)

## Star History

<a href="https://www.star-history.com/#1038lab/comfyui-rmbg&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=1038lab/comfyui-rmbg&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=1038lab/comfyui-rmbg&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=1038lab/comfyui-rmbg&type=Date" />
 </picture>
</a>

Show your appreciation for this powerful tool – please give the repo a ⭐!

## License

GPL-3.0 License