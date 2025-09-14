# ComfyUI-RMBG: Advanced Image Background Removal and Segmentation

Effortlessly remove backgrounds and segment objects with precision in ComfyUI using the [ComfyUI-RMBG](https://github.com/1038lab/ComfyUI-RMBG) custom node, powered by cutting-edge AI models.

## Key Features

*   **Robust Background Removal:** Utilizing models like RMBG-2.0, INSPYRENET, BEN, and BEN2, to remove the backgrounds from your images.
*   **Advanced Object Segmentation:** Precise object detection and segmentation using text prompts and SAM, SAM2, and GroundingDINO models, and also segment specific features like clothes or faces.
*   **Flexible Model Support:** Supports a diverse range of models including RMBG-2.0, INSPYRENET, BEN, BEN2, BiRefNet, SDMatte models, SAM, SAM2 and GroundingDINO.
*   **Real-Time Background Replacement:** Easily replace backgrounds using a variety of methods.
*   **Enhanced Edge Detection:** Improve accuracy with enhanced edge detection features.
*   **Batch Processing Support:** Allows users to process multiple images.
*   **Versatile Output Options:** Choose from transparent, black, white, green, blue, or red backgrounds.
*   **Face Segmentation**: Includes a new custom node for face parsing and segmentation with 19 facial feature categories (Skin, Nose, Eyes, Eyebrows, etc.)
*   **Clothes Segmentation**: Includes a new custom node for clothes segmentation, with 18 different categories, with the ability to select multiple items and combine segmentation.
*   **Fashion Segmentation**: Includes a new custom node for fashion segmentation.

## News & Updates

*   **[Latest Updates](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md)**

    *   **v2.9.1:**
    *   **v2.9.0:** Added `SDMatte Matting` node
    *   **v2.8.0:** Added `SAM2Segment` node for text-prompted segmentation.
    *   **v2.7.1:** Enhanced LoadImage nodes, redesigned ImageStitch node and fixed background color handling.
    *   **v2.6.0:** Added `Kontext Refence latent Mask` node.
    *   **v2.5.2:**
    *   **v2.5.1:**
    *   **v2.5.0:** Added `MaskOverlay`, `ObjectRemover`, `ImageMaskResize` new nodes and BiRefNet models and batch image support.
    *   **v2.4.0:** Added `CropObject`, `ImageCompare`, `ColorInput` nodes and new Segment V2.
    *   **v2.3.2:**
    *   **v2.3.1:**
    *   **v2.3.0:** Added IC-LoRA Concat, Image Crop nodes.
    *   **v2.2.1:**
    *   **v2.2.0:** Added new nodes: Image Combiner, Image Stitch, Image/Mask Converter, Mask Enhancer, Mask Combiner, and Mask Extractor
    *   **v2.1.1:** Enhanced compatibility with Transformers
    *   **v2.1.0:** Integrated internationalization (i18n) support for multiple languages.
    *   **v2.0.0:** Added Image and Mask Tools improved functionality.

## Installation

Choose your preferred installation method:

### 1. ComfyUI Manager

*   Use the ComfyUI Manager and search for and install `Comfyui-RMBG`.
*   Install the requirements using the embedded Python
    ```bash
    ./ComfyUI/python_embeded/python -m pip install -r requirements.txt
    ```

### 2. Manual Cloning

*   Navigate to your ComfyUI `custom_nodes` directory and clone the repository.
    ```bash
    cd ComfyUI/custom_nodes
    git clone https://github.com/1038lab/ComfyUI-RMBG
    ```
*   Install requirements using the embedded Python
    ```bash
    ./ComfyUI/python_embeded/python -m pip install -r requirements.txt
    ```

### 3. Comfy CLI

*   Ensure `pip install comfy-cli` is installed.
*   Install ComfyUI `comfy install` (if you don't have ComfyUI Installed)
*   Install the ComfyUI-RMBG, use the following command:
    ```bash
    comfy node install ComfyUI-RMBG
    ```
*   Install requirements using the embedded Python
    ```bash
    ./ComfyUI/python_embeded/python -m pip install -r requirements.txt
    ```

## Model Download

*   Models are automatically downloaded on first use to `ComfyUI/models/RMBG/`, `ComfyUI/models/SAM`, and `ComfyUI/models/grounding-dino`.
*   Manual download options are also available, as specified in the original README, for offline use or specific model versions.

## Usage

### RMBG Node

1.  Load the `RMBG (Remove Background)` node from the `🧪AILab/🧽RMBG` category.
2.  Connect an image to the input.
3.  Select a model from the dropdown menu.
4.  Set parameters as needed (optional).
5.  Outputs: Processed image and mask.

### Optional Settings :bulb: Tips

| Setting                  | Description                                                                | Tips                                                                                      |
| :----------------------- | :------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------- |
| **Sensitivity**          | Adjusts mask detection strength.                                        | Higher for complex images.                                                              |
| **Processing Resolution** | Controls image resolution.                                               | Balance detail and memory usage (256-2048).                                              |
| **Mask Blur**            | Blurs mask edges.                                                        | Values between 1 and 5 are recommended for smoother edges.                                 |
| **Mask Offset**          | Expands/shrinks mask boundary.                                             | Fine-tune between -10 and 10.                                                            |
| **Background**           | Choose output background color                                          | Alpha (transparent), Black, White, Green, Blue, Red.                                       |
| **Invert Output**        | Flips image and mask output.                                             | Invert both image and mask output                                                       |
| **Refine Foreground**    | Use Fast Foreground Color Estimation to optimize transparent background   | Enable for better edge quality and transparency handling                                  |
| **Performance Optimization** | Setting the options can enhance performance when processing multiple images. | If memory allows, consider increasing `process_res` and `mask_blur` values for better results, but be mindful of memory usage.                                                            |

### Basic Usage

1.  Load `RMBG (Remove Background)` node from the `🧪AILab/🧽RMBG` category
2.  Connect an image to the input
3.  Select a model from the dropdown menu
4.  select the parameters as needed (optional)
3.  Get two outputs:
    *   IMAGE: Processed image with transparent, black, white, green, blue, or red background
    *   MASK: Binary mask of the foreground

### Segment Node

1.  Load the `Segment (RMBG)` node from the `🧪AILab/🧽RMBG` category.
2.  Connect an image to the input.
3.  Enter a text prompt (tag-style or natural language).
4.  Select SAM and GroundingDINO models.
5.  Adjust parameters: Threshold (0.25-0.55), Mask blur, Mask offset, and background color.

## Troubleshooting

*   **401 Errors/Missing Models:** Delete cache token (`%USERPROFILE%\.cache\huggingface\token`), ensure no `HF_TOKEN`/`HUGGINGFACE_TOKEN` env vars, and re-run. Public repos download anonymously.
*   **Missing Images:** Ensure image outputs are connected and upstream nodes executed successfully.

## Credits

*   Detailed model credits and links are available in the original README.

## Star History

[Star History Chart](https://api.star-history.com/svg?repos=1038lab/comfyui-rmbg&type=Date)

## License

GPL-3.0 License