# ComfyUI-RMBG: Effortlessly Remove Backgrounds and Segment Images in ComfyUI

**ComfyUI-RMBG** is your all-in-one solution for advanced image background removal, object segmentation, and precise mask generation within ComfyUI. [Visit the original repository](https://github.com/1038lab/ComfyUI-RMBG) for the latest updates and features.

## Key Features

*   **Advanced Background Removal (RMBG Node):**
    *   Supports multiple models: RMBG-2.0, INSPYRENET, BEN, BEN2, BiRefNet, SDMatte.
    *   Offers various background options (transparent, solid colors).
    *   Provides batch processing for efficient workflow.
    *   Includes Fast Foreground Color Estimation and Invert Output options for better edge quality and transparency handling.
*   **Text-Prompted Object Segmentation (Segment Node):**
    *   Utilizes text prompts (tag-style or natural language) for flexible object detection.
    *   Integrates SAM (Segment Anything Model) and GroundingDINO for high-precision segmentation.
    *   Offers adjustable parameters for refined results.
*   **SAM2 Segmentation:**
    *   Leverages the latest Facebook Research SAM2 technology (Tiny/Small/Base+/Large).
    *   Features automatic model download with a manual placement option.
*   **Specialized Segmentation Nodes:**
    *   Face segmentation node for precise facial feature extraction.
    *   Clothes segmentation node with 18 different categories
    *   Fashion segmentation node with a new custom node for fashion segmentation.
*   **Additional Nodes for Enhanced Control:**
    *   Image Combiner, Image Stitch, Image/Mask Converter, Mask Enhancer, Mask Combiner, Mask Extractor, CropObject, ImageCompare, ColorInput, Kontext Refence latent Mask, MaskOverlay, ObjectRemover, ImageMaskResize
*   **Internationalization Support:**
    *   Includes internationalization (i18n) support for multiple languages.
    *   Improved user interface for dynamic language switching.

## Installation

Choose your preferred installation method:

1.  **ComfyUI Manager:** Search for `Comfyui-RMBG` and install directly through the manager.  Install required packages with the following command:

    ```bash
    ./ComfyUI/python_embeded/python -m pip install -r requirements.txt
    ```

2.  **Clone from GitHub:** Clone the repository into your ComfyUI `custom_nodes` folder:

    ```bash
    cd ComfyUI/custom_nodes
    git clone https://github.com/1038lab/ComfyUI-RMBG
    ```

    Then install the requirements:

    ```bash
    ./ComfyUI/python_embeded/python -m pip install -r requirements.txt
    ```

3.  **Comfy CLI:**
    ```bash
    comfy node install ComfyUI-RMBG
    ```
    Then install the requirements:
    ```bash
    ./ComfyUI/python_embeded/python -m pip install -r requirements.txt
    ```

## Model Downloads (Manual Download)

Models will be automatically downloaded to your `ComfyUI/models/RMBG/` or `ComfyUI/models/SAM` or `ComfyUI/models/sam2` or `ComfyUI/models/grounding-dino` folders upon first use.  If you encounter network restrictions, download the models manually from the following links and place the files in the appropriate model folders:

*   **RMBG-2.0:** [https://huggingface.co/1038lab/RMBG-2.0](https://huggingface.co/1038lab/RMBG-2.0)
*   **INSPYRENET:** [https://huggingface.co/1038lab/inspyrenet](https://huggingface.co/1038lab/inspyrenet)
*   **BEN:** [https://huggingface.co/1038lab/BEN](https://huggingface.co/1038lab/BEN)
*   **BEN2:** [https://huggingface.co/1038lab/BEN2](https://huggingface.co/1038lab/BEN2)
*   **BiRefNet:** [https://huggingface.co/1038lab/BiRefNet](https://huggingface.co/1038lab/BiRefNet)
*   **BiRefNet HR:** [https://huggingface.co/1038lab/BiRefNet_HR](https://huggingface.co/1038lab/BiRefNet_HR)
*   **SAM:** [https://huggingface.co/1038lab/sam](https://huggingface.co/1038lab/sam)
*   **SAM2:** [https://huggingface.co/1038lab/sam2](https://huggingface.co/1038lab/sam2)
*   **GroundingDINO:** [https://huggingface.co/1038lab/GroundingDINO](https://huggingface.co/1038lab/GroundingDINO)
*   **Clothes Segment:** [https://huggingface.co/1038lab/segformer_clothes](https://huggingface.co/1038lab/segformer_clothes)
*   **Fashion Segment:** [https://huggingface.co/1038lab/segformer_fashion](https://huggingface.co/1038lab/segformer_fashion)
*   **SDMatte:** [https://huggingface.co/1038lab/SDMatte](https://huggingface.co/1038lab/SDMatte)

## Usage

### RMBG Node

1.  Load the `RMBG (Remove Background)` node from the `🧪AILab/🧽RMBG` category.
2.  Connect an image to the input.
3.  Select a model from the dropdown.
4.  Adjust optional settings as needed (see below).
5.  Get two outputs:
    *   IMAGE: Processed image with a transparent or solid-color background.
    *   MASK: Binary mask of the foreground.

### Segment Node

1.  Load the `Segment (RMBG)` node from the `🧪AILab/🧽RMBG` category.
2.  Connect an image to the input.
3.  Enter a text prompt (tag-style or natural language).
4.  Select SAM and GroundingDINO models.
5.  Adjust parameters:
    *   Threshold: 0.25-0.35 (broad), 0.45-0.55 (precise)
    *   Mask blur and offset for refinement.
    *   Background color options.

## Optional Settings & Tips

*   **Sensitivity:** (RMBG Node) Adjusts mask detection strength (0.0-1.0). Higher values require stricter detection.
*   **Processing Resolution:** (RMBG Node) Controls detail and memory usage (256-2048, default 1024).
*   **Mask Blur:** (RMBG Node) Smooths mask edges (0-64, default 0).
*   **Mask Offset:** (RMBG Node) Expands/shrinks the mask boundary (-20 to 20, default 0).
*   **Background:** (RMBG Node) Choose output background color: Alpha (transparent), Black, White, Green, Blue, Red
*   **Invert Output:** (RMBG Node) Flip mask and image output.
*   **Refine Foreground:** (RMBG Node) Enable for better edge quality and transparency handling.
*   **Performance Optimization:** (RMBG Node) Increase `process_res` and `mask_blur` for better results (memory dependent).

## Troubleshooting

*   **401 error initializing GroundingDINO / missing `models/sam2`:** Delete cache files (`%USERPROFILE%\.cache\huggingface\token` and `%USERPROFILE%\.huggingface\token`) and re-run.
*   **Preview shows "Required input is missing: images":** Ensure image outputs are connected and upstream nodes ran successfully.

## Credits

*   **RMBG-2.0:** [https://huggingface.co/briaai/RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0)
*   **INSPYRENET:** [https://github.com/plemeri/InSPyReNet](https://github.com/plemeri/InSPyReNet)
*   **BEN:** [https://huggingface.co/PramaLLC/BEN](https://huggingface.co/PramaLLC/BEN)
*   **BEN2:** [https://huggingface.co/PramaLLC/BEN2](https://huggingface.co/PramaLLC/BEN2)
*   **BiRefNet:** [https://huggingface.co/ZhengPeng7](https://huggingface.co/ZhengPeng7)
*   **SAM:** [https://huggingface.co/facebook/sam-vit-base](https://huggingface.co/facebook/sam-vit-base)
*   **GroundingDINO:** [https://github.com/IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
*   **Clothes Segment:** [https://huggingface.co/mattmdjaga/segformer_b2_clothes](https://huggingface.co/mattmdjaga/segformer_b2_clothes)
*   **SDMatte:** [https://github.com/vivoCameraResearch/SDMatte](https://github.com/vivoCameraResearch/SDMatte)

Developed by: [AILab](https://github.com/1038lab)

## Star History

```html
<a href="https://www.star-history.com/#1038lab/comfyui-rmbg&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=1038lab/comfyui-rmbg&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=1038lab/comfyui-rmbg&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=1038lab/comfyui-rmbg&type=Date" />
 </picture>
</a>
```

## License

GPL-3.0 License