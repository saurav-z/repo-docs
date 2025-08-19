# ComfyUI-RMBG: Advanced Image Background Removal & Segmentation

**Effortlessly remove backgrounds, segment objects, and refine images within ComfyUI using a suite of powerful models.**  [See the original repository](https://github.com/1038lab/ComfyUI-RMBG) for the latest updates.

## Key Features

*   **Versatile Background Removal:**
    *   Multiple models: RMBG-2.0, INSPYRENET, BEN, BEN2, BiRefNet, SDMatte models, SAM, SAM2 and GroundingDINO
    *   Support for various background options, including transparency.
    *   Batch processing for efficient workflow.
    *   Real-time background replacement.
*   **Precise Object Segmentation:**
    *   Text-prompted object detection for flexible segmentation.
    *   Support for both tag-style and natural language prompts.
    *   Leverages SAM2 for cutting-edge segmentation.
*   **Facial Feature & Fashion Element Segmentation:**
    *   Dedicated nodes for precise facial feature extraction.
    *   Nodes for fashion segmentation.
*   **Enhanced Image Tools:**
    *   Image Combination
    *   Image Mask Conversion
    *   Mask Enhancer
    *   Mask Combination
    *   Mask Extractor
*   **Flexible Control & Optimization:**
    *   Adjustable sensitivity, processing resolution, mask blur, and mask offset.
    *   Performance optimization options for efficiency.
*   **Internationalization (i18n) Support:**
    *   Fully translatable features for non-English speaking users.
*   **Real-time Background Replacement & Edge Detection:**
    *   Refine foreground with Fast Foreground Color Estimation for better transparency handling

## News & Updates

Stay up-to-date with the latest features and improvements:

*   **[v2.9.0]** Added `SDMatte Matting` node
*   **[v2.8.0]** Added `SAM2Segment` node and enhanced color widget support
*   **[v2.7.1]** Enhanced image loading and the redesigned ImageStitch node
*   **[v2.6.0]** Added `Kontext Refence latent Mask` node
*   **[v2.5.2]** Bug Fixes
*   **[v2.5.1]** Bug Fixes
*   **[v2.5.0]** Added new nodes
*   **[v2.4.0]** Added new nodes
*   **[v2.3.2]** Bug Fixes
*   **[v2.3.1]** Bug Fixes
*   **[v2.3.0]** Added new nodes
*   **[v2.2.1]** Bug Fixes
*   **[v2.2.0]** Added new nodes
*   **[v2.1.1]** Bug Fixes
*   **[v2.1.0]** Added i18n support
*   **[v2.0.0]** Added Image and Mask Tools
*   **[v1.9.3]** Clean up the code and fix the issue
*   **[v1.9.2]** Added new foreground refinement feature
*   **[v1.9.1]** Changed repository for model management
*   **[v1.9.0]** Enhanced BiRefNet model performance
*   **[v1.8.0]** Added a new custom node for BiRefNet-HR model
*   **[v1.7.0]** Added a new custom node for BEN2 model
*   **[v1.6.0]** Added a new custom node for face parsing and segmentation
*   **[v1.5.0]** Added a new custom node for fashion segmentation
*   **[v1.4.0]** Added intelligent clothes segmentation
*   **[v1.3.2]** Enhanced background handling
*   **[v1.3.1]** Bug fixes
*   **[v1.3.0]** Added text-prompted object segmentation
*   **[v1.2.2]** Bug fixes
*   **[v1.2.1]** Bug fixes
*   **[v1.2.0]** Bug fixes
*   **[v1.1.0]** Bug fixes

*(See `update.md` in the repository for detailed changelogs.)*

## Installation

### Method 1: Install via ComfyUI Manager
Search for `ComfyUI-RMBG` and install. Install requirements with:
```bash
./ComfyUI/python_embeded/python -m pip install -r requirements.txt
```

### Method 2: Clone the Repository
1.  Navigate to your ComfyUI `custom_nodes` directory:
    ```bash
    cd ComfyUI/custom_nodes
    ```
2.  Clone the repository:
    ```bash
    git clone https://github.com/1038lab/ComfyUI-RMBG
    ```
3.  Install requirements:
    ```bash
    ./ComfyUI/python_embeded/python -m pip install -r requirements.txt
    ```

### Method 3: Install via Comfy CLI
```bash
comfy node install ComfyUI-RMBG
```
Install requirements with:
```bash
./ComfyUI/python_embeded/python -m pip install -r requirements.txt
```

### Model Downloads

*   Models are auto-downloaded on first use to the correct directory.
*   Manual download links are provided in the original README for offline installation if auto-download fails.

## Usage

### RMBG Node (Background Removal)

1.  Load the `RMBG (Remove Background)` node from the `🧪AILab/🧽RMBG` category.
2.  Connect an image to the input.
3.  Select a model.
4.  Get two outputs: processed image and a mask.

### Segment Node (Object Segmentation)

1.  Load the `Segment (RMBG)` node from the `🧪AILab/🧽RMBG` category.
2.  Connect an image to the input.
3.  Enter text prompt (tag-style or natural language).
4.  Select SAM and GroundingDINO models.

## Optional Settings & Tips
| Setting              | Description                                                                       | Tips                                                                                                    |
|----------------------|-----------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------|
| **Sensitivity**      | Adjusts the strength of mask detection.                                         | Adjust based on image complexity; higher for complex images.                                       |
| **Processing Resolution** | Controls image resolution.                                                                    | Use 256 to 2048, with a default of 1024.  Higher resolutions yield better detail.                                         |
| **Mask Blur**        | Controls mask edge smoothness.                                                 | Set between 1 and 5 for smoother edges.                                                            |
| **Mask Offset**      | Expands or shrinks the mask boundary.                                          | Fine-tune between -10 and 10.                                                                       |
| **Background**      | Choose output background color (Alpha, Black, White, Green, Blue, Red)                                                                      | Choose the appropriate color for the desired effect.                            |
| **Invert Output**      | Flip mask and image output                                                   | Use to invert image and mask output                                                             |
| **Refine Foreground** | Use Fast Foreground Color Estimation to optimize transparent background                                                    | Enable for better edge quality and transparency handling                                                            |
| **Performance Optimization** | Properly setting options can enhance performance when processing multiple images.                                                        | Increase `process_res` and `mask_blur` values for better results, be mindful of memory usage.                             |

## Troubleshooting

*   **401 Error:** Delete huggingface token, ensure no environment variables, and re-run.
*   **"Required input is missing: images":** Ensure image outputs are connected and upstream nodes ran successfully.

## Credits

*   RMBG-2.0: [https://huggingface.co/briaai/RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0)
*   INSPYRENET: [https://github.com/plemeri/InSPyReNet](https://github.com/plemeri/InSPyReNet)
*   BEN: [https://huggingface.co/PramaLLC/BEN](https://huggingface.co/PramaLLC/BEN)
*   BEN2: [https://huggingface.co/PramaLLC/BEN2](https://huggingface.co/PramaLLC/BEN2)
*   BiRefNet: [https://huggingface.co/ZhengPeng7](https://huggingface.co/ZhengPeng7)
*   SAM: [https://huggingface.co/facebook/sam-vit-base](https://huggingface.co/facebook/sam-vit-base)
*   GroundingDINO: [https://github.com/IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
*   Clothes Segment: [https://huggingface.co/mattmdjaga/segformer_b2_clothes](https://huggingface.co/mattmdjaga/segformer_b2_clothes)
*   SDMatte: [https://github.com/vivoCameraResearch/SDMatte](https://github.com/vivoCameraResearch/SDMatte)

*   Created by: [AILab](https://github.com/1038lab)

## Star History

<!-- Place the Star History Chart here -->
<a href="https://www.star-history.com/#1038lab/comfyui-rmbg&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=1038lab/comfyui-rmbg&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=1038lab/comfyui-rmbg&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=1038lab/comfyui-rmbg&type=Date" />
 </picture>
</a>
<!-- End Star History Chart -->

## License

GPL-3.0 License