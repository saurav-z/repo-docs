# Enhance Your Images with Advanced Background Removal and Segmentation: ComfyUI-RMBG

**ComfyUI-RMBG is a powerful ComfyUI custom node that provides advanced image background removal and segmentation capabilities using a variety of cutting-edge models.**  [Visit the original repository](https://github.com/1038lab/ComfyUI-RMBG) for the latest updates and more information.

## Key Features

*   **Versatile Background Removal:** Utilize models like RMBG-2.0, INSPYRENET, BEN, BEN2, and BiRefNet for precise background removal.
*   **Advanced Segmentation:**  Segment objects, faces, clothing, and fashion elements with precision using SAM, SAM2, and GroundingDINO.
*   **Real-Time Background Replacement:** Easily replace backgrounds with custom colors or images.
*   **Improved Edge Detection:** Enhanced edge detection ensures accurate and refined results.
*   **Text-Prompted Segmentation:** Easily segment objects using text prompts, supporting both tag-style and natural language inputs.
*   **Multiple Model Support:** Choose from a wide selection of models, including SAM2 (Tiny/Small/Base+/Large) and GroundingDINO for different segmentation needs.
*   **Batch Processing:** Supports batch processing for efficient handling of multiple images.
*   **Flexible Image Loading:** Load images from local paths or URLs.

## What's New

*   **v2.9.0:** Added `SDMatte Matting` node
*   **v2.8.0:** Added `SAM2Segment` node and enhanced color widget support.
*   **v2.7.1:** Enhanced LoadImage nodes and redesigned ImageStitch node.
*   **v2.6.0:** Added `Kontext Refence latent Mask` node.
*   **v2.5.2/v2.5.1/v2.5.0:** Added various new nodes and BiRefNet models and batch image support.
*   **v2.4.0:** Added new nodes: `CropObject`, `ImageCompare`, `ColorInput` and new Segment V2
*   **(Previous updates):**  Significant enhancements, including new models, improved features, and bug fixes (see original README for full details).

## Installation

Choose your preferred method:

1.  **ComfyUI Manager:** Install directly through the ComfyUI Manager by searching for "Comfyui-RMBG."
2.  **Clone Repository:**
    ```bash
    cd ComfyUI/custom_nodes
    git clone https://github.com/1038lab/ComfyUI-RMBG
    cd ComfyUI-RMBG
    ./ComfyUI/python_embeded/python -m pip install -r requirements.txt
    ```
3.  **Comfy CLI:**
    ```bash
    comfy node install ComfyUI-RMBG
    cd ComfyUI-RMBG
    ./ComfyUI/python_embeded/python -m pip install -r requirements.txt
    ```

## Model Downloads

*   **Automatic Download:** Models will be downloaded automatically to the `ComfyUI/models/RMBG/`, `ComfyUI/models/SAM/`, and `ComfyUI/models/grounding-dino/` directories upon first use.
*   **Manual Download (if needed):** Instructions for manually downloading specific models are available in the original README.

## Usage

### 1.  RMBG Node (Background Removal)

*   Load the `RMBG (Remove Background)` node from the `🧪AILab/🧽RMBG` category.
*   Connect an image to the input.
*   Select a model.
*   Adjust optional settings.
*   Get the processed image and mask outputs.

### Optional Settings and Tips

| Setting                 | Description                                                                     | Tips                                                                                                          |
| :---------------------- | :------------------------------------------------------------------------------ | :------------------------------------------------------------------------------------------------------------ |
| Sensitivity             | Controls mask detection strength.                                             | Adjust based on image complexity (default: 0.5).                                                            |
| Processing Resolution   | Controls detail and memory usage.                                             | 256-2048 (default: 1024). Higher values = more detail, higher memory.                                        |
| Mask Blur               | Softens mask edges.                                                            | Try values between 1 and 5.                                                                                 |
| Mask Offset             | Expands/shrinks the mask boundary.                                              | Experiment between -10 and 10.                                                                                |
| Background              | Choose output background color. | Alpha (transparent background) Black, White, Green, Blue, Red |
| Invert Output           | Flip mask and image output | Invert both image and mask output |
| Refine Foreground          | Use Fast Foreground Color Estimation to optimize transparent background | Enable for better edge quality and transparency handling |
| Performance Optimization | Properly setting options can enhance performance when processing multiple images. | If memory allows, consider increasing `process_res` and `mask_blur` values for better results, but be mindful of memory usage. |


### 2. Segment Node (Object Segmentation)

*   Load the `Segment (RMBG)` node from the `🧪AILab/🧽RMBG` category.
*   Connect an image.
*   Enter a text prompt (tag-style or natural language).
*   Select SAM or GroundingDINO models.
*   Adjust parameters: Threshold (0.25-0.55), Mask blur, and Offset.

## Troubleshooting

*   Refer to the original README for troubleshooting common issues, such as 401 errors and missing input errors.

## Credits

*   Developed by [AILab](https://github.com/1038lab)

## Star History

<!-- Star History Chart -->
<a href="https://www.star-history.com/#1038lab/comfyui-rmbg&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=1038lab/comfyui-rmbg&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=1038lab/comfyui-rmbg&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=1038lab/comfyui-rmbg&type=Date" />
 </picture>
</a>
<!-- End Star History Chart -->

If you find this node useful, please consider giving the repository a star!