# Enhance Your Images with ComfyUI-RMBG: AI-Powered Background Removal & Segmentation

**ComfyUI-RMBG** is a powerful ComfyUI custom node designed for advanced image background removal, object segmentation, and precise manipulation of visual elements. [Check out the original repository here](https://github.com/1038lab/ComfyUI-RMBG).

## Key Features

*   **Versatile Background Removal:** Remove backgrounds with precision using a variety of models including RMBG-2.0, INSPYRENET, BEN, and BEN2.
*   **Advanced Object Segmentation:** Segment objects using text prompts, supporting both tag-style and natural language inputs.
*   **Cutting-Edge SAM2 Integration:** Utilize the latest SAM2 models for highly accurate, text-prompted segmentation.
*   **Real-time Background Replacement:** Seamlessly replace backgrounds with custom colors or images.
*   **Enhanced Edge Detection:** Achieve improved accuracy with sophisticated edge detection.
*   **Face, Clothes, and Fashion Segmentation:** Specialized nodes for precise segmentation of faces, clothing items, and fashion elements.
*   **High Resolution Support:** Process images up to 2048x2048 resolution with select models.
*   **Flexible Masking Tools:** Utilize a variety of masking tools including MaskOverlay, ObjectRemover, ImageMaskResize, Mask Enhancer, Mask Combiner, and Mask Extractor for advanced image manipulation.
*   **Internationalization (i18n) Support:** Support for multiple languages, with dynamic language switching.

## Latest Updates

*   **v2.9.0** - Added `SDMatte Matting` node
*   **v2.8.0** - Added `SAM2Segment` node and Enhanced color widget support across all nodes
*   **v2.7.1** - Enhanced LoadImage and completely redesigned ImageStitch node
*   **v2.6.0** - Added `Kontext Refence latent Mask` node
*   **v2.5.2** - Bug fixes and enhancements
*   **v2.5.1** - Minor updates and fixes
*   **v2.5.0** - Added new nodes including `MaskOverlay`, `ObjectRemover`, `ImageMaskResize` and added BiRefNet and batch image support
*   **v2.4.0** - Added CropObject, ImageCompare, ColorInput nodes and new Segment V2
*   **v2.3.2** - Minor bug fixes
*   **v2.3.1** - Minor bug fixes
*   **v2.3.0** - Added new nodes: IC-LoRA Concat, Image Crop and resizing options for Load Image
*   **v2.2.1** - Bug fixes and enhancements
*   **v2.2.0** - Added new nodes: Image Combiner, Image Stitch, Image/Mask Converter, Mask Enhancer, Mask Combiner, and Mask Extractor. Fixed compatibility issues with transformers v4.49+
*   **v2.1.1** - Enhanced compatibility with Transformers
*   **v2.1.0** - Integrated internationalization (i18n) support for multiple languages
*   **v2.0.0** - Added Image and Mask Tools improved functionality. Enhanced code structure and documentation for better usability.
*   **v1.9.3** - Clean up the code and fix the issue
*   **v1.9.2** - with Fast Foreground Color Estimation
*   **v1.9.1** - Changed repository for model management to the new repository and Reorganized models files structure for better maintainability.
*   **v1.9.0** - with BiRefNet model improvements
*   **v1.8.0** - with new BiRefNet-HR model
*   **v1.7.0** - with new BEN2 model
*   **v1.6.0** - with new Face Segment custom node
*   **v1.5.0** - with new Fashion and accessories Segment custom node
*   **v1.4.0** - with new Clothes Segment node
*   **v1.3.2** - with background handling
*   **v1.3.1** - with bug fixes
*   **v1.3.0** - with new Segment node
*   **v1.2.2** - Bug fixes and enhancements
*   **v1.2.1** - Minor enhancements
*   **v1.2.0** - Improvements and bug fixes
*   **v1.1.0** - ComfyUI-RMBG update
  

## Installation

Choose your preferred installation method:

1.  **ComfyUI Manager:** Search for `ComfyUI-RMBG` in the ComfyUI Manager and install. Then install requirements.txt in the ComfyUI-RMBG folder.
2.  **Clone Repository:**
    ```bash
    cd ComfyUI/custom_nodes
    git clone https://github.com/1038lab/ComfyUI-RMBG
    ```
    Then install requirements.txt in the ComfyUI-RMBG folder.
3.  **Comfy CLI:**
    ```bash
    comfy node install ComfyUI-RMBG
    ```
    Then install requirements.txt in the ComfyUI-RMBG folder.

    ```bash
    ./ComfyUI/python_embeded/python -m pip install -r requirements.txt
    ```

## Model Downloads

*   Models are automatically downloaded upon first use.
*   Manual download options are provided for each model family (RMBG-2.0, INSPYRENET, BEN, BEN2, BiRefNet, SAM, SAM2, GroundingDINO, SDMatte, Clothes Segment, Fashion Segment) from Hugging Face.  Detailed download instructions and links are in the original README.

## Usage

### RMBG Node

1.  Load the `RMBG (Remove Background)` node from the `🧪AILab/🧽RMBG` category.
2.  Connect your input image.
3.  Select a model from the dropdown.
4.  Adjust parameters as needed (see Tips below).
5.  Get image and mask outputs.

### Optional Settings :bulb: Tips

| Optional Settings       | :memo: Description                                                           | :bulb: Tips                                                                                   |
| ----------------------- | ----------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| **Sensitivity**         | Adjusts mask detection strength. Higher values for stricter detection.       | Default: 0.5.  Adjust based on image complexity; more complex images may need higher sensitivity. |
| **Processing Resolution** | Controls image processing resolution, affecting detail and memory usage. | Choose between 256-2048, default is 1024.  Higher res = better detail, more memory. |
| **Mask Blur**           | Controls mask edge blurring, reducing jaggedness.                             | Default: 0.  Try 1-5 for smoother edges.                                                      |
| **Mask Offset**         | Expands/shrinks mask boundary. Positive expands, negative shrinks.              | Default: 0.  Fine-tune between -10 and 10 based on image.                                      |
| **Background**          | Choose output background color | Alpha (transparent background) Black, White, Green, Blue, Red |
| **Invert Output**      | Flip mask and image output | Invert both image and mask output |
| **Refine Foreground** | Use Fast Foreground Color Estimation to optimize transparent background | Enable for better edge quality and transparency handling |
| **Performance Optimization** | Properly setting options can enhance performance when processing multiple images. | If memory allows, consider increasing `process_res` and `mask_blur` values for better results, but be mindful of memory usage. |

### Segment Node

1.  Load `Segment (RMBG)` from `🧪AILab/🧽RMBG`.
2.  Connect an image.
3.  Enter a text prompt (tag-style or natural language).
4.  Select SAM or GroundingDINO models.
5.  Adjust parameters: Threshold, Mask blur, offset, and Background color.

## About Models

*   **RMBG-2.0**: BRIA AI, BiRefNet architecture, high accuracy, precise edge detection.
*   **INSPYRENET**: Human portrait segmentation, fast processing.
*   **BEN**: Balance of speed and accuracy for various scenes.
*   **BEN2**: Improved accuracy and speed.
*   **BIREFNET Models:** General purpose, portrait, matting, and HR models.
*   **SAM**: Powerful object detection and segmentation.
*   **SAM2**: Latest segmentation models (Tiny, Small, Base+, Large) for text-prompted segmentation.
*   **GroundingDINO**: Text-prompted object detection and segmentation.
*   **SDMatte:** Stable Diffusion Matte models for high-quality matting.
*   **Clothes Segment**: Clothes segmentation with 18 different categories.
*   **Fashion Segment**: Fashion segmentation.

## Requirements

*   ComfyUI
*   Python 3.10+
*   Required packages (automatically installed) are listed.

## Troubleshooting

*   See the original README for troubleshooting tips.

## Credits

*   See the original README for credits.

## Star History

[![Star History](https://api.star-history.com/svg?repos=1038lab/comfyui-rmbg&type=Date)](https://star-history.com/#1038lab/comfyui-rmbg&Date)

## License

GPL-3.0 License