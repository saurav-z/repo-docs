# ComfyUI-RMBG: Effortlessly Remove Backgrounds and Segment Images with Advanced AI

**ComfyUI-RMBG is a powerful ComfyUI custom node that provides cutting-edge image background removal, object segmentation, and real-time background replacement capabilities, all within your ComfyUI workflow.**  Explore the [original repository](https://github.com/1038lab/ComfyUI-RMBG) for more details.

## Key Features

*   **Advanced Background Removal:**
    *   Utilizes models like RMBG-2.0, INSPYRENET, BEN, BEN2, and BiRefNet for precise background removal.
    *   Offers various background color options (transparent, black, white, etc.) and image/mask inverting.
    *   Batch processing support for efficient workflows.
*   **Precise Object Segmentation:**
    *   Text-prompted object detection using SAM and GroundingDINO models.
    *   Supports both tag-style and natural language prompts.
    *   Fine-tune with adjustable threshold, mask blur, and offset parameters.
*   **SAM2 Segmentation:**
    *   Leverages the latest SAM2 models (Tiny, Small, Base+, and Large) for advanced, text-prompted segmentation.
    *   Automatic model download and manual placement options are available for seamless integration.
*   **Facial, Fashion, and Clothes Segmentation:**
    *   Dedicated nodes for detailed segmentation of facial features, clothing items, and fashion elements.
    *   Offers multiple selections and combined segmentation options.
*   **Edge Detection Enhancement:**  Improved edge quality and fine detail preservation.
*   **User-Friendly Interface:** Color widget support and dynamic language switching.
*   **Real-Time Background Replacement:** Fast Foreground Color Estimation for optimized transparency handling.
*   **Wide Range of Models and Resolutions:** Supports high-resolution image processing (up to 2048x2048).

## Recent Updates

*   **v2.9.0 (2025/08/18):** Added `SDMatte Matting` node
*   **v2.8.0 (2025/08/11):** Added `SAM2Segment` node and enhanced color widget support.
*   **v2.7.1 (2025/08/06):** Enhanced LoadImage nodes and ImageStitch node, plus bug fixes.
*   **(More updates in the original README)**

## Installation

Choose your preferred installation method:

### Method 1: ComfyUI Manager
1.  Install via ComfyUI-Manager, search `Comfyui-RMBG` and install.
2.  Install dependencies: `./ComfyUI/python_embeded/python -m pip install -r requirements.txt`

### Method 2: Clone from GitHub
1.  Navigate to your ComfyUI custom_nodes directory: `cd ComfyUI/custom_nodes`
2.  Clone the repository: `git clone https://github.com/1038lab/ComfyUI-RMBG`
3.  Install dependencies: `./ComfyUI/python_embeded/python -m pip install -r requirements.txt`

### Method 3: Comfy CLI
1.  Install `comfy-cli`: `pip install comfy-cli`
2.  Install ComfyUI (if you don't have it): `comfy install`
3.  Install ComfyUI-RMBG: `comfy node install ComfyUI-RMBG`
4.  Install dependencies: `./ComfyUI/python_embeded/python -m pip install -r requirements.txt`

## Model Downloads

*   Models are automatically downloaded to `/ComfyUI/models/RMBG/` on first use.
*   **Manual Download Instructions:**
    *   Download models from the listed Hugging Face links (see original README) and place them in the corresponding folders within `/ComfyUI/models/RMBG/` or `/ComfyUI/models/SAM/` and `/ComfyUI/models/sam2/`.
    *   The Hugging Face links are provided in the original README.

## Usage

### RMBG Node (Background Removal)

1.  Load the `RMBG (Remove Background)` node from the `🧪AILab/🧽RMBG` category.
2.  Connect your input image.
3.  Select a model.
4.  Adjust optional parameters.
5.  Receive the processed image and the foreground mask as outputs.

### Segment Node (Object Segmentation)

1.  Load the `Segment (RMBG)` node from the `🧪AILab/🧽RMBG` category.
2.  Connect an image to the input.
3.  Enter a text prompt (tag-style or natural language).
4.  Select a model (SAM or GroundingDINO).
5.  Adjust parameters for desired results.

## Optional Settings & Tips

| Setting               | Description                                                        | Tips                                                                                                   |
| --------------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------ |
| Sensitivity           | Controls mask detection strength.                                | Adjust based on image complexity.  Higher values for complex images.                               |
| Processing Resolution | Controls processing resolution.                                    | Choose between 256 and 2048. Higher resolutions = better detail but more memory usage.               |
| Mask Blur             | Blurs mask edges.                                                  | 1-5 for smoother edges.                                                                                |
| Mask Offset           | Expands or shrinks mask boundary.                                 | Fine-tune between -10 and 10.                                                                          |
| Background            | Choose output background color                                     | Alpha (transparent background) Black, White, Green, Blue, Red |
| Invert Output         | Flip mask and image output                                     | Invert both image and mask output |
| Refine Foreground     | Use Fast Foreground Color Estimation to optimize transparent background | Enable for better edge quality and transparency handling |
| Performance Optimization | Optimize the performance of the node                                     | If memory allows, consider increasing `process_res` and `mask_blur` values for better results, but be mindful of memory usage. |

## Troubleshooting

*   **401 Error/Missing models:** Delete cache (`%USERPROFILE%\.cache\huggingface\token` and `%USERPROFILE%\.huggingface\token`), and rerun.
*   **"Required input is missing: images":**  Ensure image outputs are connected and upstream nodes ran successfully.

## Credits

*   See original README for full model credits.
*   Created by [AILab](https://github.com/1038lab)

## Star History

[Include Star History chart - see original README for code]

Give this repo a ⭐ if you find it helpful!

## License

GPL-3.0 License