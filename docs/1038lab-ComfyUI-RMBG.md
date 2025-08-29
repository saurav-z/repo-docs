# ComfyUI-RMBG: Effortlessly Remove Backgrounds and Segment Images with Advanced AI

**Enhance your image editing workflow with ComfyUI-RMBG, a powerful custom node offering cutting-edge background removal, object segmentation, and AI-powered image manipulation.**  [Explore the original repo!](https://github.com/1038lab/ComfyUI-RMBG)

## Key Features

*   **Advanced Background Removal:**
    *   Utilizes models like RMBG-2.0, INSPYRENET, BEN, BEN2, BiRefNet, and SDMatte for precise background removal.
    *   Offers multiple background options: transparent, black, white, green, blue, and red.
    *   Supports batch processing for efficient workflow.
*   **Text-Prompted Object Segmentation:**
    *   Segment objects using text prompts (tag-style or natural language) with SAM and GroundingDINO models.
    *   Fine-tune results with adjustable threshold, mask blur, and offset.
*   **SAM2 Segmentation:**
    *   Leverages the latest SAM2 models (Tiny, Small, Base+, and Large) for high-quality, text-driven segmentation.
    *   Automatic model download and manual placement options for flexibility.
*   **Face, Clothes, and Fashion Segmentation:**
    *   Dedicated nodes for specialized segmentation of facial features, clothing items, and fashion accessories.
    *   Multiple category selection for combined segmentation.
*   **Enhanced Image Tools:**
    *   Includes nodes for image combining, stitching, and mask manipulation for comprehensive image editing.
*   **Improved Edge Detection & Performance Optimization:**
    *   Offers enhanced edge detection and detail preservation for sharper results.
    *   Provides options to optimize performance and memory usage.
    *   Includes a new feature for real-time background replacement and enhanced edge detection for improved accuracy.
*   **User-Friendly Interface:**
    *   Easy integration within ComfyUI, with intuitive controls.
    *   Clear guidance and troubleshooting steps to help you get started.

## Recent Updates

*   **v2.9.0 (2025/08/18):** Added `SDMatte Matting` node.
*   **v2.8.0 (2025/08/11):** Added `SAM2Segment` node for text-prompted segmentation with the latest Facebook Research SAM2 technology, and Enhanced color widget support across all nodes.
*   **v2.7.1 (2025/08/06):** Enhanced LoadImage into three distinct nodes and Completely redesigned ImageStitch node compatible with ComfyUI's native functionality.
*   **v2.6.0 (2025/07/15):** Added `Kontext Refence latent Mask` node.
*   **(See full update history in the original README)**

## Installation

**Choose your preferred installation method:**

1.  **ComfyUI-Manager:** Install directly from the ComfyUI Manager.
2.  **Clone from GitHub:** Clone the repository into your ComfyUI `custom_nodes` folder.
3.  **Comfy CLI:** Use `comfy node install ComfyUI-RMBG` if you have `comfy-cli` installed.

**Important:** After installation, install the required Python packages by running the following command within the `ComfyUI-RMBG` folder:

```bash
./ComfyUI/python_embeded/python -m pip install -r requirements.txt
```

## Model Download

*   Models are automatically downloaded upon first use.
*   If you have network restrictions or prefer manual downloads, access the models from the following repositories:
    *   RMBG-2.0, INSPYRENET, BEN, BEN2, BiRefNet, SAM, SAM2, GroundingDINO, Clothes Segment, Fashion Segment, SDMatte. (See [original README](https://github.com/1038lab/ComfyUI-RMBG) for specific links.)

## Usage

1.  **Load the Nodes:** Access the nodes from the `🧪AILab/🧽RMBG` category.
2.  **Connect Your Image:**  Connect an image to the input.
3.  **Select Options & Models:** Choose from the available models and configure the node's parameters.
4.  **Generate Output:**  Receive the processed image (with the selected background) and a mask.

### RMBG Node Parameters
  - `sensitivity`: Controls the background removal sensitivity (0.0-1.0)
  - `process_res`: Processing resolution (512-2048, step 128)
  - `mask_blur`: Blur amount for the mask (0-64)
  - `mask_offset`: Adjust mask edges (-20 to 20)
  - `background`: Choose output background color
  - `invert_output`: Flip mask and image output
  - `optimize`: Toggle model optimization

### Segment Node
*   Load `Segment (RMBG)` node from the `🧪AILab/🧽RMBG` category
*   Connect an image to the input
*   Enter text prompt (tag-style or natural language)
*   Select SAM and GroundingDINO models
*   Adjust parameters as needed:
   - Threshold: 0.25-0.35 for broad detection, 0.45-0.55 for precision
   - Mask blur and offset for edge refinement
   - Background color options

## Troubleshooting
  - Check that the files are installed properly, or that your network is not blocking the connection
  - Ensure image outputs are connected and upstream nodes ran successfully

## Credits
*   See original [README](https://github.com/1038lab/ComfyUI-RMBG) for model authors and contributors.
*   Created by: [AILab](https://github.com/1038lab)

## Star History
[Insert Star History Chart Here -  You can use the code provided in the original README to generate and display the chart]

**Give this repo a star if it helps you!**

## License
GPL-3.0 License