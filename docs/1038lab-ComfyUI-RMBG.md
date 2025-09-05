# ComfyUI-RMBG: Unleash Precise Image Background Removal and Segmentation

**Effortlessly remove backgrounds, segment objects, and refine images with the ComfyUI-RMBG custom node, powered by cutting-edge AI models.**  Access the original repo [here](https://github.com/1038lab/ComfyUI-RMBG).

## Key Features

*   **Advanced Background Removal:** Utilize a suite of models including RMBG-2.0, INSPYRENET, BEN, BEN2, and BiRefNet for high-quality background removal.
*   **Precise Object Segmentation:** Employ text prompts with SAM and GroundingDINO to segment objects and refine masks.
*   **SAM2 Segmentation:** Leverage the latest SAM2 models (Tiny/Small/Base+/Large) for accurate text-prompted segmentation.
*   **Clothes/Fashion Segmentation:** Dedicated nodes for intelligent clothes and fashion segmentation.
*   **Real-time Background Replacement:** Seamlessly integrate and replace backgrounds.
*   **Enhanced Edge Detection:** Improve accuracy with enhanced edge detection for more natural results.
*   **Image and Mask Tools:** Additional utilities to help refine and combine your images and masks.

## Latest Updates

*   **v2.9.0 (2025/08/18):** Added `SDMatte Matting` node.
*   **v2.8.0 (2025/08/11):** Added `SAM2Segment` node. Enhanced color widget support.
*   **v2.7.1 (2025/08/06):** Enhanced image loading and ImageStitch nodes.
*   **(See the original README for a comprehensive list of updates.)**

## Installation

Choose your preferred method:

### 1. ComfyUI Manager
Install directly from the ComfyUI Manager; search for `Comfyui-RMBG`.  Ensure to install the requirements:

```bash
./ComfyUI/python_embeded/python -m pip install -r requirements.txt
```

### 2. Clone the Repository

1.  Navigate to your ComfyUI custom_nodes folder:

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

### 3. Comfy CLI
Run the command:
```bash
comfy node install ComfyUI-RMBG
```
Install requirements:
```bash
./ComfyUI/python_embeded/python -m pip install -r requirements.txt
```

### 4. Manual Model Download (if needed)

*   Models are automatically downloaded upon first use.
*   If network issues occur, manually download models from the provided links in the original README and place them in the specified folders within the `ComfyUI/models/RMBG/` or `ComfyUI/models/SAM/` directories (or as defined in the documentation).

## Usage

### RMBG Node

1.  Load the `RMBG (Remove Background)` node from the `🧪AILab/🧽RMBG` category.
2.  Connect an image to the input.
3.  Select a model from the dropdown menu.
4.  Adjust optional parameters for optimal results.
5.  Get two outputs: the processed image (with transparency, black, white, green, blue, or red background) and the corresponding mask.

### Segment Node

1.  Load the `Segment (RMBG)` node from the `🧪AILab/🧽RMBG` category.
2.  Connect an image to the input.
3.  Enter a text prompt (tag-style or natural language).
4.  Select SAM or GroundingDINO models.
5.  Adjust parameters such as threshold, mask blur, and offset.

## Optional Settings and Tips

*   **Sensitivity:** Adjusts mask detection strength.
*   **Processing Resolution:** Controls detail and memory usage.
*   **Mask Blur:** Smooths mask edges.
*   **Mask Offset:** Expands/shrinks mask boundaries.
*   **Background:** Select output color.
*   **Invert Output:** Flip mask and image output.
*   **Refine Foreground:** Fast foreground color estimation.
*   **Performance Optimization:** Optimize settings for processing multiple images.

## Models

*   RMBG-2.0
*   INSPYRENET
*   BEN
*   BEN2
*   BiRefNet Models
*   SAM
*   SAM2
*   GroundingDINO

**(See original README "About Models" for comprehensive details on the capabilities of each model.)**

## Requirements

*   ComfyUI
*   Python 3.10+
*   Required packages (automatically installed).

## Troubleshooting

*   **401 error/missing `models/sam2`:** Delete `~/.cache/huggingface/token` (and `~/.huggingface/token`). Re-run.
*   **"Required input is missing" preview:** Ensure image outputs are connected and preceding nodes are executed successfully.

## Credits

*   See original README for model authors and additional credits.
*   Created by: [AILab](https://github.com/1038lab)

## Star History

\[Include the star history image from the original README]

**Please give a ⭐ on this repo if you find this custom node helpful!**

## License

GPL-3.0 License
```
Key improvements and SEO optimizations:

*   **Concise Hook:** A single, clear sentence immediately describes the primary function.
*   **Keyword Integration:**  Includes relevant keywords like "background removal," "segmentation," "ComfyUI," and model names throughout the text.
*   **Headings and Structure:** Well-defined headings and subheadings for readability and SEO.
*   **Bulleted Key Features:** Highlights the main benefits in a concise, scannable format.
*   **Clear Call to Action:** Encourages users to star the repository.
*   **Model Descriptions Summary:** Condensed the model descriptions while retaining their value.
*   **Installation Instructions:** Simplified and reorganized for easier understanding.
*   **Troubleshooting:** Included common issues and solutions.
*   **URL Anchors (Implied):** The structure (headings) creates internal links, improving SEO.
*   **Removed Redundancy:** Streamlined the text, removing unnecessary repetition.
*   **Cleaned Up Formatting:** Improved overall readability.