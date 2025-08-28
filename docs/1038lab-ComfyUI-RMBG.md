# ComfyUI-RMBG: Advanced Image Background Removal and Segmentation

Effortlessly remove backgrounds, segment objects, and refine details in your images with the powerful ComfyUI-RMBG custom node.  [See the original repository](https://github.com/1038lab/ComfyUI-RMBG).

## Key Features:

*   **Comprehensive Background Removal:**  Utilize advanced models like RMBG-2.0, INSPYRENET, BEN, BEN2, BiRefNet, and SDMatte for precise background removal.
*   **Text-Prompted Object Segmentation:**  Segment objects using text prompts with SAM and GroundingDINO models, supporting both tag-style and natural language inputs.
*   **SAM2 Integration:** Leverage the latest SAM2 models for high-quality, text-prompted segmentation with Tiny, Small, Base+, and Large options.
*   **Face, Clothes, and Fashion Segmentation:** Specialized nodes for detailed segmentation of facial features, clothing, and fashion elements.
*   **Real-time Background Replacement & Edge Refinement:**  Enhanced edge detection and real-time background replacement for improved accuracy.
*   **Flexible Image Handling:** Supports direct image loading from local paths or URLs, and batch processing for efficient workflows.
*   **Model Selection:** Download and utilize a variety of models from huggingface.co, including models for various tasks.

## Recent Updates

*   **v2.9.0 (2025/08/18):** Added `SDMatte Matting` node (See [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v290-20250818))
*   **v2.8.0 (2025/08/11):**  Added `SAM2Segment` node, enhanced color widget support (See [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v280-20250811))
*   **v2.7.1 (2025/08/06):** Enhanced LoadImage nodes, redesigned ImageStitch, and background color handling fixes (See [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v271-20250806))

*(See the original README for a complete update history.)*

## Installation:

Choose from the following installation methods:

### 1. ComfyUI Manager:

*   Search for and install `Comfyui-RMBG` within the ComfyUI Manager.
*   Install `requirements.txt` in the ComfyUI-RMBG folder after installing.

```bash
./ComfyUI/python_embeded/python -m pip install -r requirements.txt
```

### 2. Manual Clone:

*   Navigate to your ComfyUI custom_nodes directory.
*   Clone the repository: `git clone https://github.com/1038lab/ComfyUI-RMBG`
*   Install `requirements.txt` in the ComfyUI-RMBG folder after cloning.

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/1038lab/ComfyUI-RMBG
./ComfyUI/python_embeded/python -m pip install -r requirements.txt
```

### 3. Comfy CLI:

*   Ensure `pip install comfy-cli` is installed.
*   Install the node: `comfy node install ComfyUI-RMBG`
*   Install `requirements.txt` in the ComfyUI-RMBG folder after installing.

```bash
comfy node install ComfyUI-RMBG
./ComfyUI/python_embeded/python -m pip install -r requirements.txt
```

### 4. Model Download:

*   Models are automatically downloaded to `/ComfyUI/models/RMBG/` when first used.
*   Alternatively, manually download models from the provided Hugging Face links in the original README and place them in the appropriate folders.

## Usage:

### RMBG Node

1.  Load the `🧪AILab/🧽RMBG` node.
2.  Connect your image.
3.  Select a model.
4.  Adjust parameters (see below).
5.  Get two outputs:  Processed image and mask.

### Optional Settings:

*   **Sensitivity:** Adjust mask detection strength (0.0-1.0).
*   **Processing Resolution:** Control detail and memory usage (256-2048).
*   **Mask Blur:** Smooth mask edges (0-64).
*   **Mask Offset:** Expand/shrink mask boundary (-20 to 20).
*   **Background:** Choose output color (Alpha, Black, White, Green, Blue, Red).
*   **Invert Output:** Flip mask and image output.
*   **Refine Foreground:**  Enable for better transparency.
*   **Performance Optimization:**  Adjust resolution and blur for best results.

### Segment Node

1.  Load the `🧪AILab/🧽RMBG` node.
2.  Connect an image.
3.  Enter a text prompt.
4.  Select a SAM or GroundingDINO model.
5.  Adjust threshold, blur, offset, and background color.

**(See original README for detailed Parameter Descriptions)**

## Troubleshooting:

*(See original README for common issues and solutions)*

## Credits:

*(See original README for full credits and model sources)*

## Star History

<a href="https://www.star-history.com/#1038lab/comfyui-rmbg&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=1038lab/comfyui-rmbg&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=1038lab/comfyui-rmbg&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=1038lab/comfyui-rmbg&type=Date" />
 </picture>
</a>

Give the repo a star if you like it!

## License
GPL-3.0 License