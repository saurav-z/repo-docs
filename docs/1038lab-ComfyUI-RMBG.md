# Supercharge Your ComfyUI Workflow with Advanced Image Editing: ComfyUI-RMBG

**Effortlessly remove backgrounds, segment objects, and perform advanced image editing directly within ComfyUI using a suite of cutting-edge models.** Learn more about the powerful capabilities of this custom node and how to implement it within your workflows by visiting the [original repository](https://github.com/1038lab/ComfyUI-RMBG).

## Key Features

*   **Background Removal:**
    *   Utilize multiple models: RMBG-2.0, INSPYRENET, BEN, BEN2.
    *   Versatile background options (Transparent, Black, White, Color).
    *   Batch processing for efficient workflows.
*   **Object Segmentation:**
    *   Text-prompted object detection.
    *   Supports both tag-style and natural language prompts.
    *   High-precision segmentation with SAM and GroundingDINO.
    *   Refine edges with Mask blur and offset parameters.
*   **Advanced Segmentation:**
    *   Facial feature segmentation with 19 categories.
    *   Clothes segmentation with 18 categories.
    *   Fashion segmentation.
*   **Image Manipulation Tools:**
    *   Image Combination, Stitching, and Conversion.
    *   Mask Enhancement and Extraction.
    *   Image Mask Resizing

## Recent Updates

*   **v2.7.1** (2025/08/06): Improved `LoadImage` nodes, redesigned `ImageStitch` node, and fixed background color handling.
*   **v2.7.0** (2025/07/27): Enhancements to `LoadImage` nodes to accommodate different needs, redesigned the ImageStitch node, and fixed background color handling issues.
*   **(See full list of updates in the [original README](https://github.com/1038lab/ComfyUI-RMBG))**

## Installation

Choose your preferred method:

1.  **ComfyUI Manager:** Install directly through the ComfyUI Manager by searching for `Comfyui-RMBG`.
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

Models are automatically downloaded upon first use, but you can also download them manually and place them in the corresponding folders under `ComfyUI/models/RMBG/`:

*   **RMBG-2.0:**  `ComfyUI/models/RMBG/RMBG-2.0` (from [Hugging Face](https://huggingface.co/1038lab/RMBG-2.0))
*   **INSPYRENET:**  `ComfyUI/models/RMBG/INSPYRENET` (from [Hugging Face](https://huggingface.co/1038lab/inspyrenet))
*   **BEN:** `ComfyUI/models/RMBG/BEN` (from [Hugging Face](https://huggingface.co/1038lab/BEN))
*   **BEN2:**  `ComfyUI/models/RMBG/BEN2` (from [Hugging Face](https://huggingface.co/1038lab/BEN2))
*   **BiRefNet-HR:**  `ComfyUI/models/RMBG/BiRefNet-HR` (from [Hugging Face](https://huggingface.co/1038lab/BiRefNet_HR))
*   **SAM:**  `ComfyUI/models/SAM` (from [Hugging Face](https://huggingface.co/1038lab/sam))
*   **GroundingDINO:** `ComfyUI/models/grounding-dino` (from [Hugging Face](https://huggingface.co/1038lab/GroundingDINO))
*   **Clothes Segment:**  `ComfyUI/models/RMBG/segformer_clothes` (from [Hugging Face](https://huggingface.co/1038lab/segformer_clothes))
*   **Fashion Segment:**  `ComfyUI/models/RMBG/segformer_fashion` (from [Hugging Face](https://huggingface.co/1038lab/segformer_fashion))
*   **BiRefNet:**  `ComfyUI/models/RMBG/BiRefNet` (from [Hugging Face](https://huggingface.co/1038lab/BiRefNet))

## Usage

1.  **RMBG Node (Background Removal):** Load the `🧪AILab/🧽RMBG` node, connect an image, select a model, and adjust parameters such as sensitivity, processing resolution, mask blur, offset, and background color.
2.  **Segment Node (Object Segmentation):** Load the `🧪AILab/🧽RMBG` node, connect an image, enter a text prompt, and select SAM or GroundingDINO. Adjust parameters like threshold, mask blur, offset, and background color.

### Basic Node Parameters

*   `sensitivity`: (0.0-1.0) - Controls the background removal sensitivity.
*   `process_res`: (512-2048, step 128) - Processing resolution.
*   `mask_blur`: (0-64) - Blur amount for the mask.
*   `mask_offset`: (-20 to 20) - Adjusts mask edges.
*   `background`:  Choose output background color.
*   `invert_output`: Flip mask and image output.
*   `optimize`: Toggle model optimization.

## [Detailed Model Information](https://github.com/1038lab/ComfyUI-RMBG#about-models)

Learn more about the architecture and purpose of the various model types used in the custom node.

## Requirements

*   ComfyUI
*   Python 3.10+
*   Required Packages (automatically installed): torch, torchvision, Pillow, numpy, huggingface-hub, tqdm, transformers, transparent-background, opencv-python

## Credits

*   Developed by: [AILab](https://github.com/1038lab)
*   Model Credits: RMBG-2.0 (Bria AI), INSPYRENET (plemeri), BEN & BEN2 (PramaLLC), BiRefNet (ZhengPeng7), SAM (facebook), GroundingDINO (IDEA-Research), Clothes Segment & Fashion Segment.

## Star History

<!-- Add your star history chart here -->
<a href="https://www.star-history.com/#1038lab/comfyui-rmbg&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=1038lab/comfyui-rmbg&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=1038lab/comfyui-rmbg&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=1038lab/comfyui-rmbg&type=Date" />
 </picture>
</a>
If you like my work, please give me a ⭐ on this repo! It's a great encouragement for my efforts!

## License

GPL-3.0 License