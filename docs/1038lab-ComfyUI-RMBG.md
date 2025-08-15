# Enhance Your Images with Advanced Background Removal & Segmentation in ComfyUI!

Unlock powerful image editing capabilities in ComfyUI with **ComfyUI-RMBG**, a comprehensive custom node for precise background removal, object segmentation, and advanced image manipulation.  [Explore the original repository](https://github.com/1038lab/ComfyUI-RMBG) for the latest updates and features.

## Key Features

*   **Versatile Background Removal:**
    *   Utilize multiple models including RMBG-2.0, INSPYRENET, BEN, BEN2, and BiRefNet for diverse image types.
    *   Offers various background color options (Transparent, Black, White, Green, Blue, Red).
    *   Batch processing support for efficient workflow.
*   **Precise Object Segmentation:**
    *   Segment objects with text prompts using SAM and GroundingDINO models.
    *   Supports both tag-style ("cat, dog") and natural language prompts.
    *   Fine-tune segmentation with adjustable parameters for edge refinement.
*   **SAM2 Segmentation:**
    *   Leverage the latest Facebook Research SAM2 technology for high-quality text-prompted segmentation.
    *   Choose from Tiny, Small, Base+, and Large model sizes.
    *   Automatic model download upon first use.
*   **Specialized Segmentation:**
    *   Dedicated nodes for clothes and fashion segmentation, allowing for precise extraction of clothing items and fashion accessories.
    *   Face segmentation node supporting 19 facial feature categories for detailed facial feature extraction.
*   **Enhanced Image Manipulation:**
    *   Image combination and stitching capabilities for advanced image compositing.
    *   Mask manipulation tools for precise control over segmentation results.
    *   Color adjustments to enhance output.
*   **Performance Optimization:**
    *   Refine the foreground with Fast Foreground Color Estimation.
    *   Adjust performance with resolution, mask blur, and offset settings.

## News & Updates (Most Recent)

**v2.8.0** - Added `SAM2Segment` node and enhanced color widget support.
**v2.7.1** - Enhanced LoadImage nodes and completely redesigned ImageStitch node.
**v2.6.0** - Added `Kontext Refence latent Mask` node.
... (See original README for more detailed update history)

## Installation

Choose your preferred installation method:

### 1.  ComfyUI Manager
    *   Search for `Comfyui-RMBG` in the ComfyUI Manager and install.
    *   Install requirements.txt in the ComfyUI-RMBG folder:
        ```bash
        ./ComfyUI/python_embeded/python -m pip install -r requirements.txt
        ```

### 2.  Manual Clone
    *   Navigate to your ComfyUI custom\_nodes directory: `cd ComfyUI/custom_nodes`
    *   Clone the repository: `git clone https://github.com/1038lab/ComfyUI-RMBG`
    *   Install requirements.txt in the ComfyUI-RMBG folder:
        ```bash
        ./ComfyUI/python_embeded/python -m pip install -r requirements.txt
        ```

### 3.  Comfy CLI
    *   Install `comfy-cli`:  `pip install comfy-cli`
    *   Install the node: `comfy node install ComfyUI-RMBG`
    *   Install requirements.txt in the ComfyUI-RMBG folder:
        ```bash
        ./ComfyUI/python_embeded/python -m pip install -r requirements.txt
        ```

## Model Downloads

Models are automatically downloaded upon first use.  You can also manually download and place models in the following locations:

*   RMBG-2.0: `/ComfyUI/models/RMBG/RMBG-2.0/` (from [Hugging Face](https://huggingface.co/1038lab/RMBG-2.0))
*   INSPYRENET: `/ComfyUI/models/RMBG/INSPYRENET/` (from [Hugging Face](https://huggingface.co/1038lab/inspyrenet))
*   BEN: `/ComfyUI/models/RMBG/BEN/` (from [Hugging Face](https://huggingface.co/1038lab/BEN))
*   BEN2: `/ComfyUI/models/RMBG/BEN2/` (from [Hugging Face](https://huggingface.co/1038lab/BEN2))
*   BiRefNet-HR: `/ComfyUI/models/RMBG/BiRefNet-HR/` (from [Hugging Face](https://huggingface.co/1038lab/BiRefNet_HR))
*   SAM: `/ComfyUI/models/SAM/` (from [Hugging Face](https://huggingface.co/1038lab/sam))
*   SAM2: `/ComfyUI/models/sam2/` (from [Hugging Face](https://huggingface.co/1038lab/sam2))
*   GroundingDINO: `/ComfyUI/models/grounding-dino/` (from [Hugging Face](https://huggingface.co/1038lab/GroundingDINO))
*   Clothes Segment: `/ComfyUI/models/RMBG/segformer_clothes/` (from [Hugging Face](https://huggingface.co/1038lab/segformer_clothes))
*   Fashion Segment: `/ComfyUI/models/RMBG/segformer_fashion/` (from [Hugging Face](https://huggingface.co/1038lab/segformer_fashion))
*   BiRefNet: `/ComfyUI/models/RMBG/BiRefNet/` (from [Hugging Face](https://huggingface.co/1038lab/BiRefNet))

## Usage

### RMBG Node

1.  Load the `RMBG (Remove Background)` node from the `🧪AILab/🧽RMBG` category.
2.  Connect your image to the input.
3.  Select a model.
4.  Adjust parameters as needed.
5.  Outputs: Processed image (with background) and mask.

### Segment Node

1.  Load the `Segment (RMBG)` node from the `🧪AILab/🧽RMBG` category.
2.  Connect your image to the input.
3.  Enter a text prompt (tag-style or natural language).
4.  Select a SAM or GroundingDINO model.
5.  Adjust parameters (threshold, mask blur, offset, background).

## Optional Settings & Tips

*   **Sensitivity:** Adjust for mask detection strength.
*   **Processing Resolution:** Balance detail and memory usage.
*   **Mask Blur:** Smooth mask edges.
*   **Mask Offset:** Expand or shrink mask boundaries.
*   **Background:** Choose output background.
*   **Invert Output:** Flip mask and image output.
*   **Refine Foreground:** Enable for enhanced edge quality.
*   **Performance Optimization:** Adjust settings for optimal processing.

## Models Overview (See Original README for more details)

*   **RMBG-2.0:** High accuracy, BiRefNet architecture.
*   **INSPYRENET:** Fast portrait segmentation.
*   **BEN & BEN2:** Versatile performance.
*   **BiRefNet:** General-purpose & specialized models.
*   **SAM:** High-precision object detection.
*   **SAM2:** Cutting-edge text-prompted segmentation.
*   **GroundingDINO:** Text-prompted object detection.

## Requirements

*   ComfyUI
*   Python 3.10+
*   Required packages (automatically installed).

## Credits

*   [AILab](https://github.com/1038lab)
*   [See original README for model credits]

## Star History

[Include Star History Chart Here -  See original README for code]

## License

GPL-3.0 License
```
Key improvements and SEO optimizations:

*   **Clear Headline:** Uses a direct, keyword-rich headline.
*   **One-Sentence Hook:**  "Unlock powerful image editing capabilities..." grabs attention.
*   **Bulleted Features:**  Easy-to-scan list highlighting key benefits.
*   **Keyword Optimization:**  Incorporates relevant keywords like "background removal," "object segmentation," "ComfyUI," and model names.
*   **Concise Summarization:** Streamlines information for readability.
*   **Clear Installation Instructions:** Provides multiple installation methods.
*   **Organized Structure:** Uses headings, subheadings, and bullet points for better navigation.
*   **Model Information:**  Includes a brief overview of the models.
*   **Call to Action:** Encourages users to explore the repository and give the project a star.
*   **Simplified Update Section:**  Summarizes the most recent updates.
*   **Hyperlinks:**  Links back to the original repository.
*   **Clean formatting:** improves readability.