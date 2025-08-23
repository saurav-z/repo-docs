# ComfyUI-RMBG: Advanced Image Background Removal and Segmentation

**Effortlessly remove backgrounds, segment objects, and refine images within ComfyUI using powerful AI models.**  Explore the [ComfyUI-RMBG GitHub repository](https://github.com/1038lab/ComfyUI-RMBG) for the latest updates and detailed information.

## Key Features

*   **Background Removal:**
    *   Utilizes cutting-edge models like RMBG-2.0, INSPYRENET, BEN, and BEN2 for precise background removal.
    *   Offers various background color options (Alpha, Black, White, Green, Blue, Red).
    *   Supports batch processing for efficient workflow.
*   **Advanced Segmentation:**
    *   **Text-Prompted Segmentation:** Segment objects with tag-style (e.g., "cat, dog") or natural language prompts (e.g., "a person wearing a red jacket").
    *   Uses SAM (Segment Anything Model) and GroundingDINO for high-precision results.
    *   Features multiple models SAM2 (Tiny/Small/Base+/Large)
    *   Refine segmentation with customizable parameters like threshold, mask blur, and offset.
    *   Clothes, Fashion, and Face segmentation models included.
*   **Enhanced Image Tools:**
    *   Real-time background replacement and improved edge detection.
    *   Image and Mask Tools including Image Combiner, Image Stitch, Mask Enhancer, and more.
*   **User-Friendly:**
    *   Automatic model downloads for many models, simplifying setup.
    *   Clear usage examples and parameter explanations.

## Recent Updates

Stay up-to-date with the latest enhancements:

*   **v2.9.0:** Added `SDMatte Matting` node.
*   **v2.8.0:** Added `SAM2Segment` node with text-prompted segmentation with the latest Facebook Research SAM2 technology. Enhanced color widget support across all nodes
*   **v2.7.1:** Enhanced LoadImage into three distinct nodes to meet different needs, all supporting direct image loading from local paths or URLs. Completely redesigned ImageStitch node compatible with ComfyUI's native functionality. Fixed background color handling issues reported by users.
*   **(See the original README for a full update history)**

## Installation

Choose your preferred installation method:

1.  **ComfyUI Manager:** Search and install `Comfyui-RMBG` directly within the ComfyUI Manager.
    *   Install dependencies:  `./ComfyUI/python_embeded/python -m pip install -r requirements.txt`
2.  **Clone Repository:**
    *   `cd ComfyUI/custom_nodes`
    *   `git clone https://github.com/1038lab/ComfyUI-RMBG`
    *   Install dependencies:  `./ComfyUI/python_embeded/python -m pip install -r requirements.txt`
3.  **Comfy CLI:**
    *   Ensure `pip install comfy-cli` is installed.
    *   `comfy node install ComfyUI-RMBG`
    *   Install dependencies:  `./ComfyUI/python_embeded/python -m pip install -r requirements.txt`

## Model Downloads

*   Models are automatically downloaded upon first use.
*   Manual download options are provided in the original README for specific models (RMBG-2.0, INSPYRENET, BEN, BEN2, BiRefNet, SAM, GroundingDINO, and others) and should be placed in their respective folders within `ComfyUI/models/RMBG/` or `ComfyUI/models/SAM`.

## Usage Guide

1.  **Access Nodes:** Load the `RMBG` and `Segment` nodes from the `🧪AILab/🧽RMBG` category in ComfyUI.
2.  **Connect and Configure:** Connect an image to the input and customize settings.
3.  **RMBG Node:**
    *   Select a background removal model.
    *   Adjust parameters like sensitivity, processing resolution, mask blur, and mask offset.
    *   Choose the desired background color.
    *   Get IMAGE output (processed image) and MASK output (foreground mask).
4.  **Segment Node:**
    *   Enter a text prompt for object detection.
    *   Select SAM or GroundingDINO.
    *   Adjust parameters like threshold, mask blur, and offset.
    *   Adjust background color options.

## Troubleshooting

*   **401 error with GroundingDINO/missing `models/sam2`:** Delete the Hugging Face token cache and rerun.
*   **Missing images in preview:** Ensure image outputs are connected and upstream nodes have run successfully.
*   **View the original README for more troubleshooting steps.**

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

[Include the Star History image here. The code from the original prompt should still work]

If you find this custom node helpful, please give it a ⭐ on GitHub!

## License

GPL-3.0 License