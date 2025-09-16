# ComfyUI-RMBG: Effortlessly Remove Backgrounds and Segment Images in ComfyUI

**Enhance your image editing workflow with ComfyUI-RMBG, a powerful custom node for ComfyUI designed for precise background removal, object segmentation, and advanced image manipulation. Find the original repo [here](https://github.com/1038lab/ComfyUI-RMBG).**

## Key Features

*   **Advanced Background Removal:** Utilizes models like RMBG-2.0, INSPYRENET, BEN, and BEN2 for accurate background removal.
*   **Versatile Object Segmentation:** Features the Segment node for text-prompted object detection using SAM and GroundingDINO, supporting both tag-style and natural language prompts.
*   **SAM2 Segmentation:** Implement the latest SAM2 models for high-quality text-prompted segmentation.
*   **Clothes and Fashion Segmentation:** Specialized nodes for precise segmentation of clothing and fashion elements.
*   **Real-time background replacement:** Enhanced edge detection for improved accuracy.
*   **Image and Mask Tools:** Offers a suite of tools like Image Combiner, Image Stitch, and Mask Enhancer for comprehensive image manipulation.
*   **Flexible Options:** Adjust parameters like sensitivity, processing resolution, mask blur, and offset for optimal results.
*   **Multiple Background Options:** Transparent, Black, White, Green, Blue, Red

## News & Updates

Stay up-to-date with the latest features and improvements:

*   **v2.9.1 (2025/09/12):** Update to v2.9.1
*   **v2.9.0 (2025/08/18):** Added SDMatte Matting node.
*   **v2.8.0 (2025/08/11):** Added `SAM2Segment` node and Enhanced color widget support.
*   **v2.7.1 (2025/08/06):** Enhanced LoadImage into three distinct nodes and Completely redesigned ImageStitch node compatible with ComfyUI's native functionality.
*   **v2.6.0 (2025/07/15):** Added `Kontext Refence latent Mask` node.
*   **v2.5.2 (2025/07/11):** Update ComfyUI-RMBG to **v2.5.2**
*   **v2.5.1 (2025/07/07):** Update ComfyUI-RMBG to **v2.5.1**
*   **v2.5.0 (2025/07/01):** Added `MaskOverlay`, `ObjectRemover`, `ImageMaskResize` new nodes and Added 2 BiRefNet models: `BiRefNet_lite-matting` and `BiRefNet_dynamic`
*   **v2.4.0 (2025/06/01):** Added `CropObject`, `ImageCompare`, `ColorInput` nodes and new Segment V2
*   **v2.3.2 (2025/05/15):** Update to v2.3.2
*   **v2.3.1 (2025/05/02):** Update to v2.3.1
*   **v2.3.0 (2025/05/01):** Added new nodes: IC-LoRA Concat, Image Crop
*   **v2.2.1 (2025/04/05):** Update to v2.2.1
*   **v2.2.0 (2025/04/05):** Added new nodes: Image Combiner, Image Stitch, Image/Mask Converter, Mask Enhancer, Mask Combiner, and Mask Extractor
*   **v2.1.1 (2025/03/21):** Enhanced compatibility with Transformers
*   **v2.1.0 (2025/03/19):** Integrated internationalization (i18n) support for multiple languages.
*   **v2.0.0 (2025/03/13):** Added Image and Mask Tools improved functionality and Introduced a new category path
*   **v1.9.3 (2025/02/24):** Clean up the code and fix the issue
*   **v1.9.2 (2025/02/21):** with Fast Foreground Color Estimation
*   **v1.9.1 (2025/02/20):** Changed repository for model management to the new repository and Reorganized models files structure for better maintainability.
*   **v1.9.0 (2025/02/19):** with BiRefNet model improvements
*   **v1.8.0 (2025/02/07):** with new BiRefNet-HR model
*   **v1.7.0 (2025/02/04):** with new BEN2 model
*   **v1.6.0 (2025/01/22):** with new Face Segment custom node
*   **v1.5.0 (2025/01/05):** with new Fashion and accessories Segment custom node
*   **v1.4.0 (2025/01/02):** with new Clothes Segment node
*   **v1.3.2 (2024/12/29):** with background handling
*   **v1.3.1 (2024/12/25):** with bug fixes
*   **v1.3.0 (2024/12/23):** with new Segment node
*   **v1.2.2 (2024/12/12):** Update Comfyui-RMBG ComfyUI Custom Node to **v1.2.2**
*   **v1.2.1 (2024/12/02):** Update Comfyui-RMBG ComfyUI Custom Node to **v1.2.1**
*   **v1.2.0 (2024/11/29):** Update Comfyui-RMBG ComfyUI Custom Node to **v1.2.0**
*   **v1.1.0 (2024/11/21):** Update Comfyui-RMBG ComfyUI Custom Node to **v1.1.0**

## Installation

Choose your preferred installation method:

### 1. Install via ComfyUI Manager

Search for `ComfyUI-RMBG` within the ComfyUI Manager and install.

```bash
./ComfyUI/python_embeded/python -m pip install -r requirements.txt
```

### 2. Manual Installation

Clone the repository into your ComfyUI custom\_nodes folder:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/1038lab/ComfyUI-RMBG
```

Install the required packages:

```bash
./ComfyUI/python_embeded/python -m pip install -r requirements.txt
```

### 3. Install via Comfy CLI

Install ComfyUI using `comfy install` (if you don't have ComfyUI Installed)
Install ComfyUI-RMBG:

```bash
comfy node install ComfyUI-RMBG
```

```bash
./ComfyUI/python_embeded/python -m pip install -r requirements.txt
```

### 4. Model Downloads

*   Models are automatically downloaded to `ComfyUI/models/RMBG/` and `ComfyUI/models/SAM/` on first use.
*   Manual downloads are available from the following links, if needed.
    *   RMBG-2.0: [https://huggingface.co/1038lab/RMBG-2.0](https://huggingface.co/1038lab/RMBG-2.0)
    *   INSPYRENET: [https://huggingface.co/1038lab/inspyrenet](https://huggingface.co/1038lab/inspyrenet)
    *   BEN: [https://huggingface.co/1038lab/BEN](https://huggingface.co/1038lab/BEN)
    *   BEN2: [https://huggingface.co/1038lab/BEN2](https://huggingface.co/1038lab/BEN2)
    *   BiRefNet-HR: [https://huggingface.co/1038lab/BiRefNet_HR](https://huggingface.co/1038lab/BiRefNet_HR)
    *   SAM: [https://huggingface.co/1038lab/sam](https://huggingface.co/1038lab/sam)
    *   SAM2: [https://huggingface.co/1038lab/sam2](https://huggingface.co/1038lab/sam2)
    *   GroundingDINO: [https://huggingface.co/1038lab/GroundingDINO](https://huggingface.co/1038lab/GroundingDINO)
    *   Clothes Segment: [https://huggingface.co/1038lab/segformer_clothes](https://huggingface.co/1038lab/segformer_clothes)
    *   Fashion Segment: [https://huggingface.co/1038lab/segformer_fashion](https://huggingface.co/1038lab/segformer_fashion)
    *   BiRefNet: [https://huggingface.co/1038lab/BiRefNet](https://huggingface.co/1038lab/BiRefNet)
    *   SDMatte: [https://huggingface.co/1038lab/SDMatte](https://huggingface.co/1038lab/SDMatte)

## Usage

### RMBG Node

1.  Load the `RMBG (Remove Background)` node from the `🧪AILab/🧽RMBG` category.
2.  Connect an image input.
3.  Select a model.
4.  (Optional) Adjust parameters: `sensitivity`, `process_res`, `mask_blur`, `mask_offset`, `background`, `invert_output`, and `optimize`.
5.  Get two outputs: Processed image and a foreground mask.

### Segment Node

1.  Load `Segment (RMBG)` from the `🧪AILab/🧽RMBG` category.
2.  Connect an image input.
3.  Enter a text prompt.
4.  Select models from SAM and GroundingDINO.
5.  Adjust `threshold`, `mask_blur`, `mask_offset`, and `background` as needed.

## Troubleshooting

*   **401 Error:** Delete `%USERPROFILE%\.cache\huggingface\token` and/or `%USERPROFILE%\.huggingface\token` and re-run.
*   **Missing images:** Ensure image outputs are connected and upstream nodes ran successfully.

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

Created by: [AILab](https://github.com/1038lab)

## Star History
```
<a href="https://www.star-history.com/#1038lab/comfyui-rmbg&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=1038lab/comfyui-rmbg&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=1038lab/comfyui-rmbg&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=1038lab/comfyui-rmbg&type=Date" />
 </picture>
</a>
```

## License

GPL-3.0 License