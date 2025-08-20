# ComfyUI-RMBG: Effortlessly Remove Backgrounds and Segment Images with Advanced AI

**Enhance your ComfyUI workflows with ComfyUI-RMBG, a powerful custom node for precise image background removal, object segmentation, and real-time background replacement.**  [Visit the original repo](https://github.com/1038lab/ComfyUI-RMBG)

## Key Features

*   **Advanced Background Removal:** Utilizing models like RMBG-2.0, INSPYRENET, BEN, and BEN2, seamlessly remove backgrounds from your images.
*   **Precise Object Segmentation:** Employing SAM, SAM2, and GroundingDINO for text-prompted object detection and segmentation, including faces, clothing, and fashion items.
*   **Real-Time Background Replacement:**  Easily swap out backgrounds.
*   **Enhanced Edge Detection:**  Achieve improved accuracy with advanced edge detection features.
*   **Model Variety:** Supports a wide range of models, including BiRefNet, SDMatte models and more, offering flexibility for diverse needs.
*   **User-Friendly Interface:**  Features customizable parameters like sensitivity, processing resolution, mask blur, and offset for optimal results.
*   **Regular Updates:** Stay up-to-date with new features and models.

## Recent Updates

*   **v2.9.0 (2025/08/18):** Added `SDMatte Matting` node. (See [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v290-20250818))
*   **v2.8.0 (2025/08/11):** Added `SAM2Segment` node for text-prompted segmentation with the latest Facebook Research SAM2 technology and enhanced color widget support across all nodes. (See [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v280-20250811))
*   **v2.7.1 (2025/08/06):** Enhanced LoadImage nodes, redesigned ImageStitch node, and fixed background color handling issues. (See [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v271-20250806))
*   **v2.6.0 (2025/07/15):** Added `Kontext Refence latent Mask` node. (See [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v260-20250715))
*   **v2.5.2, v2.5.1, v2.5.0:** Added several new nodes. (See [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v252-20250711) & [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v251-20250707) & [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v250-20250701))
*   **(And many more updates – see below for a full history!)**

## Installation

### Method 1: Install via ComfyUI Manager
Search for `ComfyUI-RMBG` in the ComfyUI Manager and install.  Then, install the required packages:

```bash
./ComfyUI/python_embeded/python -m pip install -r requirements.txt
```

### Method 2: Clone to custom_nodes folder

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/1038lab/ComfyUI-RMBG
cd ComfyUI-RMBG
./ComfyUI/python_embeded/python -m pip install -r requirements.txt
```

### Method 3: Install via Comfy CLI
Ensure you have `pip install comfy-cli` installed.
```bash
comfy node install ComfyUI-RMBG
cd ComfyUI-RMBG
./ComfyUI/python_embeded/python -m pip install -r requirements.txt
```

## Model Management

*   **Automatic Download:** Models are automatically downloaded to `ComfyUI/models/RMBG/` upon first use.
*   **Manual Download:** If automatic download fails, manually download models from the links below and place them in the appropriate folders within `ComfyUI/models/RMBG/`.

### Model Download Links

*   **RMBG-2.0:** [https://huggingface.co/1038lab/RMBG-2.0](https://huggingface.co/1038lab/RMBG-2.0)
*   **INSPYRENET:** [https://huggingface.co/1038lab/inspyrenet](https://huggingface.co/1038lab/inspyrenet)
*   **BEN:** [https://huggingface.co/1038lab/BEN](https://huggingface.co/1038lab/BEN)
*   **BEN2:** [https://huggingface.co/1038lab/BEN2](https://huggingface.co/1038lab/BEN2)
*   **BiRefNet-HR:** [https://huggingface.co/1038lab/BiRefNet_HR](https://huggingface.co/1038lab/BiRefNet_HR)
*   **SAM:** [https://huggingface.co/1038lab/sam](https://huggingface.co/1038lab/sam)
*   **SAM2:** [https://huggingface.co/1038lab/sam2](https://huggingface.co/1038lab/sam2)
*   **GroundingDINO:** [https://huggingface.co/1038lab/GroundingDINO](https://huggingface.co/1038lab/GroundingDINO)
*   **Clothes Segment:** [https://huggingface.co/1038lab/segformer_clothes](https://huggingface.co/1038lab/segformer_clothes)
*   **Fashion Segment:** [https://huggingface.co/1038lab/segformer_fashion](https://huggingface.co/1038lab/segformer_fashion)
*   **BiRefNet Models:** [https://huggingface.co/1038lab/BiRefNet](https://huggingface.co/1038lab/BiRefNet)
*   **SDMatte models:** [https://huggingface.co/1038lab/SDMatte](https://huggingface.co/1038lab/SDMatte)

## Usage

### RMBG Node (Background Removal)

1.  Load the `RMBG (Remove Background)` node from the `🧪AILab/🧽RMBG` category.
2.  Connect an image to the input.
3.  Select a model from the dropdown menu.
4.  Adjust parameters as needed (optional).
5.  Get two outputs:  Processed image and a foreground mask.

### Segment Node (Object Segmentation)

1.  Load the `Segment (RMBG)` node from the `🧪AILab/🧽RMBG` category.
2.  Connect an image to the input.
3.  Enter a text prompt (tag-style or natural language).
4.  Select SAM or GroundingDINO models.
5.  Adjust parameters as needed.

## Optional Settings & Tips

| Setting               | Description                                                   | Tips                                                                                                |
| --------------------- | ------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| Sensitivity           | Adjusts mask detection strength.                              | Higher values for stricter detection.                                                              |
| Processing Resolution | Controls image processing resolution.                          |  Higher for more detail, but more memory.  Choose between 256 and 2048.                                       |
| Mask Blur             | Blurs mask edges.                                              | 1-5 for smoother edges.                                                                             |
| Mask Offset           | Expands/shrinks mask boundary.                                | Adjust between -10 and 10.                                                                           |
| Background            | Set output background color                                     | Choose Alpha (transparent background), Black, White, Green, Blue or Red.                                                                            |
| Invert Output         | Flip mask and image output                                         | Invert both image and mask output                                                                           |
| Refine Foreground     | Use Fast Foreground Color Estimation to optimize transparent background | Enable for better edge quality and transparency handling |
| Performance Optimization  | Properly setting options can enhance performance when processing multiple images. | If memory allows, consider increasing `process_res` and `mask_blur` values for better results, but be mindful of memory usage. |

## About the Models

*   **RMBG-2.0:** Developed by BRIA AI, uses BiRefNet architecture for high accuracy, precise edges, and multiple object support.
*   **INSPYRENET:** Optimized for human portrait segmentation.
*   **BEN:** Good balance of speed and accuracy, suitable for various scenes.
*   **BEN2:** Improved accuracy and speed, better handling of complex scenes, suitable for batch processing.
*   **BiRefNet Models:** Multiple BiRefNet variants for various needs.
*   **SAM:** A powerful model for object detection and segmentation.
*   **SAM2:** Latest segmentation model family designed for efficient, high-quality text-prompted segmentation.
*   **GroundingDINO:** Model for text-prompted object detection and segmentation.
*   **SDMatte:** Stable Diffusion Matte Model
    
## Requirements
*   ComfyUI
*   Python 3.10+
*   Required packages (automatically installed):
    *   huggingface-hub>=0.19.0
    *   transparent-background>=1.1.2
    *   segment-anything>=1.0
    *   groundingdino-py>=0.4.0
    *   opencv-python>=4.7.0
    *   onnxruntime>=1.15.0
    *   onnxruntime-gpu>=1.15.0
    *   protobuf>=3.20.2,<6.0.0
    *   hydra-core>=1.3.0
    *   omegaconf>=2.3.0
    *   iopath>=0.1.9

### SDMatte models (manual download)
*   Auto-download on first run to `models/RMBG/SDMatte/`
*   If network restricted, place weights manually:
    *   `models/RMBG/SDMatte/SDMatte.safetensors` (standard) or `SDMatte_plus.safetensors` (plus)
    *   Components (config files) are auto-downloaded; if needed, mirror the structure from the Hugging Face repo to `models/RMBG/SDMatte/` (`scheduler/`, `text_encoder/`, `tokenizer/`, `unet/`, `vae/`)

## Troubleshooting
*   **401 error with GroundingDINO / missing `models/sam2`:** Delete `%USERPROFILE%\.cache\huggingface\token` and `%USERPROFILE%\.huggingface\token`. Ensure no `HF_TOKEN`/`HUGGINGFACE_TOKEN` env vars are set.  Re-run; public repos download anonymously.
*   **"Required input is missing: images"**: Ensure image outputs are connected and upstream nodes ran successfully.

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

<a href="https://www.star-history.com/#1038lab/comfyui-rmbg&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=1038lab/comfyui-rmbg&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=1038lab/comfyui-rmbg&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=1038lab/comfyui-rmbg&type=Date" />
 </picture>
</a>

If this custom node helps you or you like my work, please give me ⭐ on this repo! It's a great encouragement for my efforts!

## License

GPL-3.0 License