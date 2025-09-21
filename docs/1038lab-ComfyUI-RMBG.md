# ComfyUI-RMBG: Effortlessly Remove Backgrounds and Segment Images with AI

Unleash the power of AI to precisely remove backgrounds, segment objects, and enhance images within ComfyUI using ComfyUI-RMBG.  [Visit the original repository](https://github.com/1038lab/ComfyUI-RMBG) for more details!

## Key Features

*   **Advanced Background Removal:** Utilize a variety of models like RMBG-2.0, INSPYRENET, BEN, and BEN2 for accurate background removal.
*   **Precise Object Segmentation:**  Employ text prompts for object detection and segmentation, with support for SAM, SAM2, and GroundingDINO models.
*   **Real-time Background Replacement:** Easily replace backgrounds with custom colors or images.
*   **Enhanced Edge Detection:** Achieve superior accuracy with improved edge refinement and transparency handling.
*   **Face, Fashion, and Clothes Segmentation:** Dedicated nodes for segmenting faces, fashion items, and clothing articles.
*   **Flexible Model Support:** Works with a wide range of models including  BiRefNet, SDMatte, and others.
*   **User-Friendly Interface:** Intuitive controls and options for customization.
*   **Batch Processing:** Efficiently process multiple images at once.
*   **Internationalization (i18n):** Supports multiple languages, improving accessibility.

## What's New

*   **v2.9.1** (2025/09/12): Update ComfyUI-RMBG to v2.9.1 ([update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v291-20250912) )
*   **v2.9.0** (2025/08/18): Added SDMatte Matting node ([update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v290-20250818))
*   **v2.8.0** (2025/08/11): Added SAM2Segment node and enhanced color widget support ([update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v280-20250811))
*   **v2.7.1** (2025/08/06): Enhanced LoadImage nodes, redesigned ImageStitch node, and fixed background color handling ([update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v271-20250806))
*   **v2.6.0** (2025/07/15): Added Kontext Refence latent Mask node ([update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v260-20250715))
*   **v2.5.2** (2025/07/11): (update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v252-20250711) )
*   **v2.5.1** (2025/07/07): (update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v251-20250707) )
*   **v2.5.0** (2025/07/01): Added MaskOverlay, ObjectRemover, ImageMaskResize nodes, added 2 BiRefNet models, and added batch image support for Segment nodes ([update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v250-20250701) )
*   **v2.4.0** (2025/06/01): Added CropObject, ImageCompare, ColorInput nodes and new Segment V2 nodes ([update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v240-20250601) )
*   **v2.3.2** (2025/05/15): ([update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v232-20250515) )
*   **v2.3.1** (2025/05/02): ([update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v231-20250502) )
*   **v2.3.0** (2025/05/01): Added IC-LoRA Concat, Image Crop nodes and resizing options for Load Image: Longest Side, Shortest Side, Width, and Height, enhancing flexibility. ([update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v230-20250501) )
*   **v2.2.1** (2025/04/05): ([update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v221-20250405) )
*   **v2.2.0** (2025/04/05): Added Image Combiner, Image Stitch, Image/Mask Converter, Mask Enhancer, Mask Combiner, and Mask Extractor. Fixed compatibility issues with transformers v4.49+ and i18n translation errors and added mask image output to segment nodes ([update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v220-20250405) )
*   **v2.1.1** (2025/03/21): Enhanced compatibility with Transformers. ([update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v211-20250321) )
*   **v2.1.0** (2025/03/19): Integrated internationalization (i18n) support for multiple languages ([update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v210-20250319) )
*   **v2.0.0** (2025/03/13): Added Image and Mask Tools improved functionality and introduced a new category path: `🧪AILab/🛠️UTIL/🖼️IMAGE`. ([update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v200-20250313) )
*   **v1.9.3** (2025/02/24): Clean up the code and fix the issue ([update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v193-20250224) )
*   **v1.9.2** (2025/02/21): Fast Foreground Color Estimation ([update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v192-20250221) )
*   **v1.9.1** (2025/02/20): Changed repository for model management ([update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v191-20250220) )
*   **v1.9.0** (2025/02/19): BiRefNet model improvements ([update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v190-20250219) )
*   **v1.8.0** (2025/02/07): New BiRefNet-HR model ([update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v180-20250207) )
*   **v1.7.0** (2025/02/04): New BEN2 model ([update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v170-20250204) )
*   **v1.6.0** (2025/01/22): New Face Segment custom node ([update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v160-20250122) )
*   **v1.5.0** (2025/01/05): New Fashion and accessories Segment custom node ([update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v150-20250105) )
*   **v1.4.0** (2025/01/02): New Clothes Segment node ([update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v140-20250102) )
*   **v1.3.2** (2024/12/29): Background handling ([update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v132-20241229) )
*   **v1.3.1** (2024/12/25): Bug fixes ([update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v131-20241225) )
*   **v1.3.0** (2024/12/23): New Segment node ([update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v140-20241222) )
*   **v1.2.2** (2024/12/12): Comfyui-RMBG ComfyUI Custom Node ([update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v122-20241212) )
*   **v1.2.1** (2024/12/02): Comfyui-RMBG ComfyUI Custom Node ([update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.mdv121-20241202) )
*   **v1.2.0** (2024/11/29): Comfyui-RMBG ComfyUI Custom Node ([update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v120-20241129) )
*   **v1.1.0** (2024/11/21): Comfyui-RMBG ComfyUI Custom Node ([update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v110-20241121) )

## Installation

Choose your preferred method:

### 1. ComfyUI Manager (Recommended)

*   Search for "Comfyui-RMBG" and install directly through the ComfyUI Manager.

*   After installing, install requirment.txt in the ComfyUI-RMBG folder:
    ```bash
    ./ComfyUI/python_embeded/python -m pip install -r requirements.txt
    ```

### 2. Manual Installation (Clone)

*   Navigate to your ComfyUI custom\_nodes directory: `cd ComfyUI/custom_nodes`
*   Clone the repository: `git clone https://github.com/1038lab/ComfyUI-RMBG`
*   After installing, install requirment.txt in the ComfyUI-RMBG folder:
    ```bash
    ./ComfyUI/python_embeded/python -m pip install -r requirements.txt
    ```

### 3. Comfy CLI

*   Ensure `pip install comfy-cli` is installed.
*   Use: `comfy node install ComfyUI-RMBG`
*   After installing, install requirment.txt in the ComfyUI-RMBG folder:
    ```bash
    ./ComfyUI/python_embeded/python -m pip install -r requirements.txt
    ```

### 4.  Model Download

*   Models will be automatically downloaded to `/ComfyUI/models/RMBG/` and other relevant directories upon first use.
*   Alternatively, download models manually from the provided links and place them in the corresponding folders:
    *   RMBG-2.0:  [`/ComfyUI/models/RMBG/RMBG-2.0`](https://huggingface.co/1038lab/RMBG-2.0)
    *   INSPYRENET: [`/ComfyUI/models/RMBG/INSPYRENET`](https://huggingface.co/1038lab/inspyrenet)
    *   BEN: [`/ComfyUI/models/RMBG/BEN`](https://huggingface.co/1038lab/BEN)
    *   BEN2: [`/ComfyUI/models/RMBG/BEN2`](https://huggingface.co/1038lab/BEN2)
    *   BiRefNet-HR: [`/ComfyUI/models/RMBG/BiRefNet-HR`](https://huggingface.co/1038lab/BiRefNet_HR)
    *   SAM: [`/ComfyUI/models/SAM`](https://huggingface.co/facebook/sam-vit-base)
    *   SAM2: [`/ComfyUI/models/sam2`](https://huggingface.co/1038lab/sam2)
    *   GroundingDINO: [`/ComfyUI/models/grounding-dino`](https://huggingface.co/1038lab/GroundingDINO)
    *   Clothes Segment: [`/ComfyUI/models/RMBG/segformer_clothes`](https://huggingface.co/mattmdjaga/segformer_b2_clothes)
    *   Fashion Segment: [`/ComfyUI/models/RMBG/segformer_fashion`](https://huggingface.co/1038lab/segformer_fashion)
    *   BiRefNet: [`/ComfyUI/models/RMBG/BiRefNet`](https://huggingface.co/1038lab/BiRefNet)
    *   SDMatte: [`/ComfyUI/models/RMBG/SDMatte`](https://huggingface.co/1038lab/SDMatte)

## Usage

### RMBG Node (Background Removal)

1.  Load the `RMBG (Remove Background)` node from the `🧪AILab/🧽RMBG` category.
2.  Connect your image to the input.
3.  Select a model.
4.  Adjust the optional settings (see below) as needed.
5.  Outputs: Processed image (with background) and a foreground mask.

### Optional Settings for RMBG Node

| Setting                 | Description                                                                   | Tips                                                                                                |
| ----------------------- | ----------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| **Sensitivity**         | Adjusts mask detection strength. Higher values = stricter detection.         | Default: 0.5. Adjust for image complexity.                                                         |
| **Processing Resolution** | Controls image processing resolution, affecting detail and memory usage.     | Choose a value between 256 and 2048 (default: 1024). Higher = more detail, more memory.             |
| **Mask Blur**           | Blurs mask edges.                                                              | Default: 0. Experiment with values between 1 and 5 for smoother edges.                              |
| **Mask Offset**         | Expands or shrinks the mask boundary.                                         | Default: 0. Fine-tune between -10 and 10.                                                           |
| **Background**          | Choose the output background color.                                            | Alpha (transparent), Black, White, Green, Blue, Red.                                                 |
| **Invert Output**       | Flip image and mask output.                                                  | Invert both image and mask output.                                                                  |
| **Refine Foreground**    | Use Fast Foreground Color Estimation.                                        | Enable for improved transparency and edge quality.                                                  |
| **Performance Optimization** |  Properly setting options can enhance performance when processing multiple images. | If memory allows, consider increasing `process_res` and `mask_blur` values for better results, but be mindful of memory usage.                                                                |

### Segment Node (Object Segmentation)

1.  Load the `Segment (RMBG)` node from the `🧪AILab/🧽RMBG` category.
2.  Connect an image.
3.  Enter a text prompt (tag-style or natural language).
4.  Select SAM or GroundingDINO models.
5.  Adjust parameters:
    *   Threshold: 0.25-0.35 (broad), 0.45-0.55 (precise).
    *   Mask blur and offset for edge refinement.
    *   Background color options.

## Model Information

<details>
<summary><h2>Detailed Model Descriptions</h2></summary>

### RMBG-2.0
Developed by BRIA AI, utilizing the BiRefNet architecture, RMBG-2.0 offers high accuracy in complex environments, precise edge detection, and excellent detail handling. It supports multiple objects per image and provides batch processing capabilities. The model is trained on a diverse dataset of 15,000 high-quality images to ensure robust performance across various scenarios.

### INSPYRENET
Specialized in human portrait segmentation, INSPYRENET offers fast processing speeds and good edge detection, making it ideal for portraits and human subjects.

### BEN
BEN provides a good balance between speed and accuracy, suitable for various image types and batch processing.

### BEN2
An advanced version of BEN, offering improved accuracy, speed, and enhanced handling of complex scenes, suitable for batch processing and a wider range of image types.

### BIREFNET MODELS
This family of models, based on the BiRefNet architecture, includes general-purpose, optimized, and specialized models for specific tasks:
*   BiRefNet-general purpose model (balanced performance)
*   BiRefNet_512x512 model (optimized for 512x512 resolution)
*   BiRefNet-portrait model (optimized for portrait/human matting)
*   BiRefNet-matting model (general purpose matting)
*   BiRefNet-HR model (high resolution up to 2560x2560)
*   BiRefNet-HR-matting model (high resolution matting)
*   BiRefNet_lite model (lightweight version for faster processing)
*   BiRefNet_lite-2K model (lightweight version for 2K resolution)

### SAM
Segment Anything Model (SAM) excels in object detection and segmentation, offering high accuracy, precise edge detection, and support for multiple objects in a single image.

### SAM2
SAM2 is the latest segmentation model family designed for efficient, high-quality text-prompted segmentation:
- Multiple sizes: Tiny, Small, Base+, Large
- Optimized inference with strong accuracy
- Automatic download on first use; manual placement supported in `ComfyUI/models/sam2`

### GroundingDINO
GroundingDINO provides accurate text-prompted object detection and segmentation with excellent detail handling.

### BiRefNet Models
*   BiRefNet-general purpose model (balanced performance)
*   BiRefNet_512x512 model (optimized for 512x512 resolution)
*   BiRefNet-portrait model (optimized for portrait/human matting)
*   BiRefNet-matting model (general purpose matting)
*   BiRefNet-HR model (high resolution up to 2560x2560)
*   BiRefNet-HR-matting model (high resolution matting)
*   BiRefNet_lite model (lightweight version for faster processing)
*   BiRefNet_lite-2K model (lightweight version for 2K resolution)
</details>

## Requirements

*   ComfyUI
*   Python 3.10+
*   Automatically installed packages:
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

*   **401 Error (GroundingDINO / missing sam2):**
    *   Delete `%USERPROFILE%\.cache\huggingface\token` and `%USERPROFILE%\.huggingface\token`.
    *   Ensure no `HF_TOKEN`/`HUGGINGFACE_TOKEN` environment variables.
    *   Re-run; public repos download anonymously.
*   **"Required input is missing: images"**:
    *   Ensure image outputs are connected and upstream nodes ran successfully.

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

[![Star History Chart](https://api.star-history.com/svg?repos=1038lab/comfyui-rmbg&type=Date)](https://www.star-history.com/#1038lab/comfyui-rmbg&Date)

If you find this custom node helpful, please give the repo a ⭐!

## License

GPL-3.0 License