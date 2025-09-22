# ComfyUI-RMBG: Advanced Image Background Removal and Segmentation

**Effortlessly remove backgrounds and segment images with precision using ComfyUI-RMBG, your all-in-one solution for image editing.**  [Explore the original repository](https://github.com/1038lab/ComfyUI-RMBG)

## Key Features

*   **Comprehensive Background Removal:**
    *   Utilizes a variety of models, including RMBG-2.0, INSPYRENET, BEN, BEN2, and BiRefNet.
    *   Offers diverse background options, including transparency (Alpha), black, white, green, blue, and red.
    *   Supports batch processing for efficient workflow.
*   **Advanced Segmentation Capabilities:**
    *   Includes text-prompted object segmentation.
    *   Supports both tag-style and natural language prompts.
    *   Integrates SAM2 models (Tiny/Small/Base+/Large) for cutting-edge segmentation.
    *   Uses GroundingDINO for precise object detection.
    *   Offers dedicated nodes for Clothes, Fashion, and Face segmentation.
*   **Real-time Background Replacement & Refinement:**
    *   Real-time background replacement for instant results.
    *   Enhanced edge detection for improved accuracy.
    *   Foreground color estimation for quality transparency.
*   **User-Friendly Interface & Flexibility:**
    *   Intuitive controls with adjustable parameters like sensitivity, mask blur, and offset.
    *   Flexible image resizing options.
    *   Optimized for performance, allowing you to process multiple images.
    *   Automatic model downloads.
*   **Continuous Updates & Enhanced Features:**
    *   Regular updates with new models, nodes, and improvements.

## News and Updates

*   **v2.9.1** (2025/09/12): Update ComfyUI-RMBG to v2.9.1 ([update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v291-20250912))
*   **v2.9.0** (2025/08/18): Update ComfyUI-RMBG to v2.9.0 ([update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v290-20250818))
    *   Added `SDMatte Matting` node
*   **v2.8.0** (2025/08/11): Update ComfyUI-RMBG to v2.8.0 ([update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v280-20250811))
    *   Added `SAM2Segment` node for text-prompted segmentation with the latest Facebook Research SAM2 technology.
    *   Enhanced color widget support across all nodes
*   **v2.7.1** (2025/08/06): Update ComfyUI-RMBG to v2.7.1 ([update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v271-20250806))
    *   Enhanced LoadImage into three distinct nodes to meet different needs, all supporting direct image loading from local paths or URLs
    *   Completely redesigned ImageStitch node compatible with ComfyUI's native functionality
    *   Fixed background color handling issues reported by users
*   **v2.6.0** (2025/07/15): Update ComfyUI-RMBG to v2.6.0 ([update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v260-20250715))
    *   Added `Kontext Refence latent Mask` node, Which uses a reference latent and mask for precise region conditioning.
*   **v2.5.2** (2025/07/11): Update ComfyUI-RMBG to v2.5.2 ([update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v252-20250711))
*   **v2.5.1** (2025/07/07): Update ComfyUI-RMBG to v2.5.1 ([update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v251-20250707))
*   **v2.5.0** (2025/07/01): Update ComfyUI-RMBG to v2.5.0 ([update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v250-20250701))
    *   Added `MaskOverlay`, `ObjectRemover`, `ImageMaskResize` new nodes.
    *   Added 2 BiRefNet models: `BiRefNet_lite-matting` and `BiRefNet_dynamic`
    *   Added batch image support for `Segment_v1` and `Segment_V2` nodes
*   **v2.4.0** (2025/06/01): Update ComfyUI-RMBG to v2.4.0 ([update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v240-20250601))
    *   Added `CropObject`, `ImageCompare`, `ColorInput` nodes and new Segment V2 (see update.md for details)
*   **v2.3.2** (2025/05/15): Update ComfyUI-RMBG to v2.3.2 ([update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v232-20250515))
*   **v2.3.1** (2025/05/02): Update ComfyUI-RMBG to v2.3.1 ([update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v231-20250502))
*   **v2.3.0** (2025/05/01): Update ComfyUI-RMBG to v2.3.0 ([update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v230-20250501))
    *   Added new nodes: IC-LoRA Concat, Image Crop
    *   Added resizing options for Load Image: Longest Side, Shortest Side, Width, and Height, enhancing flexibility.
*   **v2.2.1** (2025/04/05): Update ComfyUI-RMBG to v2.2.1 ([update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v221-20250405))
*   **v2.2.0** (2025/04/05): Update ComfyUI-RMBG to v2.2.0 ([update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v220-20250405))
    *   Added new nodes: Image Combiner, Image Stitch, Image/Mask Converter, Mask Enhancer, Mask Combiner, and Mask Extractor
    *   Fixed compatibility issues with transformers v4.49+
    *   Fixed i18n translation errors
    *   Added mask image output to segment nodes
*   **v2.1.1** (2025/03/21): Update ComfyUI-RMBG to v2.1.1 ([update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v211-20250321))
    *   Enhanced compatibility with Transformers
*   **v2.1.0** (2025/03/19): Update ComfyUI-RMBG to v2.1.0 ([update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v210-20250319))
    *   Integrated internationalization (i18n) support for multiple languages.
    *   Improved user interface for dynamic language switching.
    *   Enhanced accessibility for non-English speaking users with fully translatable features.
*   **v2.0.0** (2025/03/13): Update ComfyUI-RMBG to v2.0.0 ([update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v200-20250313))
    *   Added Image and Mask Tools improved functionality.
    *   Enhanced code structure and documentation for better usability.
    *   Introduced a new category path: `🧪AILab/🛠️UTIL/🖼️IMAGE`.
*   **v1.9.3** (2025/02/24): Clean up the code and fix the issue ([update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v193-20250224))
*   **v1.9.2** (2025/02/21): with Fast Foreground Color Estimation ([update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v192-20250221))
    *   Added new foreground refinement feature for better transparency handling
    *   Improved edge quality and detail preservation
    *   Enhanced memory optimization
*   **v1.9.1** (2025/02/20): Update ComfyUI-RMBG to v1.9.1 ([update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v191-20250220))
    *   Changed repository for model management to the new repository and Reorganized models files structure for better maintainability.
*   **v1.9.0** (2025/02/19): with BiRefNet model improvements ([update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v190-20250219))
    *   Enhanced BiRefNet model performance and stability
    *   Improved memory management for large images
*   **v1.8.0** (2025/02/07): with new BiRefNet-HR model ([update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v180-20250207))
    *   Added a new custom node for BiRefNet-HR model.
    *   Support high resolution image processing (up to 2048x2048)
*   **v1.7.0** (2025/02/04): with new BEN2 model ([update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v170-20250204))
    *   Added a new custom node for BEN2 model.
*   **v1.6.0** (2025/01/22): with new Face Segment custom node ([update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v160-20250122))
    *   Added a new custom node for face parsing and segmentation
    *   Support for 19 facial feature categories (Skin, Nose, Eyes, Eyebrows, etc.)
    *   Precise facial feature extraction and segmentation
    *   Multiple feature selection for combined segmentation
    *   Same parameter controls as other RMBG nodes
*   **v1.5.0** (2025/01/05): with new Fashion and accessories Segment custom node ([update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v150-20250105))
    *   Added a new custom node for fashion segmentation.
*   **v1.4.0** (2025/01/02): with new Clothes Segment node ([update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v140-20250102))
    *   Added intelligent clothes segmentation with 18 different categories
    *   Support multiple item selection and combined segmentation
    *   Same parameter controls as other RMBG nodes
*   **v1.3.2** (2024/12/29): with background handling ([update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v132-20241229))
    *   Enhanced background handling to support RGBA output when "Alpha" is selected.
    *   Ensured RGB output for all other background color selections.
*   **v1.3.1** (2024/12/25): with bug fixes ([update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v131-20241225))
    *   Fixed an issue with mask processing when the model returns a list of masks.
    *   Improved handling of image formats to prevent processing errors.
*   **v1.3.0** (2024/12/23): with new Segment node ([update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v140-20241222))
    *   Added text-prompted object segmentation
    *   Support both tag-style ("cat, dog") and natural language ("a person wearing red jacket") prompts
    *   Multiple models: SAM (vit_h/l/b) and GroundingDINO (SwinT/B) (as always model file will be downloaded automatically when first time using the specific model)
    *   This update requires install requirements.txt
*   **v1.2.2** (2024/12/12): Update Comfyui-RMBG ComfyUI Custom Node to v1.2.2 ([update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v122-20241212))
*   **v1.2.1** (2024/12/02): Update Comfyui-RMBG ComfyUI Custom Node to v1.2.1 ([update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.mdv121-20241202))
*   **v1.2.0** (2024/11/29): Update Comfyui-RMBG ComfyUI Custom Node to v1.2.0 ([update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v120-20241129))
*   **v1.1.0** (2024/11/21): Update Comfyui-RMBG ComfyUI Custom Node to v1.1.0 ([update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v110-20241121))

## Installation

### Method 1: Install via ComfyUI Manager

1.  In ComfyUI, open the ComfyUI Manager.
2.  Search for `ComfyUI-RMBG` and click install.
3.  After installation install requirements.txt in the ComfyUI-RMBG folder
    ```bash
    ./ComfyUI/python_embeded/python -m pip install -r requirements.txt
    ```
    >   **Note:** If your environment cannot install dependencies with the system Python, you can use ComfyUI's embedded Python instead.
    >   Example (embedded Python): `./ComfyUI/python_embeded/python.exe -m pip install --no-user --no-cache-dir -r requirements.txt`

### Method 2: Manual Installation

1.  Navigate to your ComfyUI custom\_nodes folder:
    ```bash
    cd ComfyUI/custom_nodes
    ```
2.  Clone the repository:
    ```bash
    git clone https://github.com/1038lab/ComfyUI-RMBG
    ```
3.  Install requirements.txt in the ComfyUI-RMBG folder
    ```bash
    ./ComfyUI/python_embeded/python -m pip install -r requirements.txt
    ```

### Method 3: Install via Comfy CLI

1.  Ensure `pip install comfy-cli` is installed.
2.  Install ComfyUI: `comfy install` (if you don't have ComfyUI Installed).
3.  Install ComfyUI-RMBG:
    ```bash
    comfy node install ComfyUI-RMBG
    ```
4.  Install requirements.txt in the ComfyUI-RMBG folder
    ```bash
    ./ComfyUI/python_embeded/python -m pip install -r requirements.txt
    ```

### Model Download

*   The required models will automatically download to `ComfyUI/models/RMBG/` upon first use.
*   For manual downloads and placement, refer to the links and folder structure detailed below:

    *   **RMBG-2.0:** [https://huggingface.co/1038lab/RMBG-2.0](https://huggingface.co/1038lab/RMBG-2.0) (`/ComfyUI/models/RMBG/RMBG-2.0`)
    *   **INSPYRENET:** [https://huggingface.co/1038lab/inspyrenet](https://huggingface.co/1038lab/inspyrenet) (`/ComfyUI/models/RMBG/INSPYRENET`)
    *   **BEN:** [https://huggingface.co/1038lab/BEN](https://huggingface.co/1038lab/BEN) (`/ComfyUI/models/RMBG/BEN`)
    *   **BEN2:** [https://huggingface.co/1038lab/BEN2](https://huggingface.co/1038lab/BEN2) (`/ComfyUI/models/RMBG/BEN2`)
    *   **BiRefNet-HR:** [https://huggingface.co/1038lab/BiRefNet_HR](https://huggingface.co/1038lab/BiRefNet_HR) (`/ComfyUI/models/RMBG/BiRefNet-HR`)
    *   **SAM:** [https://huggingface.co/1038lab/sam](https://huggingface.co/1038lab/sam) (`/ComfyUI/models/SAM`)
    *   **SAM2:** [https://huggingface.co/1038lab/sam2](https://huggingface.co/1038lab/sam2) (`/ComfyUI/models/sam2`)
    *   **GroundingDINO:** [https://huggingface.co/1038lab/GroundingDINO](https://huggingface.co/1038lab/GroundingDINO) (`/ComfyUI/models/grounding-dino`)
    *   **Clothes Segment:** [https://huggingface.co/1038lab/segformer_clothes](https://huggingface.co/1038lab/segformer_clothes) (`/ComfyUI/models/RMBG/segformer_clothes`)
    *   **Fashion Segment:** [https://huggingface.co/1038lab/segformer_fashion](https://huggingface.co/1038lab/segformer_fashion) (`/ComfyUI/models/RMBG/segformer_fashion`)
    *   **BiRefNet:** [https://huggingface.co/1038lab/BiRefNet](https://huggingface.co/1038lab/BiRefNet) (`/ComfyUI/models/RMBG/BiRefNet`)
    *   **SDMatte:** [https://huggingface.co/1038lab/SDMatte](https://huggingface.co/1038lab/SDMatte) (`/ComfyUI/models/RMBG/SDMatte`)

## Usage

### RMBG Node

![RMBG](https://github.com/user-attachments/assets/cd0eb92e-8f2e-4ae4-95f1-899a6d83cab6)

1.  Load the `RMBG (Remove Background)` node from the `🧪AILab/🧽RMBG` category.
2.  Connect your image to the input.
3.  Select a model from the dropdown menu.
4.  Adjust parameters as needed (optional).
5.  The node provides two outputs:
    *   **IMAGE:** Processed image with a transparent, black, white, green, blue, or red background.
    *   **MASK:** A binary mask of the foreground.

### Optional Settings and Tips

| Setting                   | Description                                               | Tips                                                                                                   |
| ------------------------- | --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| **Sensitivity**           | Adjusts mask detection strength. Higher = stricter.      | Default: 0.5.  Adjust based on image complexity; higher for complex images.                        |
| **Processing Resolution** | Controls resolution, affecting detail and memory usage.  | Choose 256-2048, default 1024. Higher resolutions give better detail but use more memory.         |
| **Mask Blur**             | Blurs mask edges.                                         | Default: 0.  Try values between 1 and 5 for smoother edges.                                           |
| **Mask Offset**           | Expands or shrinks the mask boundary.                    | Default: 0.  Fine-tune between -10 and 10 for specific images.                                      |
| **Background**            | Select output background color.                           | Alpha (transparent), Black, White, Green, Blue, Red.                                                  |
| **Invert Output**         | Flip image and mask output.                               | Invert both image and mask output.                                                                     |
| **Refine Foreground**     | Use Fast Foreground Color Estimation for transparency.     | Enable for better edge quality and transparency.                                                         |
| **Performance Optimization** | Proper setting to enhance performance when processing multiple images. | If memory allows, consider increasing `process_res` and `mask_blur` values for better results, but be mindful of memory usage. |

### Segment Node

1.  Load `Segment (RMBG)` node from the `🧪AILab/🧽RMBG` category.
2.  Connect your image to the input.
3.  Enter a text prompt (tag-style or natural language).
4.  Select models like SAM and GroundingDINO.
5.  Adjust parameters as needed:
    *   Threshold: 0.25-0.35 for broad detection, 0.45-0.55 for precision.
    *   Mask blur and offset for edge refinement.
    *   Background color options.

<details>
<summary><h2>About Models</h2></summary>

### RMBG-2.0

RMBG-2.0, developed by BRIA AI, uses the BiRefNet architecture, known for:

*   High accuracy in complex environments
*   Precise edge detection
*   Excellent handling of fine details
*   Support for multiple objects
*   Output Comparison
*   Output with background
*   Batch output for video

The model is trained on a diverse dataset of over 15,000 high-quality images.

### INSPYRENET

INSPYRENET specializes in human portrait segmentation, providing:

*   Fast processing
*   Good edge detection
*   Ideal for portraits and human subjects

### BEN

BEN provides a balance of speed and accuracy on various image types.

### BEN2

BEN2, an advanced version of BEN, offers:

*   Improved accuracy and speed
*   Better handling of complex scenes
*   Support for more image types

### BIREFNET Models

BIREFNET models offer a range of options for image segmentation:

*   BiRefNet-general purpose model
*   BiRefNet_512x512 model (optimized for 512x512 resolution)
*   BiRefNet-portrait model
*   BiRefNet-matting model
*   BiRefNet-HR model (high resolution up to 2560x2560)
*   BiRefNet-HR-matting model
*   BiRefNet_lite model
*   BiRefNet_lite-2K model

### SAM

SAM is a powerful model for object detection and segmentation, offering:

*   High accuracy in complex environments
*   Precise edge detection and preservation
*   Excellent handling of fine details
*   Support for multiple objects in a single image
*   Output Comparison
*   Output with background
*   Batch output for video

### SAM2

SAM2 is the latest segmentation model family:

*   Multiple sizes: Tiny, Small, Base+, Large
*   Optimized inference
*   Automatic download on first use

### GroundingDINO

GroundingDINO is a model for text-prompted object detection and segmentation, offering:

*   High accuracy in complex environments
*   Precise edge detection and preservation
*   Excellent handling of fine details
*   Support for multiple objects in a single image
*   Output Comparison
*   Output with background
*   Batch output for video

### BiRefNet Models
* BiRefNet-general purpose model (balanced performance)
* BiRefNet_512x512 model (optimized for 512x512 resolution)
* BiRefNet-portrait model (optimized for portrait/human matting)
* BiRefNet-matting model (general purpose matting)
* BiRefNet-HR model (high resolution up to 2560x2560)
* BiRefNet-HR-matting model (high resolution matting)
* BiRefNet_lite model (lightweight version for faster processing)
* BiRefNet_lite-2K model (lightweight version for 2K resolution)
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

### SDMatte Models (Manual Download)

*   Auto-downloads to `models/RMBG/SDMatte/` on first run.
*   If network restricted:
    *   Place manually: `models/RMBG/SDMatte/SDMatte.safetensors` (standard) or `SDMatte_plus.safetensors` (plus).
    *   If needed, mirror the Hugging Face repo's structure for the components (`scheduler/`, `text_encoder/`, `tokenizer/`, `unet/`, `vae/`) to `models/RMBG/SDMatte/`.

## Troubleshooting

*   **401 error** when initializing GroundingDINO or missing `models/sam2`:
    *   Delete `%USERPROFILE%\.cache\huggingface\token` and `%USERPROFILE%\.huggingface\token`.
    *   Ensure no `HF_TOKEN`/`HUGGINGFACE_TOKEN` environment variables are set.
    *   Re-run; public repos download anonymously (no login required).
*   **Preview shows "Required input is missing: images":**
    *   Ensure image outputs are connected and that upstream nodes ran successfully.

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

[![Star History](https://api.star-history.com/svg?repos=1038lab/comfyui-rmbg&type=Date)](https://star-history.com/#1038lab/comfyui-rmbg&Date)

**Show your appreciation!** If you find this custom node helpful, please give the repository a star!

## License

GPL-3.0 License