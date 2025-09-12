# Enhance Your Images with ComfyUI-RMBG: Advanced Background Removal and Segmentation

Tired of tedious background removal? ComfyUI-RMBG is your all-in-one solution for **precise image background removal and object segmentation** within ComfyUI, offering a wide range of models and features.  Access the original repo [here](https://github.com/1038lab/ComfyUI-RMBG).

## Key Features

*   **Advanced Background Removal:**  Utilize state-of-the-art models like RMBG-2.0, INSPYRENET, BEN, BEN2, and BiRefNet for accurate and detailed background removal.
*   **Text-Prompted Segmentation:** Segment objects, faces, clothing, and fashion elements using text prompts with SAM, SAM2 and GroundingDINO, offering unparalleled control.
*   **Real-Time Background Replacement:** Easily replace backgrounds for seamless image integration.
*   **Enhanced Edge Detection:** Improve accuracy and detail with advanced edge detection techniques.
*   **Model Variety:** Choose from a diverse set of models, including SDMatte for specialized matting tasks.
*   **User-Friendly Interface:**  Control parameters like sensitivity, processing resolution, and mask blur for optimal results.
*   **Batch Processing:** Supports batch processing to streamline your workflow.
*   **Internationalization (i18n):** Support for multiple languages for ease of use.
*   **Image Tools:** Image Combiner, Image Stitch, Image/Mask Converter, Mask Enhancer, Mask Combiner, and Mask Extractor.

## Updates
*   **v2.9.0** Added `SDMatte Matting` node
*   **v2.8.0** Added `SAM2Segment` node for text-prompted segmentation with the latest Facebook Research SAM2 technology.
*   **v2.7.1** Enhanced LoadImage into three distinct nodes to meet different needs, all supporting direct image loading from local paths or URLs
*   **v2.6.0** Added `Kontext Refence latent Mask` node
*   **v2.5.2** Bug Fixes
*   **v2.5.1** Bug Fixes
*   **v2.5.0** Added `MaskOverlay`, `ObjectRemover`, `ImageMaskResize` new nodes.
*   **v2.4.0** Added `CropObject`, `ImageCompare`, `ColorInput` nodes and new Segment V2 (see update.md for details)
*   **v2.3.2** Bug Fixes
*   **v2.3.1** Bug Fixes
*   **v2.3.0** Added new nodes: IC-LoRA Concat, Image Crop
*   **v2.2.1** Bug Fixes
*   **v2.2.0** Added new nodes: Image Combiner, Image Stitch, Image/Mask Converter, Mask Enhancer, Mask Combiner, and Mask Extractor
*   **v2.1.1** Enhanced compatibility with Transformers
*   **v2.1.0** Integrated internationalization (i18n) support for multiple languages.
*   **v2.0.0** Added Image and Mask Tools improved functionality.
*   **v1.9.3** Clean up the code and fix the issue
*   **v1.9.2** with Fast Foreground Color Estimation
*   **v1.9.1** Changed repository for model management to the new repository and Reorganized models files structure for better maintainability.
*   **v1.9.0** with BiRefNet model improvements
*   **v1.8.0** with new BiRefNet-HR model
*   **v1.7.0** with new BEN2 model
*   **v1.6.0** with new Face Segment custom node
*   **v1.5.0** with new Fashion and accessories Segment custom node
*   **v1.4.0** with new Clothes Segment node
*   **v1.3.2** with background handling
*   **v1.3.1** with bug fixes
*   **v1.3.0** with new Segment node
*   **v1.2.2** Bug Fixes
*   **v1.2.1** Bug Fixes
*   **v1.2.0** Bug Fixes
*   **v1.1.0** Added features
## Installation

Choose your preferred method:

### 1. ComfyUI Manager Installation
   *   Search for `Comfyui-RMBG` in the ComfyUI Manager and install.
   *   Install `requirements.txt`:
      ```bash
      ./ComfyUI/python_embeded/python -m pip install -r requirements.txt
      ```

### 2. Manual Installation
   *   Clone the repository into your ComfyUI `custom_nodes` folder:
      ```bash
      cd ComfyUI/custom_nodes
      git clone https://github.com/1038lab/ComfyUI-RMBG
      ```
   *   Install the dependencies:
      ```bash
      ./ComfyUI/python_embeded/python -m pip install -r requirements.txt
      ```

### 3. Comfy CLI Installation
   *   Ensure `pip install comfy-cli` is installed.
   *   Install ComfyUI  `comfy install` (if you don't have ComfyUI Installed)
   *   Install ComfyUI-RMBG
      ```bash
      comfy node install ComfyUI-RMBG
      ```
   *   Install the dependencies:
      ```bash
      ./ComfyUI/python_embeded/python -m pip install -r requirements.txt
      ```
### 4. Model Downloads
*   Models are automatically downloaded upon first use to `ComfyUI/models/RMBG/`
*   Alternatively, manually download models from their respective Hugging Face repos (links provided below) and place them in the appropriate `ComfyUI/models/RMBG/` subfolders (e.g., `/RMBG-2.0`, `/INSPYRENET`, `/BEN2`, `/BiRefNet-HR`, `/SDMatte`, `/SAM`,`/SAM2`, `/grounding-dino`, `/segformer_clothes`, `/segformer_fashion`, `/BiRefNet`).

## Usage

### RMBG Node
1.  Load the `RMBG (Remove Background)` node from the `🧪AILab/🧽RMBG` category.
2.  Connect an image to the input.
3.  Select a model.
4.  Adjust optional parameters as needed.
5.  Get the processed image (with background) and mask outputs.

### Optional Settings :bulb: Tips
| Optional Settings    | :memo: Description                                                           | :bulb: Tips                                                                                   |
|----------------------|-----------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| **Sensitivity**      | Adjusts the strength of mask detection. Higher values result in stricter detection. | Default value is 0.5. Adjust based on image complexity; more complex images may require higher sensitivity. |
| **Processing Resolution** | Controls the processing resolution of the input image, affecting detail and memory usage. | Choose a value between 256 and 2048, with a default of 1024. Higher resolutions provide better detail but increase memory consumption. |
| **Mask Blur**        | Controls the amount of blur applied to the mask edges, reducing jaggedness. | Default value is 0. Try setting it between 1 and 5 for smoother edge effects.                    |
| **Mask Offset**      | Allows for expanding or shrinking the mask boundary. Positive values expand the boundary, while negative values shrink it. | Default value is 0. Adjust based on the specific image, typically fine-tuning between -10 and 10. |
| **Background**      | Choose output background color | Alpha (transparent background) Black, White, Green, Blue, Red |
| **Invert Output**      | Flip mask and image output | Invert both image and mask output |
| **Refine Foreground** | Use Fast Foreground Color Estimation to optimize transparent background | Enable for better edge quality and transparency handling |
| **Performance Optimization** | Properly setting options can enhance performance when processing multiple images. | If memory allows, consider increasing `process_res` and `mask_blur` values for better results, but be mindful of memory usage. |

### Basic Usage
1. Load `RMBG (Remove Background)` node from the `🧪AILab/🧽RMBG` category
2. Connect an image to the input
3. Select a model from the dropdown menu
4. select the parameters as needed (optional)
3. Get two outputs:
   - IMAGE: Processed image with transparent, black, white, green, blue, or red background
   - MASK: Binary mask of the foreground

### Parameters
- `sensitivity`: Controls the background removal sensitivity (0.0-1.0)
- `process_res`: Processing resolution (512-2048, step 128)
- `mask_blur`: Blur amount for the mask (0-64)
- `mask_offset`: Adjust mask edges (-20 to 20)
- `background`: Choose output background color
- `invert_output`: Flip mask and image output
- `optimize`: Toggle model optimization

### Segment Node
1.  Load the `Segment (RMBG)` node from `🧪AILab/🧽RMBG`.
2.  Connect an image.
3.  Enter a text prompt (tag-style or natural language).
4.  Select SAM or GroundingDINO models.
5.  Adjust parameters as needed.

## About Models (Expanded Details)

<details>
<summary><h2>About Models</h2></summary>

## RMBG-2.0
RMBG-2.0 is is developed by BRIA AI and uses the BiRefNet architecture which includes:
- High accuracy in complex environments
- Precise edge detection and preservation
- Excellent handling of fine details
- Support for multiple objects in a single image
- Output Comparison
- Output with background
- Batch output for video
The model is trained on a diverse dataset of over 15,000 high-quality images, ensuring:
- Balanced representation across different image types
- High accuracy in various scenarios
- Robust performance with complex backgrounds

## INSPYRENET
INSPYRENET is specialized in human portrait segmentation, offering:
- Fast processing speed
- Good edge detection capability
- Ideal for portrait photos and human subjects

## BEN
BEN is robust on various image types, offering:
- Good balance between speed and accuracy
- Effective on both simple and complex scenes
- Suitable for batch processing

## BEN2
BEN2 is a more advanced version of BEN, offering:
- Improved accuracy and speed
- Better handling of complex scenes
- Support for more image types
- Suitable for batch processing

## BIREFNET MODELS
BIREFNET is a powerful model for image segmentation, offering:
- BiRefNet-general purpose model (balanced performance)
- BiRefNet_512x512 model (optimized for 512x512 resolution)
- BiRefNet-portrait model (optimized for portrait/human matting)
- BiRefNet-matting model (general purpose matting)
- BiRefNet-HR model (high resolution up to 2560x2560)
- BiRefNet-HR-matting model (high resolution matting)
- BiRefNet_lite model (lightweight version for faster processing)
- BiRefNet_lite-2K model (lightweight version for 2K resolution)

## SAM
SAM is a powerful model for object detection and segmentation, offering:
- High accuracy in complex environments
- Precise edge detection and preservation
- Excellent handling of fine details
- Support for multiple objects in a single image
- Output Comparison
- Output with background
- Batch output for video

## SAM2
SAM2 is the latest segmentation model family designed for efficient, high-quality text-prompted segmentation:
- Multiple sizes: Tiny, Small, Base+, Large
- Optimized inference with strong accuracy
- Automatic download on first use; manual placement supported in `ComfyUI/models/sam2`

## GroundingDINO
GroundingDINO is a model for text-prompted object detection and segmentation, offering:
- High accuracy in complex environments
- Precise edge detection and preservation
- Excellent handling of fine details
- Support for multiple objects in a single image
- Output Comparison
- Output with background
- Batch output for video

## BiRefNet Models
- BiRefNet-general purpose model (balanced performance)
- BiRefNet_512x512 model (optimized for 512x512 resolution)
- BiRefNet-portrait model (optimized for portrait/human matting)
- BiRefNet-matting model (general purpose matting)
- BiRefNet-HR model (high resolution up to 2560x2560)
- BiRefNet-HR-matting model (high resolution matting)
- BiRefNet_lite model (lightweight version for faster processing)
- BiRefNet_lite-2K model (lightweight version for 2K resolution)
</details>

## Requirements

*   ComfyUI
*   Python 3.10+
*   Automatically installed packages include:
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
- Auto-download on first run to `models/RMBG/SDMatte/`
- If network restricted, place weights manually:
  - `models/RMBG/SDMatte/SDMatte.safetensors` (standard) or `SDMatte_plus.safetensors` (plus)
  - Components (config files) are auto-downloaded; if needed, mirror the structure from the Hugging Face repo to `models/RMBG/SDMatte/` (`scheduler/`, `text_encoder/`, `tokenizer/`, `unet/`, `vae/`)

## Troubleshooting

*   **401 error** when initializing GroundingDINO or missing `models/sam2`:
    *   Delete `%USERPROFILE%\.cache\huggingface\token` and `%USERPROFILE%\.huggingface\token` if present.
    *   Ensure no `HF_TOKEN`/`HUGGINGFACE_TOKEN` environment variables are set.
    *   Re-run (public repos download anonymously).
*   **Preview shows "Required input is missing: images"**:
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

*   Developed by: [AILab](https://github.com/1038lab)

## Star History

<a href="https://www.star-history.com/#1038lab/comfyui-rmbg&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=1038lab/comfyui-rmbg&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=1038lab/comfyui-rmbg&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=1038lab/comfyui-rmbg&type=Date" />
 </picture>
</a>

## License

GPL-3.0 License