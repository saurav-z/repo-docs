# ComfyUI-RMBG: Effortlessly Remove Backgrounds and Segment Images in ComfyUI

Tired of tedious background removal? ComfyUI-RMBG offers a powerful and versatile solution, enabling precise image segmentation and background manipulation directly within ComfyUI.  [Check out the original repo for more details!](https://github.com/1038lab/ComfyUI-RMBG)

## Key Features

*   **Advanced Background Removal:** Utilizes cutting-edge models like RMBG-2.0, INSPYRENET, BEN, and BEN2 for precise background removal.
*   **Object & Face Segmentation:**  Accurately segments objects, faces, clothing, and fashion elements using a range of models, including SAM2 and GroundingDINO.
*   **Real-time Background Replacement:** Easily swap backgrounds with transparent, solid colors or custom images.
*   **Enhanced Edge Detection:** Improves accuracy with advanced edge refinement techniques.
*   **Versatile Model Support:** Supports various segmentation models like SAM, SAM2, GroundingDINO, and SDMatte.
*   **User-Friendly Interface:** Integrates seamlessly with ComfyUI, providing intuitive controls and settings.
*   **Batch Processing:** Supports batch image processing, enabling efficient workflows.
*   **Internationalization (i18n) Support:** Supports multiple languages.

## Updates

*   **v2.9.1** (2025/09/12) - See [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v291-20250912) for details.
    *   **(Images & Updates Removed For Brevity - Original Repo Contains Details)**

*   **v2.9.0** (2025/08/18) - See [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v290-20250818) for details.
    *   Added `SDMatte Matting` node

*   **v2.8.0** (2025/08/11) - See [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v280-20250811) for details.
    *   Added `SAM2Segment` node for text-prompted segmentation.
    *   Enhanced color widget support across all nodes

*   **(Further Updates & Images Removed For Brevity - Original Repo Contains Full History)**

## Installation

Choose your preferred method:

1.  **ComfyUI Manager:** Search for "Comfyui-RMBG" in the ComfyUI Manager and install.  Then install requirements with:
    ```bash
    ./ComfyUI/python_embeded/python -m pip install -r requirements.txt
    ```
    *Note:  Use the embedded Python if needed.*

2.  **Clone Repository:**
    ```bash
    cd ComfyUI/custom_nodes
    git clone https://github.com/1038lab/ComfyUI-RMBG
    ```
    Then install requirements with:
    ```bash
    ./ComfyUI/python_embeded/python -m pip install -r requirements.txt
    ```

3.  **Comfy CLI:**
    ```bash
    comfy node install ComfyUI-RMBG
    ```
    Then install requirements with:
    ```bash
    ./ComfyUI/python_embeded/python -m pip install -r requirements.txt
    ```

4.  **Manually Download Models:** The models are auto-downloaded when the nodes are first used. You can manually download and place model files into the `/ComfyUI/models/RMBG/` or `/ComfyUI/models/SAM/` or `/ComfyUI/models/grounding-dino/` etc folders based on the models. Refer to the original README for model download links.

## Usage

### RMBG Node (Background Removal)

1.  Load the `RMBG (Remove Background)` node from the `🧪AILab/🧽RMBG` category.
2.  Connect an image input.
3.  Select a model.
4.  Adjust optional parameters (see below).
5.  Outputs: Processed image (with background) and mask.

### Optional Settings :bulb: Tips

| Optional Settings         | :memo: Description                                                           | :bulb: Tips                                                                                   |
|---------------------------|-----------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| **Sensitivity**           | Adjusts mask detection strength. Higher values mean stricter detection. | Default is 0.5. Adjust based on image complexity; more complex images may need higher values. |
| **Processing Resolution** | Controls processing resolution (detail vs. memory).                         | Values between 256 and 2048. Higher = better detail, but uses more memory.  Default: 1024.         |
| **Mask Blur**             | Blurs mask edges.                                                         | Default: 0.  Values 1-5 can smooth edges.                                                       |
| **Mask Offset**           | Expands or shrinks the mask.                                                 | Default: 0.  Fine-tune between -10 and 10.                                                       |
| **Background**            | Select background color. | Alpha (transparent), Black, White, Green, Blue, Red.    |
| **Invert Output**         | Flip mask and image output. | Invert both image and mask output.       |
| **Refine Foreground**      | Use Fast Foreground Color Estimation for optimization of transparent background | Enable for better edge quality and transparency handling |
| **Performance Optimization** | Set options can enhance performance when processing multiple images. | Consider increasing `process_res` and `mask_blur` values for better results, if memory allows, but be mindful of memory usage. |

### Basic Usage
1. Load `RMBG (Remove Background)` node from the `🧪AILab/🧽RMBG` category
2. Connect an image to the input
3. Select a model from the dropdown menu
4. select the parameters as needed (optional)
3. Get two outputs:
   - IMAGE: Processed image with transparent, black, white, green, blue, or red background
   - MASK: Binary mask of the foreground

### Segment Node (Object Segmentation)

1.  Load the `Segment (RMBG)` node from `🧪AILab/🧽RMBG`.
2.  Connect an image input.
3.  Enter a text prompt (tag-style or natural language).
4.  Select SAM or GroundingDINO models.
5.  Adjust parameters.

    *   Threshold: 0.25-0.35 (broad), 0.45-0.55 (precise).
    *   Mask blur and offset for edge refinement.
    *   Background color options.

### Parameters
- `sensitivity`: Controls the background removal sensitivity (0.0-1.0)
- `process_res`: Processing resolution (512-2048, step 128)
- `mask_blur`: Blur amount for the mask (0-64)
- `mask_offset`: Adjust mask edges (-20 to 20)
- `background`: Choose output background color
- `invert_output`: Flip mask and image output
- `optimize`: Toggle model optimization

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
*   Required packages (automatically installed during install):
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

## Troubleshooting (short)
- 401 error when initializing GroundingDINO / missing `models/sam2`:
  - Delete `%USERPROFILE%\.cache\huggingface\token` (and `%USERPROFILE%\.huggingface\token` if present)
  - Ensure no `HF_TOKEN`/`HUGGINGFACE_TOKEN` env vars are set
  - Re-run; public repos download anonymously (no login required)
- Preview shows "Required input is missing: images":
  - Ensure image outputs are connected and upstream nodes ran successfully

## Credits

*   RMBG-2.0: https://huggingface.co/briaai/RMBG-2.0
*   INSPYRENET: https://github.com/plemeri/InSPyReNet
*   BEN: https://huggingface.co/PramaLLC/BEN
*   BEN2: https://huggingface.co/PramaLLC/BEN2
*   BiRefNet: https://huggingface.co/ZhengPeng7
*   SAM: https://huggingface.co/facebook/sam-vit-base
*   GroundingDINO: https://github.com/IDEA-Research/GroundingDINO
*   Clothes Segment: https://huggingface.co/mattmdjaga/segformer_b2_clothes
*   SDMatte: https://github.com/vivoCameraResearch/SDMatte

*   Created by: [AILab](https://github.com/1038lab)

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