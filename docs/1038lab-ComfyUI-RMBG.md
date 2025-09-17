# ComfyUI-RMBG: Effortlessly Remove Backgrounds and Segment Images in ComfyUI

Tired of complicated background removal? **ComfyUI-RMBG** is your all-in-one solution for precise image background removal, object segmentation, and advanced image manipulation within ComfyUI.  [Visit the original repository](https://github.com/1038lab/ComfyUI-RMBG) for the latest updates and to contribute!

## Key Features

*   **Advanced Background Removal:** Utilize a range of models like RMBG-2.0, INSPYRENET, BEN, BEN2, BiRefNet, and SDMatte models for accurate and versatile background removal.
*   **Precise Object Segmentation:** Leverage text prompts with SAM, SAM2, and GroundingDINO models to isolate specific objects, faces, clothing, and fashion elements with ease.
*   **Real-Time Background Replacement:** Seamlessly integrate objects into new backgrounds or create transparent images.
*   **Enhanced Edge Detection:** Improve accuracy with features for better edge refinement and detail preservation.
*   **Flexible Parameter Control:** Fine-tune results with adjustable sensitivity, processing resolution, mask blurring, and mask offset settings.
*   **Batch Processing Support:** Process multiple images at once for efficient workflows.
*   **Multiple Model Support:**  Choose the best model for your task, from general-purpose to specialized portrait, clothes and fashion segmentation.

## News & Updates

*   **v2.9.1 (2025/09/12):**  Update details in [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v291-20250912)
    ![v2.9.1](https://github.com/user-attachments/assets/9b6c3e6c-5866-4807-91ba-669eb7efc52b)
*   **v2.9.0 (2025/08/18):**  Added `SDMatte Matting` node.  See [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v290-20250818).
    ![v2 9 0](https://github.com/user-attachments/assets/de4398ab-ce3c-4c3e-af0b-d82c2a8c8481)
*   **v2.8.0 (2025/08/11):**  Added `SAM2Segment` node and enhanced color widget support.  See [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v280-20250811).
    ![v2 8 0](https://github.com/user-attachments/assets/16c5a67c-1aec-4def-9aa2-db9dcf2354a8)
*   **v2.7.1 (2025/08/06):** Enhanced LoadImage nodes, redesigned ImageStitch node, and fixed background color handling issues.  See [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v271-20250806).
    ![v2.7.0_ImageStitch](https://github.com/user-attachments/assets/3f31fe25-a453-4f86-bf3d-dc12a8affd39)
*   **v2.6.0 (2025/07/15):** Added `Kontext Refence latent Mask` node. See [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v260-20250715).
    ![ReferenceLatentMaskr](https://github.com/user-attachments/assets/756641b7-0833-4fe0-b32f-2b848a14574e)
*   **v2.5.2 (2025/07/11):**  See [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v252-20250711).
    ![V 2 5 2](https://github.com/user-attachments/assets/4b41887a-0d8a-4a5a-9128-1e866f410b60)
*   **v2.5.1 (2025/07/07):**  See [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v251-20250707).
*   **v2.5.0 (2025/07/01):** Added new nodes and BiRefNet models. See [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v250-20250701).
    ![mask_overlay](https://github.com/user-attachments/assets/d82abb5a-9702-4d21-a5cf-e6776c7b4c06)
*   **v2.4.0 (2025/06/01):** Added new nodes and new Segment V2. See [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v240-20250601).
    ![ComfyUI-RMBG_V2 4 0 new nodes](https://github.com/user-attachments/assets/7ab023e7-70b4-4b97-910a-e608c03841cf)
*   **v2.3.2 (2025/05/15):**  See [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v232-20250515).
    ![v 2 3 2](https://github.com/user-attachments/assets/fc852183-6796-4ef7-a41a-499dbe6a4519)
*   **v2.3.1 (2025/05/02):**  See [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v231-20250502).
*   **v2.3.0 (2025/05/01):**  Added new nodes and resizing options for Load Image. See [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v230-20250501).
    ![v2 3 0_node](https://github.com/user-attachments/assets/f53be704-bb53-4fdf-9e7f-fad00dcd5add)
*   **v2.2.1 (2025/04/05):**  See [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v221-20250405).
*   **v2.2.0 (2025/04/05):** Added new nodes and fixed compatibility issues. See [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v220-20250405).
    ![Comfyu-rmbg_v2 2 1_node_sample](https://github.com/user-attachments/assets/68f4233c-b992-473e-aa30-ca32086f5221)
*   **v2.1.1 (2025/03/21):** Enhanced compatibility with Transformers. See [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v211-20250321).
*   **v2.1.0 (2025/03/19):** Integrated internationalization (i18n) support. See [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v210-20250319).
    https://github.com/user-attachments/assets/7faa00d3-bbe2-42b8-95ed-2c830a1ff04f
*   **v2.0.0 (2025/03/13):** Added Image and Mask Tools, enhanced code structure, and added a new category path. See [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v200-20250313).
    ![image_mask_preview](https://github.com/user-attachments/assets/5e2b2679-4b63-4db1-a6c1-3b26b6f97df3)
*   **v1.9.3 (2025/02/24):** Cleaned up code and fixed issues. See [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v193-20250224).
*   **v1.9.2 (2025/02/21):** Added Fast Foreground Color Estimation. See [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v192-20250221).
    ![RMBG_V1 9 2](https://github.com/user-attachments/assets/aaf51bff-931b-47ef-b20b-0dabddc49873)
*   **v1.9.1 (2025/02/20):** Changed repository for model management and reorganized models files. See [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v191-20250220).
*   **v1.9.0 (2025/02/19):** Improved BiRefNet model and memory management. See [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v190-20250219).
    ![rmbg_v1 9 0](https://github.com/user-attachments/assets/a7649781-42c9-4af4-94c7-6841e9395f5a)
*   **v1.8.0 (2025/02/07):** Added new BiRefNet-HR model. See [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v180-20250207).
    ![RMBG-v1 8 0](https://github.com/user-attachments/assets/d4a1309c-a635-443a-97b5-2639fb48c27a)
*   **v1.7.0 (2025/02/04):** Added a new BEN2 model. See [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v170-20250204).
    ![rmbg_v1 7 0](https://github.com/user-attachments/assets/22053105-f3db-4e24-be66-ae0ad2cc248e)
*   **v1.6.0 (2025/01/22):** Added new Face Segment custom node. See [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v160-20250122).
    ![RMBG_v1 6 0](https://github.com/user-attachments/assets/9ccefec1-4370-4708-a12d-544c90888bf2)
*   **v1.5.0 (2025/01/05):** Added new Fashion and accessories Segment custom node. See [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v150-20250105).
    ![RMBGv_1 5 0](https://github.com/user-attachments/assets/a250c1a6-8425-4902-b902-a6e1a8bfe959)
*   **v1.4.0 (2025/01/02):** Added intelligent clothes segmentation. See [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v140-20250102).
    ![rmbg_v1 4 0](https://github.com/user-attachments/assets/978c168b-03a8-4937-aa03-06385f34b820)
*   **v1.3.2 (2024/12/29):** Enhanced background handling. See [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v132-20241229).
*   **v1.3.1 (2024/12/25):** Bug fixes. See [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v131-20241225).
*   **v1.3.0 (2024/12/23):** Added new Segment node. See [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v140-20241222).
    ![rmbg v1.3.0](https://github.com/user-attachments/assets/7607546e-ffcb-45e2-ab90-83267292757e)
*   **v1.2.2 (2024/12/12):**  See [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v122-20241212).
    ![RMBG1 2 2](https://github.com/user-attachments/assets/cb7b1ad0-a2ca-4369-9401-54957af6c636)
*   **v1.2.1 (2024/12/02):**  See [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.mdv121-20241202).
    ![GIF_TO_AWEBP](https://github.com/user-attachments/assets/7f8275d0-06e5-4880-adfe-930f045df673)
*   **v1.2.0 (2024/11/29):**  See [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v120-20241129).
    ![RMBGv1 2 0](https://github.com/user-attachments/assets/4fd10123-6c95-4f9e-8d25-fdb39b5fc792)
*   **v1.1.0 (2024/11/21):**  See [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v110-20241121).
    ![comfyui-rmbg version compare](https://github.com/user-attachments/assets/2d23cf42-ca74-49e5-a8bf-9de377bd71aa)

## Installation

Choose your preferred installation method:

### 1. ComfyUI Manager (Recommended)

*   Search for `Comfyui-RMBG` in the ComfyUI Manager and install it.
*   Install the requirements:
    ```bash
    ./ComfyUI/python_embeded/python -m pip install -r requirements.txt
    ```
    *Tip:  If you encounter issues with the system Python, use ComfyUI's embedded Python:*
    ```bash
    ./ComfyUI/python_embeded/python.exe -m pip install --no-user --no-cache-dir -r requirements.txt
    ```

### 2. Manual Installation

*   Clone the repository into your ComfyUI `custom_nodes` directory:

    ```bash
    cd ComfyUI/custom_nodes
    git clone https://github.com/1038lab/ComfyUI-RMBG
    ```

*   Install the requirements:

    ```bash
    ./ComfyUI/python_embeded/python -m pip install -r requirements.txt
    ```

### 3. Comfy CLI

*   Ensure `pip install comfy-cli` is installed.
*   Install ComfyUI if you don't have it:  `comfy install`
*   Install ComfyUI-RMBG:

    ```bash
    comfy node install ComfyUI-RMBG
    ```
*   Install the requirements:

    ```bash
    ./ComfyUI/python_embeded/python -m pip install -r requirements.txt
    ```

### 4. Model Downloads

*   Models are automatically downloaded to `ComfyUI/models/RMBG/` upon first use.
*   **Manual Download (if automatic download fails):**
    *   **RMBG-2.0:** [https://huggingface.co/1038lab/RMBG-2.0](https://huggingface.co/1038lab/RMBG-2.0) (Place files in `/ComfyUI/models/RMBG/RMBG-2.0`)
    *   **INSPYRENET:** [https://huggingface.co/1038lab/inspyrenet](https://huggingface.co/1038lab/inspyrenet) (Place files in `/ComfyUI/models/RMBG/INSPYRENET`)
    *   **BEN:** [https://huggingface.co/1038lab/BEN](https://huggingface.co/1038lab/BEN) (Place files in `/ComfyUI/models/RMBG/BEN`)
    *   **BEN2:** [https://huggingface.co/1038lab/BEN2](https://huggingface.co/1038lab/BEN2) (Place files in `/ComfyUI/models/RMBG/BEN2`)
    *   **BiRefNet-HR:** [https://huggingface.co/1038lab/BiRefNet_HR](https://huggingface.co/1038lab/BiRefNet_HR) (Place files in `/ComfyUI/models/RMBG/BiRefNet-HR`)
    *   **SAM:** [https://huggingface.co/1038lab/sam](https://huggingface.co/1038lab/sam) (Place files in `/ComfyUI/models/SAM`)
    *   **SAM2:** [https://huggingface.co/1038lab/sam2](https://huggingface.co/1038lab/sam2) (Place files in `/ComfyUI/models/sam2`)
    *   **GroundingDINO:** [https://huggingface.co/1038lab/GroundingDINO](https://huggingface.co/1038lab/GroundingDINO) (Place files in `/ComfyUI/models/grounding-dino`)
    *   **Clothes Segment:** [https://huggingface.co/1038lab/segformer_clothes](https://huggingface.co/1038lab/segformer_clothes) (Place files in `/ComfyUI/models/RMBG/segformer_clothes`)
    *   **Fashion Segment:** [https://huggingface.co/1038lab/segformer_fashion](https://huggingface.co/1038lab/segformer_fashion) (Place files in `/ComfyUI/models/RMBG/segformer_fashion`)
    *   **BiRefNet:** [https://huggingface.co/1038lab/BiRefNet](https://huggingface.co/1038lab/BiRefNet) (Place files in `/ComfyUI/models/RMBG/BiRefNet`)
    *   **SDMatte:** [https://huggingface.co/1038lab/SDMatte](https://huggingface.co/1038lab/SDMatte) (Place files in `/ComfyUI/models/RMBG/SDMatte`)

## Usage

### RMBG Node

1.  Load the `RMBG (Remove Background)` node from the `🧪AILab/🧽RMBG` category.
2.  Connect an image to the input.
3.  Select a model from the dropdown menu.
4.  Adjust parameters as needed (optional - see "Optional Settings" below).
5.  Receive two outputs:
    *   `IMAGE`: Processed image with a transparent, black, white, green, blue, or red background.
    *   `MASK`: A binary mask of the foreground.

### Segment Node

1.  Load the `Segment (RMBG)` node from the `🧪AILab/🧽RMBG` category.
2.  Connect an image to the input.
3.  Enter a text prompt (tag-style or natural language).
4.  Select SAM and GroundingDINO models.
5.  Adjust parameters as needed:
    *   Threshold: 0.25-0.35 for broad detection, 0.45-0.55 for precision
    *   Mask blur and offset for edge refinement
    *   Background color options

### Optional Settings :bulb: Tips

| Setting              | Description                                                            | :bulb: Tip                                                                                              |
| -------------------- | ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| **Sensitivity**      | Adjusts mask detection strength (higher = stricter).                   | Default: 0.5. Increase for complex images.                                                           |
| **Processing Resolution** | Controls image resolution, affecting detail and memory usage.      | Choose between 256-2048 (step 128). Higher res = better detail, higher memory.                        |
| **Mask Blur**        | Blurs mask edges for smoother results.                                 | Default: 0.  Try 1-5 for smoother edges.                                                              |
| **Mask Offset**      | Expands/shrinks the mask boundary.                                     | Default: 0. Fine-tune between -10 and 10.                                                              |
| **Background**      | Choose output background color. | Alpha (transparent background) Black, White, Green, Blue, Red |
| **Invert Output**      | Flip mask and image output | Invert both image and mask output |
| **Refine Foreground** | Use Fast Foreground Color Estimation to optimize transparent background | Enable for better edge quality and transparency handling |
| **Performance Optimization** | Properly setting options can enhance performance when processing multiple images. | If memory allows, consider increasing `process_res` and `mask_blur` values for better results, but be mindful of memory usage. |

### Basic Usage

1.  Load the `RMBG (Remove Background)` node from the `🧪AILab/🧽RMBG` category
2.  Connect an image to the input
3.  Select a model from the dropdown menu
4.  select the parameters as needed (optional)
3.  Get two outputs:
   - IMAGE: Processed image with transparent, black, white, green, blue, or red background
   - MASK: Binary mask of the foreground

### Parameters

-   `sensitivity`: Controls the background removal sensitivity (0.0-1.0)
-   `process_res`: Processing resolution (512-2048, step 128)
-   `mask_blur`: Blur amount for the mask (0-64)
-   `mask_offset`: Adjust mask edges (-20 to 20)
-   `background`: Choose output background color
-   `invert_output`: Flip mask and image output
-   `optimize`: Toggle model optimization

### Segment Node

1. Load `Segment (RMBG)` node from the `🧪AILab/🧽RMBG` category
2. Connect an image to the input
3. Enter text prompt (tag-style or natural language)
4. Select SAM and GroundingDINO models
5. Adjust parameters as needed:
   - Threshold: 0.25-0.35 for broad detection, 0.45-0.55 for precision
   - Mask blur and offset for edge refinement
   - Background color options

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
- Auto-download on first run to `models/RMBG/SDMatte/`
- If network restricted, place weights manually:
  - `models/RMBG/SDMatte/SDMatte.safetensors` (standard) or `SDMatte_plus.safetensors` (plus)
  - Components (config files) are auto-downloaded; if needed, mirror the structure from the Hugging Face repo to `models/RMBG/SDMatte/` (`scheduler/`, `text_encoder/`, `tokenizer/`, `unet/`, `vae/`)

## Troubleshooting

*   **401 error initializing GroundingDINO / missing `models/sam2`:**
    *   Delete `%USERPROFILE%\.cache\huggingface\token` and `%USERPROFILE%\.huggingface\token` (if present)
    *   Ensure no `HF_TOKEN`/`HUGGINGFACE_TOKEN` environment variables are set.
    *   Re-run; public repos download anonymously (no login required)
*   **Preview shows "Required input is missing: images":**
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

<a href="https://www.star-history.com/#1038lab/comfyui-rmbg&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=1038lab/comfyui-rmbg&type=Date&theme=dark" />
   <source media="(prefers-color