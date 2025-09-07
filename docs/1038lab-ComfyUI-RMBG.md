# ComfyUI-RMBG: Effortlessly Remove Backgrounds and Segment Images with AI

Unleash the power of AI with ComfyUI-RMBG, a cutting-edge custom node for ComfyUI, designed to seamlessly remove backgrounds and perform advanced image segmentation for a variety of applications. [Check out the original repo](https://github.com/1038lab/ComfyUI-RMBG)

## Key Features

*   **Advanced Background Removal:**
    *   Utilizes a diverse range of models including RMBG-2.0, INSPYRENET, BEN, BEN2, and BiRefNet for precise background removal.
    *   Offers multiple background options (transparent, black, white, green, blue, red).
    *   Supports batch processing for efficient workflows.

*   **Precise Object Segmentation:**
    *   Features text-prompted object detection using SAM and GroundingDINO models.
    *   Supports both tag-style ("cat, dog") and natural language ("a person wearing red jacket") prompts.
    *   Includes SAM2 segmentation with various model sizes (Tiny/Small/Base+/Large) for optimal quality and speed.

*   **Specialized Segmentation:**
    *   Offers dedicated nodes for segmenting specific objects such as Clothes and Fashion/accessories.
    *   Provides precise segmentation with detailed categories.

*   **Image Tools and Enhancements:**
    *   Includes nodes for Image Combiner, Image Stitch, Mask Tools, Mask Enhancer, Mask Combiner, and Mask Extractor to improve workflows.

*   **Enhanced Edge Detection:**
    *   Real-time background replacement and improved edge detection for more accurate results.

## Key Updates

*   **v2.9.0 (2025/08/18):** Added `SDMatte Matting` node.
*   **v2.8.0 (2025/08/11):** Added `SAM2Segment` node for advanced text-prompted segmentation, and enhanced color widget support.
*   **v2.7.1 (2025/08/06):** Enhanced LoadImage into three distinct nodes and redesigned ImageStitch node and fixed background color handling issues.
*   **(See original README for a full update log with more details on the new nodes and features!)**

## Installation

Choose your preferred method for installing ComfyUI-RMBG:

### Method 1: Using ComfyUI Manager

1.  Open ComfyUI Manager.
2.  Search for `Comfyui-RMBG`.
3.  Click Install.
4.  Install `requirements.txt` in the ComfyUI-RMBG folder using the embedded Python: `./ComfyUI/python_embeded/python -m pip install -r requirements.txt`

### Method 2: Cloning the Repository

1.  Navigate to your ComfyUI's custom_nodes directory: `cd ComfyUI/custom_nodes`
2.  Clone the repository: `git clone https://github.com/1038lab/ComfyUI-RMBG`
3.  Install `requirements.txt` in the ComfyUI-RMBG folder using the embedded Python: `./ComfyUI/python_embeded/python -m pip install -r requirements.txt`

### Method 3: Comfy CLI

1.  Ensure `pip install comfy-cli` is installed.
2.  Run `comfy node install ComfyUI-RMBG`.
3.  Install `requirements.txt` in the ComfyUI-RMBG folder using the embedded Python: `./ComfyUI/python_embeded/python -m pip install -r requirements.txt`

## Manual Model Download

**Note:** Models are automatically downloaded on first use. Follow these steps if you need to manually download and place the models in the correct folders.

1.  RMBG-2.0: Download from [Hugging Face](https://huggingface.co/1038lab/RMBG-2.0) and place files in `/ComfyUI/models/RMBG/RMBG-2.0`
2.  INSPYRENET: Download from [Hugging Face](https://huggingface.co/1038lab/inspyrenet) and place files in `/ComfyUI/models/RMBG/INSPYRENET`
3.  BEN: Download from [Hugging Face](https://huggingface.co/1038lab/BEN) and place files in `/ComfyUI/models/RMBG/BEN`
4.  BEN2: Download from [Hugging Face](https://huggingface.co/1038lab/BEN2) and place files in `/ComfyUI/models/RMBG/BEN2`
5.  BiRefNet-HR: Download from [Hugging Face](https://huggingface.co/1038lab/BiRefNet_HR) and place files in `/ComfyUI/models/RMBG/BiRefNet-HR`
6.  SAM: Download from [Hugging Face](https://huggingface.co/1038lab/sam) and place files in `/ComfyUI/models/SAM`
7.  SAM2: Download from [Hugging Face](https://huggingface.co/1038lab/sam2) and place files (e.g., `sam2.1_hiera_tiny.safetensors`) in `/ComfyUI/models/sam2`
8.  GroundingDINO: Download from [Hugging Face](https://huggingface.co/1038lab/GroundingDINO) and place files in `/ComfyUI/models/grounding-dino`
9.  Clothes Segment: Download from [Hugging Face](https://huggingface.co/1038lab/segformer_clothes) and place files in `/ComfyUI/models/RMBG/segformer_clothes`
10. Fashion Segment: Download from [Hugging Face](https://huggingface.co/1038lab/segformer_fashion) and place files in `/ComfyUI/models/RMBG/segformer_fashion`
11. BiRefNet: Download from [Hugging Face](https://huggingface.co/1038lab/BiRefNet) and place files in `/ComfyUI/models/RMBG/BiRefNet`
12. SDMatte: Download from [Hugging Face](https://huggingface.co/1038lab/SDMatte) and place files in `/ComfyUI/models/RMBG/SDMatte`

## Usage

### RMBG Node

1.  Load `RMBG (Remove Background)` node from the `🧪AILab/🧽RMBG` category.
2.  Connect an image to the input.
3.  Select a model from the dropdown menu.
4.  Adjust optional parameters as needed (see below).
5.  Receive two outputs:
    *   IMAGE: Processed image with a transparent, black, white, green, blue, or red background.
    *   MASK: Binary mask of the foreground.

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

1.  Load the `RMBG (Remove Background)` node from the `🧪AILab/🧽RMBG` category.
2.  Connect an image to the input.
3.  Select a model from the dropdown menu.
4.  Get two outputs:
    *   IMAGE: Processed image with transparent, black, white, green, blue, or red background
    *   MASK: Binary mask of the foreground

### Segment Node

1.  Load `Segment (RMBG)` node from the `🧪AILab/🧽RMBG` category.
2.  Connect an image to the input.
3.  Enter text prompt (tag-style or natural language).
4.  Select SAM and GroundingDINO models.
5.  Adjust parameters as needed:
    *   Threshold: 0.25-0.35 for broad detection, 0.45-0.55 for precision.
    *   Mask blur and offset for edge refinement.
    *   Background color options.

## Troubleshooting

*   **401 error when initializing GroundingDINO / missing `models/sam2`:**  Delete the huggingface token and re-run.
*   **Preview shows "Required input is missing: images":** Ensure image outputs are connected and upstream nodes ran successfully

## Credits

*   RMBG-2.0: [https://huggingface.co/briaai/RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0)
*   (and many other models - see the original README for a comprehensive list!)
*   Created by: [AILab](https://github.com/1038lab)

## Star History
(See the original README for the Star History chart)

## License

GPL-3.0 License