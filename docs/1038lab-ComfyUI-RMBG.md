# ComfyUI-RMBG: Effortlessly Remove Backgrounds and Segment Images in ComfyUI

Easily remove backgrounds, segment objects, and refine images with advanced models in ComfyUI using [ComfyUI-RMBG](https://github.com/1038lab/ComfyUI-RMBG), a powerful custom node.

## Key Features

*   **Advanced Background Removal:** Utilize models like RMBG-2.0, INSPYRENET, BEN, and BEN2 for precise background removal.
*   **Object and Face Segmentation:** Segment objects, faces, clothing, and fashion elements with text prompts using SAM, SAM2, and GroundingDINO.
*   **Real-time Background Replacement:** Quickly replace backgrounds and improve accuracy with enhanced edge detection.
*   **Multiple Models:** Supports a wide variety of models, including BiRefNet, SDMatte and more, offering flexibility and precision.
*   **User-Friendly:** Easy to install and use with clear parameters.
*   **Regular Updates:** Benefit from frequent updates with new models and features.

## What's New

*   **v2.9.0:** Added `SDMatte Matting` node
*   **v2.8.0:** Added `SAM2Segment` node for text-prompted segmentation with the latest Facebook Research SAM2 technology.
*   **v2.7.1:** Enhanced LoadImage into three distinct nodes to meet different needs
*   **v2.6.0:** Added `Kontext Refence latent Mask` node, Which uses a reference latent and mask for precise region conditioning.
*   **See all the updates:**  [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md)

## Installation

Choose your preferred installation method:

### Method 1: Using ComfyUI Manager

1.  In your ComfyUI interface, open the ComfyUI Manager.
2.  Search for `Comfyui-RMBG` and install it directly.
3.  Install requirements.txt in the ComfyUI-RMBG folder:
    ```bash
    ./ComfyUI/python_embeded/python -m pip install -r requirements.txt
    ```

### Method 2: Cloning the Repository

1.  Navigate to your ComfyUI custom\_nodes directory:
    ```bash
    cd ComfyUI/custom_nodes
    ```
2.  Clone the repository:
    ```bash
    git clone https://github.com/1038lab/ComfyUI-RMBG
    ```
3.  Install requirements.txt in the ComfyUI-RMBG folder:
    ```bash
    ./ComfyUI/python_embeded/python -m pip install -r requirements.txt
    ```

### Method 3: Using Comfy CLI

1.  Make sure that `pip install comfy-cli` is installed.
2.  Install the ComfyUI-RMBG, using the following command:
    ```bash
    comfy node install ComfyUI-RMBG
    ```
3.  Install requirements.txt in the ComfyUI-RMBG folder:
    ```bash
    ./ComfyUI/python_embeded/python -m pip install -r requirements.txt
    ```

## Usage

### RMBG Node

1.  Load the `RMBG (Remove Background)` node from the `🧪AILab/🧽RMBG` category.
2.  Connect an image to the input.
3.  Select a model from the dropdown menu.
4.  (Optional) Adjust parameters such as sensitivity, processing resolution, mask blur, and mask offset.
5.  Get two outputs:
    *   **IMAGE:** Processed image with a transparent, black, white, green, blue, or red background.
    *   **MASK:** A binary mask of the foreground.

### Segment Node

1.  Load the `Segment (RMBG)` node from the `🧪AILab/🧽RMBG` category.
2.  Connect an image to the input.
3.  Enter a text prompt (tag-style or natural language).
4.  Select the desired SAM or GroundingDINO models.
5.  Adjust parameters like threshold, mask blur, and offset as needed.

### Optional Settings :bulb: Tips

| Optional Settings       | :memo: Description                                                           | :bulb: Tips                                                                                   |
| ----------------------- | ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| **Sensitivity**         | Adjusts the strength of mask detection. Higher values result in stricter detection. | Default value is 0.5. Adjust based on image complexity; more complex images may require higher sensitivity. |
| **Processing Resolution** | Controls the processing resolution of the input image, affecting detail and memory usage. | Choose a value between 256 and 2048, with a default of 1024. Higher resolutions provide better detail but increase memory consumption. |
| **Mask Blur**           | Controls the amount of blur applied to the mask edges, reducing jaggedness. | Default value is 0. Try setting it between 1 and 5 for smoother edge effects.                    |
| **Mask Offset**         | Allows for expanding or shrinking the mask boundary. Positive values expand the boundary, while negative values shrink it. | Default value is 0. Adjust based on the specific image, typically fine-tuning between -10 and 10. |
| **Background**          | Choose output background color | Alpha (transparent background) Black, White, Green, Blue, Red |
| **Invert Output**          | Flip mask and image output | Invert both image and mask output |
| **Refine Foreground** | Use Fast Foreground Color Estimation to optimize transparent background | Enable for better edge quality and transparency handling |
| **Performance Optimization** | Properly setting options can enhance performance when processing multiple images. | If memory allows, consider increasing `process_res` and `mask_blur` values for better results, but be mindful of memory usage. |

## Model Management

*   Models are automatically downloaded to the `ComfyUI/models/RMBG/` and `ComfyUI/models/SAM/` directories upon first use.
*   If you prefer manual download, follow the links provided in the original README to download the necessary model files and place them in the corresponding directories.
*   Check the original README for the model links and file structure for a specific model.

## Troubleshooting

*   **401 Errors/Missing SAM2 models:** Delete `%USERPROFILE%\.cache\huggingface\token` and any `HF_TOKEN`/`HUGGINGFACE_TOKEN` environment variables. Re-run; public repos download anonymously.
*   **Preview Error: "Required input is missing: images":** Ensure image outputs are correctly connected and the upstream nodes executed successfully.

## Credits

*   Created by [AILab](https://github.com/1038lab).
*   Based on contributions from various open-source projects (see original README).

## Support and Contribution

If you found this project helpful, please give me ⭐ on this repo!  Your support motivates me to keep improving it!

## License

GPL-3.0 License