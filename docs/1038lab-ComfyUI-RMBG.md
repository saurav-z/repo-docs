# ComfyUI-RMBG: Advanced Background Removal and Segmentation for ComfyUI

**Effortlessly remove backgrounds and segment objects with cutting-edge AI models directly within ComfyUI using ComfyUI-RMBG!**  [View on GitHub](https://github.com/1038lab/ComfyUI-RMBG)

## Key Features

*   **Versatile Background Removal:** Utilize various models including RMBG-2.0, INSPYRENET, BEN, and BEN2 for precise background removal.
*   **Object Segmentation:** Segment objects with text prompts, leveraging the power of SAM and GroundingDINO.
*   **SAM2 Segmentation:** Segment objects with the latest SAM2 models (Tiny/Small/Base+/Large).
*   **Face, Clothes and Fashion Segmentation:** Dedicated nodes for face parsing, clothing, and fashion element segmentation.
*   **Real-time background replacement and enhanced edge detection** for improved accuracy.
*   **Flexible Background Options:** Choose from transparent (alpha), black, white, green, blue, or red backgrounds.
*   **Batch Processing:** Supports processing multiple images simultaneously for efficiency.
*   **Model Variety:** Includes models like BiRefNet, SDMatte, and more.
*   **User-Friendly:** Easy installation via ComfyUI Manager or manual methods.
*   **Up-to-date with new Nodes and Features:** Stay up to date with the latest improvements and features.

## Recent Updates

*   **v2.9.0** (2025/08/18): Added `SDMatte Matting` node.
*   **v2.8.0** (2025/08/11): Added `SAM2Segment` node for text-prompted segmentation with the latest Facebook Research SAM2 technology and enhanced color widget support across all nodes.
*   **v2.7.1** (2025/08/06): Enhanced `LoadImage` nodes, a redesigned `ImageStitch` node, and fixed background color issues.

*   **View All Updates**: See the [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md) file for a full changelog.

## Installation

Choose your preferred installation method:

*   **ComfyUI Manager:** Search for `Comfyui-RMBG` and install directly.  Install the `requirements.txt` file in the ComfyUI-RMBG folder.
    ```bash
    ./ComfyUI/python_embeded/python -m pip install -r requirements.txt
    ```
    >   **Note:** If you experience dependency issues, use ComfyUI's embedded Python.

*   **Manual Clone:**
    ```bash
    cd ComfyUI/custom_nodes
    git clone https://github.com/1038lab/ComfyUI-RMBG
    ```
    Install the `requirements.txt` file in the ComfyUI-RMBG folder.
    ```bash
    ./ComfyUI/python_embeded/python -m pip install -r requirements.txt
    ```

*   **Comfy CLI:**
    ```bash
    comfy node install ComfyUI-RMBG
    ```
    Install the `requirements.txt` file in the ComfyUI-RMBG folder.
    ```bash
    ./ComfyUI/python_embeded/python -m pip install -r requirements.txt
    ```

## Model Downloads

*   Models are automatically downloaded upon first use.
*   **Manual Download:**  Download models from the provided links (see below) and place them in the appropriate folders within `ComfyUI/models/RMBG/`, `ComfyUI/models/SAM`, `ComfyUI/models/sam2` and  `ComfyUI/models/grounding-dino`.

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
2.  Connect your image to the input.
3.  Select a model from the dropdown menu.
4.  Adjust the parameters as needed (see below).
5.  Get two outputs:
    *   IMAGE: Processed image with your chosen background.
    *   MASK: Foreground mask.

### Segment Node

1.  Load the `Segment (RMBG)` node from the `🧪AILab/🧽RMBG` category.
2.  Connect an image to the input.
3.  Enter your text prompt (tag-style or natural language).
4.  Select SAM and GroundingDINO models.
5.  Adjust parameters as needed.

### Parameters

*   `sensitivity`: Background removal sensitivity (0.0-1.0).
*   `process_res`: Processing resolution (512-2048, step 128).
*   `mask_blur`: Mask blur amount (0-64).
*   `mask_offset`: Adjust mask edges (-20 to 20).
*   `background`: Choose background color (Alpha, Black, White, Green, Blue, Red).
*   `invert_output`: Flip mask and image output.
*   `refine_foreground`: Use Fast Foreground Color Estimation to optimize transparent background.
*   `optimize`: Toggle model optimization.

###  Optional Settings :bulb: Tips

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

## Troubleshooting

*   **401 error with GroundingDINO / missing `models/sam2`:** Delete `%USERPROFILE%\.cache\huggingface\token` (and `%USERPROFILE%\.huggingface\token` if present) and ensure no `HF_TOKEN`/`HUGGINGFACE_TOKEN` env vars are set. Re-run.
*   **"Required input is missing: images":** Ensure image outputs are connected and upstream nodes ran successfully.

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

## License

GPL-3.0 License