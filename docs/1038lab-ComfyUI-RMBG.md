# ComfyUI-RMBG: Effortlessly Remove Backgrounds and Segment Images

**Enhance your ComfyUI workflows with ComfyUI-RMBG, a powerful custom node for precise image background removal, object segmentation, and more!**  Explore the features of this versatile node by visiting the original repository: [1038lab/ComfyUI-RMBG](https://github.com/1038lab/ComfyUI-RMBG).

## Key Features

*   **Advanced Background Removal:** Utilize a range of models (RMBG-2.0, INSPYRENET, BEN, BEN2, BiRefNet) for accurate background removal, with options for different background colors and transparency.
*   **Text-Prompted Object Segmentation:**  Segment images by simply typing what you want to isolate using SAM and GroundingDINO models.
*   **SAM2 Segmentation:** Leverage the latest SAM2 models (Tiny/Small/Base+/Large) for text-prompted segmentation.
*   **Enhanced Edge Detection:**  Improved edge quality and detail preservation, even with complex images.
*   **Fashion and Clothes Segmentation**: Segment clothes and fashion elements based on text prompt
*   **Face segmentation**: Segment facial feature categories like skin, nose, eyes, and eyebrows.
*   **Versatile Background Options:** Choose from transparent (alpha), black, white, green, blue, or red backgrounds.
*   **Flexible Image Tools:**  Includes useful tools such as Image/Mask Converter, Mask Enhancer, Mask Combiner, Mask Extractor.
*   **Batch Processing Support:** Efficiently process multiple images at once.
*   **Easy Installation:** Install via ComfyUI Manager or by cloning the repository.

## What's New

*   **v2.9.1 (2025/09/12):** [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v291-20250912)
*   **v2.9.0 (2025/08/18):** [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v290-20250818)  Added `SDMatte Matting` node
*   **v2.8.0 (2025/08/11):** [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v280-20250811) Added `SAM2Segment` and enhanced color widget support.
*   **v2.7.1 (2025/08/06):** [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v271-20250806) Enhanced `LoadImage` nodes and redesigned `ImageStitch` node.
*   **v2.6.0 (2025/07/15):** [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v260-20250715) Added `Kontext Refence latent Mask` node.
*   **v2.5.2 (2025/07/11):** [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v252-20250711)
*   **v2.5.1 (2025/07/07):** [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v251-20250707)
*   **v2.5.0 (2025/07/01):** [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v250-20250701) Added `MaskOverlay`, `ObjectRemover`, `ImageMaskResize` and new BiRefNet models.
*   **v2.4.0 (2025/06/01):** [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v240-20250601) Added `CropObject`, `ImageCompare`, `ColorInput` nodes and new Segment V2.

*(See the [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md) for more detailed change logs)*

## Installation

### Method 1: Using ComfyUI Manager

1.  Open ComfyUI Manager.
2.  Search for "Comfyui-RMBG" and install.
3.  Install `requirements.txt` (see below).

### Method 2: Manual Installation

1.  Navigate to your ComfyUI `custom_nodes` directory:

    ```bash
    cd ComfyUI/custom_nodes
    ```
2.  Clone the repository:

    ```bash
    git clone https://github.com/1038lab/ComfyUI-RMBG
    ```
3.  Install the required Python packages:

    ```bash
    ./ComfyUI/python_embeded/python -m pip install -r requirements.txt
    ```
    or
    ```bash
    python -m pip install -r requirements.txt
    ```
    *if you are in a correct python environment.
    *if the python environment is not set correctly please refer to ComfyUI official doc.

### Method 3: Using Comfy CLI

1.  Make sure you have `comfy-cli` installed:
    ```bash
    pip install comfy-cli
    ```

2.  Install ComfyUI if you don't already have it:
    ```bash
    comfy install
    ```
3.  Install ComfyUI-RMBG:
    ```bash
    comfy node install ComfyUI-RMBG
    ```
4.  Install requirements:

    ```bash
    ./ComfyUI/python_embeded/python -m pip install -r requirements.txt
    ```
    or
    ```bash
    python -m pip install -r requirements.txt
    ```
    *if you are in a correct python environment.
    *if the python environment is not set correctly please refer to ComfyUI official doc.

### Model Downloads

*   Models are automatically downloaded to `ComfyUI/models/RMBG/` and `ComfyUI/models/SAM/` on first use, or can be downloaded manually and placed in the corresponding folders.  See original README for specific links to download the models.

## Usage

### RMBG Node

1.  Load the `RMBG (Remove Background)` node from the `🧪AILab/🧽RMBG` category.
2.  Connect an image to the input.
3.  Select a model from the dropdown menu.
4.  Adjust optional parameters (see below).
5.  Get two outputs: the processed image and a foreground mask.

### Optional Settings:

| Parameter               | Description                                                              | Tips                                                                                                                                                                             |
| :---------------------- | :----------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Sensitivity             | Adjusts mask detection strength. Higher values mean stricter detection. | Default: 0.5. Adjust based on image complexity.                                                                                                                                |
| Processing Resolution   | Affects detail and memory usage.                                       | Choose between 256 and 2048 (default: 1024). Higher resolutions provide better detail but increase memory consumption.                                                            |
| Mask Blur               | Smooths mask edges.                                                       | Default: 0. Set between 1 and 5 for smoother edges.                                                                                                                               |
| Mask Offset             | Expands or shrinks the mask boundary.                                   | Default: 0. Fine-tune between -10 and 10.                                                                                                                                        |
| Background              | Choose the output background color                                          | Alpha(transparent), Black, White, Green, Blue, Red                                                                                                                                  |
| Invert Output           | Flip mask and image output                                               |                                                                                                                                                                                |
| Refine Foreground       | Use Fast Foreground Color Estimation to optimize transparent background | Enable for better edge quality and transparency handling                                                                                                                          |
| Performance Optimization| Properly setting options can enhance performance when processing multiple images. | If memory allows, consider increasing `process_res` and `mask_blur` values for better results, but be mindful of memory usage.                                          |

### Segment Node

1.  Load the `Segment (RMBG)` node from the `🧪AILab/🧽RMBG` category.
2.  Connect an image to the input.
3.  Enter a text prompt (tag-style or natural language).
4.  Select SAM or GroundingDINO models.
5.  Adjust parameters as needed:
    *   Threshold: 0.25-0.35 for broad detection, 0.45-0.55 for precision.
    *   Mask blur and offset for edge refinement.
    *   Background color options.

## Troubleshooting

*   **401 error** when initializing GroundingDINO/missing `models/sam2`: Delete `%USERPROFILE%\.cache\huggingface\token` and `%USERPROFILE%\.huggingface\token`.  Ensure no `HF_TOKEN`/`HUGGINGFACE_TOKEN` env vars are set. Re-run (public repos download anonymously).
*   **Preview shows "Required input is missing: images"**: Ensure image outputs are connected and upstream nodes ran successfully.

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

**If you find this custom node helpful, please give it a ⭐ on this repository!**

## License

GPL-3.0 License