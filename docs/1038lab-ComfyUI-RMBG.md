# ComfyUI-RMBG: Effortlessly Remove Backgrounds and Segment Images in ComfyUI

**Enhance your ComfyUI workflows with ComfyUI-RMBG, a powerful custom node for advanced image background removal, object segmentation, and more.** ([View on GitHub](https://github.com/1038lab/ComfyUI-RMBG))

**Key Features:**

*   **Advanced Background Removal:**
    *   Utilizes models like RMBG-2.0, INSPYRENET, BEN, and BEN2 for precise background removal.
    *   Offers flexible background options (transparent, black, white, green, blue, red).
    *   Batch processing support.
*   **Precise Object Segmentation:**
    *   Segment images using text prompts (tag-style or natural language).
    *   Supports SAM and GroundingDINO models for accurate object detection.
    *   Fine-tune results with adjustable parameters.
*   **SAM2 Segmentation:**
    *   Leverages the latest Facebook Research SAM2 models.
    *   Text-prompted segmentation with various model sizes.
    *   Automatic model download for ease of use.
*   **Comprehensive Model Support:** Includes RMBG-2.0, INSPYRENET, BEN, BEN2, BiRefNet (various versions), SAM, SAM2, GroundingDINO, and clothes/fashion segmentation.
*   **Enhanced Performance and Usability:** Features like real-time background replacement, edge detection, and batch processing optimize workflows.
*   **Regular Updates:** Stay up-to-date with the latest advancements through frequent updates and new model integrations.

**News & Updates:**

*   **v2.8.0 (2025/08/11):** Added `SAM2Segment` node with latest SAM2 technology and enhanced color widget support.
*   **v2.7.1 (2025/08/06):** Revamped LoadImage nodes, redesigned ImageStitch node, and background color handling fixes.
*   **v2.6.0 (2025/07/15):** Added `Kontext Refence latent Mask` node.
*   **(More updates available in the original README)**

**Installation:**

*   **ComfyUI Manager:** Search and install `Comfyui-RMBG`. Then install requirements.txt
*   **Clone:**
    ```bash
    cd ComfyUI/custom_nodes
    git clone https://github.com/1038lab/ComfyUI-RMBG
    ```
    Then install requirements.txt
*   **Comfy CLI:** `comfy node install ComfyUI-RMBG` . Then install requirements.txt
*   **Install Requirements:** (in the ComfyUI-RMBG folder)
    ```bash
    ./ComfyUI/python_embeded/python -m pip install -r requirements.txt
    ```

**Manual Model Download:**

*   Models are automatically downloaded upon first use.
*   Manual download instructions are available in the original README.

**Usage:**

*   **RMBG Node:** Remove backgrounds with various models.
*   **Segment Node:** Text-prompted object segmentation.

**Parameters:**

*   Sensitivity, Processing Resolution, Mask Blur, Mask Offset, Background, Invert Output, Refine Foreground, Performance Optimization.

**About Models**
(See the original README for detailed information on available models and their features).

**Requirements:**

*   ComfyUI
*   Python 3.10+
*   Required packages (automatically installed).

**Credits:**

*   [AILab](https://github.com/1038lab)

**Star History:**

[<img src="https://api.star-history.com/svg?repos=1038lab/comfyui-rmbg&type=Date" alt="Star History Chart"/>](https://www.star-history.com/#1038lab/comfyui-rmbg&Date)

**Give the repo a star if you find this custom node helpful!**

**License:**
* GPL-3.0 License
```
Key improvements and SEO optimization in this version:

*   **Concise Hook:** The first sentence acts as a clear and compelling hook.
*   **Clear Headings:** Uses well-defined headings for easy navigation.
*   **Keyword Optimization:** Includes relevant keywords (e.g., "ComfyUI," "background removal," "image segmentation," "object detection," "SAM," "GroundingDINO") throughout the text.
*   **Bulleted Lists:** Uses bullet points for key features, making them easy to read and digest.
*   **Concise Summaries:** Summarizes the update logs.
*   **Clear Installation Instructions:** Clear instructions with methods for ease of use.
*   **Model Details:** Summarized the model information.
*   **Call to Action:** Encourages users to give the repo a star.
*   **Structure:** improved overall readability.