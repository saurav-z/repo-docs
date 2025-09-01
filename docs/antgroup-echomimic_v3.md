# EchoMimicV3: Revolutionizing Human Animation with 1.3B Parameters

EchoMimicV3 offers a groundbreaking approach to unified multi-modal and multi-task human animation, all while leveraging a compact 1.3 billion parameter model.  [Explore the original repository on GitHub](https://github.com/antgroup/echomimic_v3).

**Key Features:**

*   **Unified Approach:**  Achieves multi-modal (audio, text) and multi-task (talking head, talking body) human animation within a single model.
*   **Compact Model Size:** Demonstrates state-of-the-art performance with only 1.3 billion parameters, making it efficient and accessible.
*   **Gradio Demo:** A user-friendly Gradio demo is available on [ModelScope](https://modelscope.cn/studios/BadToBest/EchoMimicV3) for easy experimentation.
*   **ComfyUI Support:**  Integrates with [ComfyUI](https://github.com/smthemex/ComfyUI_EchoMimic) for broader usage and customization options, requiring only 16GB VRAM.
*   **Model Availability:**  Pre-trained models are readily available on both [Hugging Face](https://huggingface.co/BadToBest/EchoMimicV3) and [ModelScope](https://modelscope.cn/models/BadToBest/EchoMimicV3).
*   **Open-Source:** Code is publicly available on GitHub, enabling community contributions and customization.
*   **Video Generation with Low VRAM:** Easily generate video with 12GB VRAM, according to the authors.

**Updates:**

*   **August 21, 2025:** Gradio demo released on ModelScope.
*   **August 12, 2025:**  12GB VRAM required for video generation.
*   **August 12, 2025:** ComfyUI integration released.
*   **August 9, 2025:** Models released on ModelScope.
*   **August 8, 2025:** Code released on GitHub and models on Hugging Face.
*   **July 8, 2025:** Paper released on arXiv ([https://arxiv.org/abs/2507.03905](https://arxiv.org/abs/2507.03905)).

**Gallery:**

[Include images and videos as in the original README]

**Quick Start:**

*   **Environment Setup:**
    *   Tested on Centos 7.2/Ubuntu 22.04, Cuda >= 12.1
    *   Compatible with A100 (80G), RTX 4090D (24G), and V100 (16G) GPUs.
    *   Requires Python 3.10 or 3.11.

*   **Installation:**
    *   **Windows:** Use the [one-click installation package](https://pan.baidu.com/share/init?surl=cV7i2V0wF4exDtKjJrAUeA).
    *   **Linux:**
        ```bash
        conda create -n echomimic_v3 python=3.10
        conda activate echomimic_v3
        pip install -r requirements.txt
        ```

*   **Model Preparation:**
    *   Download the required models (links provided in original README).  Place the weights as described in the original document.

*   **Quick Inference:**
    ```bash
    python infer.py
    ```
    * For Quantified GradioUI version:
    ```bash
    python app_mm.py
    ```
    *   **Datasets:** Images, audios, masks, and prompts are available in `datasets/echomimicv3_demos`.

**Tips:**

*   **Audio CFG:** Audio CFG `audio_guidance_scale` works optimally between 2~3. Increase for better lip synchronization, decrease for better visual quality.
*   **Text CFG:** Text CFG `guidance_scale` works optimally between 3~6. Increase for better prompt following, decrease for better visual quality.
*   **TeaCache:**  The optimal range for `teacache_threshold` is between 0~0.1.
*   **Sampling Steps:** Use 5 steps for talking head, 15~25 steps for talking body.
*   **Long Video Generation:** Use Long Video CFG for videos longer than 138 frames.
*   **VRAM Optimization:** Try setting `partial_video_length` to 81, 65 or smaller to reduce VRAM usage.

**TODO List:**

[Include the TODO list from the original README]

**EchoMimic Series:**

*   EchoMimicV3:  [GitHub](https://github.com/antgroup/echomimic_v3)
*   EchoMimicV2:  [GitHub](https://github.com/antgroup/echomimic_v2)
*   EchoMimicV1:  [GitHub](https://github.com/antgroup/echomimic)

**Citation:**

```
@misc{meng2025echomimicv3,
  title={EchoMimicV3: 1.3B Parameters are All You Need for Unified Multi-Modal and Multi-Task Human Animation},
  author={Rang Meng, Yan Wang, Weipeng Wu, Ruobing Zheng, Yuming Li, Chenguang Ma},
  year={2025},
  eprint={2507.03905},
  archivePrefix={arXiv}
}
```

**References:**

[Include references from the original README]

**License:**

Licensed under the Apache 2.0 License.

**Star History:**

[Include Star History Chart as in the original README]