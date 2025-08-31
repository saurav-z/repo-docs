# EchoMimicV3: Revolutionizing Human Animation with 1.3B Parameters

EchoMimicV3 offers a cutting-edge solution for unified multi-modal and multi-task human animation, making realistic video generation accessible.  [Explore the EchoMimicV3 repository on GitHub](https://github.com/antgroup/echomimic_v3).

**Key Features:**

*   **Unified Multi-Modal Animation:** Seamlessly integrates various modalities, including audio and text, for versatile animation.
*   **Multi-Task Capabilities:**  Handles diverse animation tasks, from talking heads to full-body movements.
*   **Efficient Architecture:** Achieves high-quality results with only 1.3 billion parameters, making it efficient and accessible.
*   **Model Availability:** Pre-trained models available on both [Hugging Face](https://huggingface.co/BadToBest/EchoMimicV3) and [ModelScope](https://modelscope.cn/models/BadToBest/EchoMimicV3).
*   **Gradio Demo:** Interactive demo available on [ModelScope](https://modelscope.cn/studios/BadToBest/EchoMimicV3).
*   **Community Contributions:**  Support for ComfyUI and GradioUI through community contributions.

**What's New:**

*   **[2025.08.21]** 🔥 EchoMimicV3 Gradio demo on [ModelScope](https://modelscope.cn/studios/BadToBest/EchoMimicV3).
*   **[2025.08.12]** 🔥🚀 **12G VRAM is All YOU NEED to Generate Video**. Please use this [GradioUI](https://github.com/antgroup/echomimic_v3/blob/main/app_mm.py). Check the [tutorial](https://www.bilibili.com/video/BV1W8tdzEEVN) from @[gluttony-10](https://github.com/gluttony-10). Thanks for the contribution.
*   **[2025.08.12]** 🔥 EchoMimicV3 can run on **16G VRAM** using [ComfyUI](https://github.com/smthemex/ComfyUI_EchoMimic). Thanks @[smthemex](https://github.com/smthemex) for the contribution.
*   **[2025.08.09]** 🔥 Models released on [ModelScope](https://modelscope.cn/models/BadToBest/EchoMimicV3).
*   **[2025.08.08]** 🔥 Codes and models released on [GitHub](https://github.com/antgroup/echomimic_v3) and [HuggingFace](https://huggingface.co/BadToBest/EchoMimicV3).
*   **[2025.07.08]** 🔥 Paper released on arXiv ([https://arxiv.org/abs/2507.03905](https://arxiv.org/abs/2507.03905)).

**Gallery:**

[Include a well-captioned image or GIF showcasing EchoMimicV3's capabilities.  Consider a collage of the images/videos from the original README]

**Quick Start:**

1.  **Environment Setup:**
    *   Tested Systems: CentOS 7.2 / Ubuntu 22.04
    *   CUDA: >= 12.1
    *   GPUs: A100 (80G) / RTX4090D (24G) / V100 (16G)
    *   Python: 3.10 / 3.11

2.  **Installation (Windows):**

    *   Use the [one-click installation package](https://pan.baidu.com/share/init?surl=cV7i2V0wF4exDtKjJrAUeA) for a quick start (Quantified version).

3.  **Installation (Linux):**

    ```bash
    conda create -n echomimic_v3 python=3.10
    conda activate echomimic_v3
    pip install -r requirements.txt
    ```

4.  **Model Preparation:**

    | Model                             | Download Link                                                                     | Notes               |
    | --------------------------------- | --------------------------------------------------------------------------------- | ------------------- |
    | Wan2.1-Fun-V1.1-1.3B-InP          | [Huggingface](https://huggingface.co/alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP)          | Base model          |
    | wav2vec2-base                     | [Huggingface](https://huggingface.co/facebook/wav2vec2-base-960h)                | Audio encoder       |
    | EchoMimicV3-preview               | [Huggingface](https://huggingface.co/BadToBest/EchoMimicV3)                      | Our weights         |
    | EchoMimicV3-preview               | [ModelScope](https://modelscope.cn/models/BadToBest/EchoMimicV3)                  | Our weights         |

    *   Model weights are organized as follows:
        ```
        ./models/
        ├── Wan2.1-Fun-V1.1-1.3B-InP
        ├── wav2vec2-base-960h
        └── transformer
            └── diffusion_pytorch_model.safetensors
        ```

5.  **Quick Inference:**

    ```bash
    python infer.py
    ```

    For Quantified GradioUI version:

    ```bash
    python app_mm.py
    ```

    *   Sample data (images, audios, masks, prompts) are provided in the `datasets/echomimicv3_demos` directory.

    **Tips:**

    *   Audio CFG: `audio_guidance_scale` (2~3) for optimal lip sync.
    *   Text CFG: `guidance_scale` (3~6) for better prompt following.
    *   TeaCache: `teacache_threshold` (0~0.1).
    *   Sampling steps: 5 (talking head), 15-25 (talking body).
    *   Long video generation: Use Long Video CFG.
    *   Reduce VRAM:  Set `partial_video_length` to 81, 65, or lower.

**TODO List:**

| Status  | Milestone                                                                |
| :------ | :------------------------------------------------------------------------- |
| ✅      | Release inference code on GitHub                                             |
| ✅      | Release EchoMimicV3-preview model on Hugging Face                          |
| ✅      | Release EchoMimicV3-preview model on ModelScope                            |
| 🚀      | ModelScope Space                                                           |
| 🚀      | Preview version Pretrained models trained on English and Chinese on ModelScope |
| 🚀      | 720P Pretrained models trained on English and Chinese on HuggingFace        |
| 🚀      | 720P Pretrained models trained on English and Chinese on ModelScope          |
|  🚀     | Release training code on GitHub |

**EchoMimic Series:**

*   EchoMimicV3: 1.3B Parameters are All You Need for Unified Multi-Modal and Multi-Task Human Animation. [GitHub](https://github.com/antgroup/echomimic_v3)
*   EchoMimicV2: Towards Striking, Simplified, and Semi-Body Human Animation. [GitHub](https://github.com/antgroup/echomimic_v2)
*   EchoMimicV1: Lifelike Audio-Driven Portrait Animations through Editable Landmark Conditioning. [GitHub](https://github.com/antgroup/echomimic)

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

*   Wan2.1: [https://github.com/Wan-Video/Wan2.1/](https://github.com/Wan-Video/Wan2.1/)
*   VideoX-Fun: [https://github.com/aigc-apps/VideoX-Fun/](https://github.com/aigc-apps/VideoX-Fun/)

**License:**

EchoMimicV3 models are licensed under the Apache 2.0 License.  You are free to use the generated content while adhering to the license terms.  You are responsible for your use and must avoid any content that violates laws or harms others.

**Star History:**

[Include the Star History chart using the provided code]
```
[![Star History Chart](https://api.star-history.com/svg?repos=antgroup/echomimic_v3&type=Date)](https://www.star-history.com/#antgroup/echomimic_v3&Date)
```