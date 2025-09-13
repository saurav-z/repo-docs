# EchoMimicV3: Animate Humans with Just 1.3 Billion Parameters!

EchoMimicV3 revolutionizes human animation, offering unified multi-modal and multi-task capabilities with a compact 1.3 billion parameter model. Check out the original repository on [GitHub](https://github.com/antgroup/echomimic_v3) for the latest updates and resources.

## Key Features:

*   **Unified Multi-Modal Animation:** Supports various inputs, including audio and text, for versatile animation.
*   **Multi-Task Capabilities:** Performs diverse animation tasks with a single model.
*   **Compact Model Size:** Achieves impressive results with only 1.3B parameters, making it more accessible and efficient.
*   **Gradio Demo:** Experience EchoMimicV3 through a user-friendly Gradio interface on [ModelScope](https://modelscope.cn/studios/BadToBest/EchoMimicV3).
*   **ComfyUI Integration:**  Run EchoMimicV3 on 16G VRAM using [ComfyUI](https://github.com/smthemex/ComfyUI_EchoMimic).

## What's New:

*   **[2025.08.21]** 🔥 Gradio demo on [modelscope](https://modelscope.cn/studios/BadToBest/EchoMimicV3) is ready.
*   **[2025.08.12]** 🔥🚀 **12G VRAM is All YOU NEED to Generate Video**.  Check the [tutorial](https://www.bilibili.com/video/BV1W8tdzEEVN). Thanks for the contribution.
*   **[2025.08.12]** 🔥 EchoMimicV3 can run on **16G VRAM** using [ComfyUI](https://github.com/smthemex/ComfyUI_EchoMimic). Thanks @[smthemex](https://github.com/smthemex) for the contribution.
*   **[2025.08.09]** 🔥 Released [models](https://modelscope.cn/models/BadToBest/EchoMimicV3) on ModelScope.
*   **[2025.08.08]** 🔥 Released [codes](https://github.com/antgroup/echomimic_v3) on GitHub and [models](https://huggingface.co/BadToBest/EchoMimicV3) on Huggingface.
*   **[2025.07.08]** 🔥 [Paper](https://arxiv.org/abs/2507.03905) is available on arXiv.

## Gallery:

[Include a compelling visual gallery of animated examples here. Consider using a grid layout.]

## Quick Start:

### Environment Setup

*   **Tested System:** Centos 7.2/Ubuntu 22.04, Cuda >= 12.1
*   **Tested GPUs:** A100(80G) / RTX4090D (24G) / V100(16G)
*   **Tested Python Version:** 3.10 / 3.11

### Installation

#### For Windows:

*   Use the [one-click installation package](https://pan.baidu.com/share/init?surl=cV7i2V0wF4exDtKjJrAUeA) (passport: glut) for a streamlined setup.

#### For Linux:

1.  **Create a Conda environment:**

    ```bash
    conda create -n echomimic_v3 python=3.10
    conda activate echomimic_v3
    ```

2.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

### Model Preparation

[Include a table with download links for the necessary models, including Hugging Face and ModelScope links. Specify where to place the model files in the directory structure.]

```
./models/
├── Wan2.1-Fun-V1.1-1.3B-InP
├── wav2vec2-base-960h
└── transformer
    └── diffusion_pytorch_model.safetensors
```

### Quick Inference

```bash
python infer.py
```

For Quantified GradioUI version:

```bash
python app_mm.py
```

**Sample data** is provided in `datasets/echomimicv3_demos`.

#### Tips
> - Audio CFG: Audio CFG `audio_guidance_scale` works optimally between 2~3. Increase the audio CFG value for better lip synchronization, while decreasing the audio CFG value can improve the visual quality.
> - Text CFG: Text CFG `guidance_scale` works optimally between 3~6. Increase the text CFG value for better prompt following, while decreasing the text CFG value can improve the visual quality.
> - TeaCache: The optimal range for `teacache_threshold` is between 0~0.1.
> - Sampling steps: 5 steps for talking head, 15~25 steps for talking body. 
> - ​Long video generation: If you want to generate a video longer than 138 frames, you can use Long Video CFG.
> - Try setting `partial_video_length` to 81, 65 or smaller to reduce VRAM usage.

## TODO List:

[A concise table showing the project's milestones and their status (e.g., ✅ for completed, 🚀 for in progress).]

## EchoMimic Series:

*   **EchoMimicV3:** (This Repository)
*   [EchoMimicV2: Towards Striking, Simplified, and Semi-Body Human Animation.](https://github.com/antgroup/echomimic_v2)
*   [EchoMimicV1: Lifelike Audio-Driven Portrait Animations through Editable Landmark Conditioning.](https://github.com/antgroup/echomimic)

## Citation:

```
@misc{meng2025echomimicv3,
  title={EchoMimicV3: 1.3B Parameters are All You Need for Unified Multi-Modal and Multi-Task Human Animation},
  author={Rang Meng, Yan Wang, Weipeng Wu, Ruobing Zheng, Yuming Li, Chenguang Ma},
  year={2025},
  eprint={2507.03905},
  archivePrefix={arXiv}
}
```

## References:

*   Wan2.1: [https://github.com/Wan-Video/Wan2.1/](https://github.com/Wan-Video/Wan2.1/)
*   VideoX-Fun: [https://github.com/aigc-apps/VideoX-Fun/](https://github.com/aigc-apps/VideoX-Fun/)

## License:

This project is licensed under the Apache 2.0 License.  You are responsible for your use of the models and should adhere to the license's terms, including avoiding content that violates laws or harms others.

## Star History:

[![Star History Chart](https://api.star-history.com/svg?repos=antgroup/echomimic_v3&type=Date)](https://www.star-history.com/#antgroup/echomimic_v3&Date)