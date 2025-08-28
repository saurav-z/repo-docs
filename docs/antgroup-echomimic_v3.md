# EchoMimicV3: Unleash Human Animation with 1.3B Parameters

EchoMimicV3 revolutionizes human animation by offering unified multi-modal and multi-task capabilities with a compact 1.3 billion parameter model, making it easier than ever to bring your ideas to life. Explore the original repository for further information: [EchoMimicV3 on GitHub](https://github.com/antgroup/echomimic_v3)

## Key Features

*   **Unified Multi-Modal:** Supports various input modalities like audio, text, and images for versatile animation.
*   **Multi-Task Capabilities:** Performs various animation tasks, including talking head and body animation.
*   **Compact Model Size:** Achieves impressive results with only 1.3B parameters, enabling efficient deployment and faster generation.
*   **Gradio Demo:**  Interactive demo available on [ModelScope](https://modelscope.cn/studios/BadToBest/EchoMimicV3) for easy experimentation.
*   **ComfyUI Support:**  Integrates with [ComfyUI](https://github.com/smthemex/ComfyUI_EchoMimic) for advanced users.
*   **Low VRAM Usage:** Generates videos with as little as 12GB VRAM using the GradioUI.

## What's New

*   **[2025.08.21]** 🔥 EchoMimicV3 Gradio demo on [ModelScope](https://modelscope.cn/studios/BadToBest/EchoMimicV3).
*   **[2025.08.12]** 🔥🚀 **12G VRAM is All YOU NEED to Generate Video**.  Use the provided [GradioUI](https://github.com/antgroup/echomimic_v3/blob/main/app_mm.py) and check out the [tutorial](https://www.bilibili.com/video/BV1W8tdzEEVN).
*   **[2025.08.12]** 🔥 EchoMimicV3 compatible with **16G VRAM** via [ComfyUI](https://github.com/smthemex/ComfyUI_EchoMimic).
*   **[2025.08.09]** 🔥 Models released on [ModelScope](https://modelscope.cn/models/BadToBest/EchoMimicV3).
*   **[2025.08.08]** 🔥 Codes released on GitHub and models released on [Huggingface](https://huggingface.co/BadToBest/EchoMimicV3).
*   **[2025.07.08]** 🔥 Paper released on arxiv [https://arxiv.org/abs/2507.03905](https://arxiv.org/abs/2507.03905).

## Gallery

[Include the images and videos from the original README, formatted appropriately for markdown, keeping in mind the limitations of markdown in terms of video playback within the text.]

For more demo videos, please refer to the [project page](https://antgroup.github.io/ai/echomimic_v3/)

## Quick Start

### Environment Setup

*   **Operating System:** Centos 7.2/Ubuntu 22.04
*   **CUDA:** >= 12.1
*   **GPUs:** A100(80G) / RTX4090D (24G) / V100(16G)
*   **Python:** 3.10 / 3.11

### Installation

#### For Windows

##### Get started quickly using the [one-click installation package](https://pan.baidu.com/share/init?surl=cV7i2V0wF4exDtKjJrAUeA) .

#### For Linux

1.  **Create a Conda Environment:**

    ```bash
    conda create -n echomimic_v3 python=3.10
    conda activate echomimic_v3
    ```

2.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

### Model Preparation

| Model                                  | Download Link                                           | Notes                      |
| :------------------------------------- | :------------------------------------------------------ | :------------------------- |
| Wan2.1-Fun-V1.1-1.3B-InP                | [Huggingface](https://huggingface.co/alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP) | Base model                 |
| wav2vec2-base                          | [Huggingface](https://huggingface.co/facebook/wav2vec2-base-960h) | Audio encoder              |
| EchoMimicV3-preview                    | [Huggingface](https://huggingface.co/BadToBest/EchoMimicV3)   | Our weights                |
| EchoMimicV3-preview                    | [ModelScope](https://modelscope.cn/models/BadToBest/EchoMimicV3)    | Our weights                |

-- **Weights Directory Structure:**

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

For the Quantified GradioUI version:

```bash
python app_mm.py
```

**Images, audios, masks, and prompts are provided in `datasets/echomimicv3_demos`**

#### Tips

*   **Audio CFG:** Optimal `audio_guidance_scale`: 2~3.  Increase for better lip sync, decrease for better visual quality.
*   **Text CFG:** Optimal `guidance_scale`: 3~6. Increase for better prompt following, decrease for better visual quality.
*   **TeaCache:** Optimal `teacache_threshold`: 0~0.1.
*   **Sampling Steps:** 5 steps for talking head, 15~25 steps for talking body.
*   **Long Video Generation:** Use Long Video CFG for videos longer than 138 frames.
*   **Reduce VRAM Usage:** Set `partial_video_length` to 81, 65, or smaller.

## Roadmap

| Status   | Milestone                                                                |
| :------- | :------------------------------------------------------------------------- |
| ✅        | Inference code on GitHub                                                    |
| ✅        | EchoMimicV3-preview model on HuggingFace                                  |
| ✅        | EchoMimicV3-preview model on ModelScope                                   |
| 🚀        | ModelScope Space                                                          |
| 🚀        | Preview version Pretrained models trained on English and Chinese on ModelScope |
| 🚀        | 720P Pretrained models trained on English and Chinese on HuggingFace       |
| 🚀        | 720P Pretrained models trained on English and Chinese on ModelScope        |
| 🚀        | Training code on GitHub                                                    |

## EchoMimic Series

*   **EchoMimicV3:** 1.3B Parameters for Unified Multi-Modal and Multi-Task Human Animation.
    [GitHub](https://github.com/antgroup/echomimic_v3)
*   **EchoMimicV2:** Towards Striking, Simplified, and Semi-Body Human Animation.
    [GitHub](https://github.com/antgroup/echomimic_v2)
*   **EchoMimicV1:** Lifelike Audio-Driven Portrait Animations through Editable Landmark Conditioning.
    [GitHub](https://github.com/antgroup/echomimic)

## Citation

```
@misc{meng2025echomimicv3,
  title={EchoMimicV3: 1.3B Parameters are All You Need for Unified Multi-Modal and Multi-Task Human Animation},
  author={Rang Meng, Yan Wang, Weipeng Wu, Ruobing Zheng, Yuming Li, Chenguang Ma},
  year={2025},
  eprint={2507.03905},
  archivePrefix={arXiv}
}
```

## Reference

*   Wan2.1: [https://github.com/Wan-Video/Wan2.1/](https://github.com/Wan-Video/Wan2.1/)
*   VideoX-Fun: [https://github.com/aigc-apps/VideoX-Fun/](https://github.com/aigc-apps/VideoX-Fun/)

## License

The models in this repository are licensed under the Apache 2.0 License.