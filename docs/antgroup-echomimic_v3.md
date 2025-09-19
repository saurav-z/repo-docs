# EchoMimicV3: Revolutionizing Human Animation with a 1.3B Parameter Model

**Unleash lifelike, multi-modal human animation with EchoMimicV3, a cutting-edge model from Ant Group.  Check out the original repository [here](https://github.com/antgroup/echomimic_v3).**

## Key Features:

*   **Unified Approach:** Seamlessly handles multi-modal (audio, text) and multi-task (talking head, talking body) human animation.
*   **Efficiency:** Achieves impressive results with only 1.3 billion parameters.
*   **Simplified Generation:**  Generate video on as low as 12GB VRAM with the provided Gradio UI, or utilize ComfyUI for 16GB VRAM.
*   **Model Availability:** Access pre-trained models on Hugging Face and ModelScope.
*   **Easy to Use:** Quick start guide and sample datasets provided for immediate inference.
*   **Active Community:**  Join the discussion through the [GitHub Discussions](https://github.com/antgroup/echomimic_v3/discussions)

## What's New?

*   **[2025.08.21]** ModelScope demo is ready for you!
*   **[2025.08.12]** Generate video with 12G VRAM using [GradioUI](https://github.com/antgroup/echomimic_v3/blob/main/app_mm.py) Check the [tutorial](https://www.bilibili.com/video/BV1W8tdzEEVN). Thanks @[gluttony-10]
*   **[2025.08.12]** Can run on 16G VRAM using [ComfyUI](https://github.com/smthemex/ComfyUI_EchoMimic). Thanks @[smthemex](https://github.com/smthemex)
*   **[2025.08.09]** Models released on [ModelScope](https://modelscope.cn/models/BadToBest/EchoMimicV3).
*   **[2025.08.08]** Code and models released on [GitHub](https://github.com/antgroup/echomimic_v3) and [Huggingface](https://huggingface.co/BadToBest/EchoMimicV3).
*   **[2025.07.08]** Paper released on [arXiv](https://arxiv.org/abs/2507.03905).

## Demo Videos

[Include a gallery of impressive demo videos here.  Use thumbnails or small embedded videos with descriptive captions.  Example:]

**Talking Head Demo:**  *A short video demonstrating the model animating a face based on audio input.*

<video controls loop src="https://github.com/user-attachments/assets/f33edb30-66b1-484b-8be0-a5df20a44f3b" muted="false"></video>

**Talking Body Demo:** *Showcasing the model's ability to animate the full body based on audio.*

<video controls loop src="https://github.com/user-attachments/assets/056105d8-47cd-4a78-8ec2-328ceaf95a5a" muted="false"></video>

[Include more demo videos here, potentially for "Chinese Driven Audio", or other use cases.]

## Quick Start Guide

### Environment Setup

*   **Operating Systems:**  Centos 7.2/Ubuntu 22.04.
*   **CUDA:**  CUDA >= 12.1.
*   **GPUs:** Tested on A100(80G), RTX4090D (24G), V100(16G).
*   **Python:**  Python 3.10 / 3.11

### 🛠️ Installation
**Windows**: Use the [one-click installation package](https://pan.baidu.com/share/init?surl=cV7i2V0wF4exDtKjJrAUeA) (passport: glut).

**Linux:**

1.  **Create a Conda Environment:**
    ```bash
    conda create -n echomimic_v3 python=3.10
    conda activate echomimic_v3
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### 🧱 Model Preparation

| Model                                 | Download Link                                                                                                | Notes                     |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------ | ------------------------- |
| Wan2.1-Fun-V1.1-1.3B-InP              | [Huggingface](https://huggingface.co/alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP)                                  | Base model                |
| wav2vec2-base                         | [Huggingface](https://huggingface.co/facebook/wav2vec2-base-960h)                                           | Audio encoder             |
| EchoMimicV3-preview                   | [Huggingface](https://huggingface.co/BadToBest/EchoMimicV3)                                                  | Our weights               |
| EchoMimicV3-preview                   | [ModelScope](https://modelscope.cn/models/BadToBest/EchoMimicV3)                                                  | Our weights               |

-- **Weights Directory Structure:**

```
./models/
├── Wan2.1-Fun-V1.1-1.3B-InP
├── wav2vec2-base-960h
└── transformer
    └── diffusion_pytorch_model.safetensors
```

### 🔑 Quick Inference

```bash
python infer.py
```

For Quantified GradioUI version:

```bash
python app_mm.py
```

**Datasets:**  Images, audio, masks and prompts are available in `datasets/echomimicv3_demos`.

#### Tips:

*   **Audio CFG:**  `audio_guidance_scale` (2~3) for optimal lip sync and visual quality.
*   **Text CFG:**  `guidance_scale` (3~6) for prompt following and visual quality.
*   **TeaCache:** `teacache_threshold` (0~0.1)
*   **Sampling Steps:**  5 steps for talking heads, 15~25 steps for talking bodies.
*   **Long Video Generation:** Use Long Video CFG.
*   **Reduce VRAM Usage:** Set `partial_video_length` to 81, 65, or smaller.

## 📝 TODO List

| Status   | Milestone                                       |
| :------- | :---------------------------------------------- |
| ✅        | Inference Code on GitHub                     |
| ✅        | EchoMimicV3-preview Model on Hugging Face      |
| ✅        | EchoMimicV3-preview Model on ModelScope      |
| ✅        | ModelScope Space                             |
| 🚀        | 720P Pretrained models                        |
| 🚀        | Training Code on GitHub                       |

##  &#x1F680; EchoMimic Series

*   **EchoMimicV3:** 1.3B Parameters are All You Need for Unified Multi-Modal and Multi-Task Human Animation. [GitHub](https://github.com/antgroup/echomimic_v3)
*   **EchoMimicV2:** Towards Striking, Simplified, and Semi-Body Human Animation. [GitHub](https://github.com/antgroup/echomimic_v2)
*   **EchoMimicV1:** Lifelike Audio-Driven Portrait Animations through Editable Landmark Conditioning. [GitHub](https://github.com/antgroup/echomimic)

## &#x1F4D2; Citation

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
- Wan2.1: https://github.com/Wan-Video/Wan2.1/
- VideoX-Fun: https://github.com/aigc-apps/VideoX-Fun/

## 📜 License
This project is licensed under the Apache 2.0 License.

## &#x1F31F; Star History
[![Star History Chart](https://api.star-history.com/svg?repos=antgroup/echomimic_v3&type=Date)](https://www.star-history.com/#antgroup/echomimic_v3&Date)