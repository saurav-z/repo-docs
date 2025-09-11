<!-- Improved & Summarized README for EchoMimicV3 -->

# EchoMimicV3: Revolutionizing Human Animation with Just 1.3 Billion Parameters

>  EchoMimicV3 is a groundbreaking model that uses only 1.3 billion parameters to achieve unified multi-modal and multi-task human animation, opening new possibilities in the field. Explore the original project on GitHub: [https://github.com/antgroup/echomimic_v3](https://github.com/antgroup/echomimic_v3).

<p align="center">
  <img src="asset/EchoMimicV3_logo.png.jpg"  height=60>
</p>

<div align='center'>
    <a href='https://github.com/mengrang' target='_blank'>Rang Meng</a><sup>1</sup>&emsp;
    <a href='https://github.com/' target='_blank'>Yan Wang</a>&emsp;
    <a href='https://github.com/' target='_blank'>Weipeng Wu</a>&emsp;
    <a href='https://github.com/' target='_blank'>Ruobing Zheng</a>&emsp;
    <a href='https://lymhust.github.io/' target='_blank'>Yuming Li</a><sup>2</sup>&emsp;
    <a href='https://openreview.net/profile?id=~Chenguang_Ma3' target='_blank'>Chenguang Ma</a><sup>2</sup>
</div>
<div align='center'>
Terminal Technology Department, Alipay, Ant Group.
</div>
<p align='center'>
    <sup>1</sup>Core Contributor&emsp;
    <sup>2</sup>Corresponding Authors
</p>
<div align='center'>
    <a href='https://github.com/antgroup/echomimic_v3'><img src='https://img.shields.io/github/stars/antgroup/echomimic_v3?style=social'></a>
    <a href='https://antgroup.github.io/ai/echomimic_v3/'><img src='https://img.shields.io/badge/Project-Page-blue'></a>
    <a href='https://arxiv.org/abs/2507.03905'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
    <a href='https://huggingface.co/BadToBest/EchoMimicV3'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-yellow'></a>
    <a href='https://modelscope.cn/models/BadToBest/EchoMimicV3'><img src='https://img.shields.io/badge/ModelScope-Model-purple'></a>
    <a href='https://github.com/antgroup/echomimic_v3/blob/main/asset/wechat_group.png'><img src='https://badges.aleen42.com/src/wechat.svg'></a>
    <a href='https://github.com/antgroup/echomimic_v3/discussions/18'><img src='https://img.shields.io/badge/中文版-常见问题汇总-orange'></a>
</div>

<p align="center">
  <img src="asset/algo_framework.jpg"  height=700>
</p>


## ✨ Key Features

*   **Unified Multi-Modal & Multi-Task:**  Handles diverse animation tasks from audio to visual output.
*   **Efficient Parameter Usage:** Achieves high-quality results with only 1.3B parameters.
*   **Low VRAM Requirement:**  Can run on 12G VRAM and 16G VRAM with ComfyUI.
*   **Easy-to-Use Demos:** Gradio and ComfyUI demo interfaces available for quick experimentation.
*   **Model Availability:** Models available on Hugging Face and ModelScope.

## 📣 Updates

*   **[2025.08.21]**  Gradio demo on ModelScope is ready: [modelscope](https://modelscope.cn/studios/BadToBest/EchoMimicV3)
*   **[2025.08.12]**  Generate video with just **12G VRAM**! Use the [GradioUI](https://github.com/antgroup/echomimic_v3/blob/main/app_mm.py) or [ComfyUI](https://github.com/smthemex/ComfyUI_EchoMimic). 
*   **[2025.08.09]** Models released on ModelScope.
*   **[2025.08.08]** Codes and Models released on GitHub and Huggingface.
*   **[2025.07.08]** Paper released on Arxiv.

## 🎬 Gallery

<!-- Include video and image examples showcasing EchoMimicV3 capabilities -->

<p align="center">
  <img src="asset/echomimicv3.jpg"  height=1000>
</p>

<table class="center">
<tr>
    <td width=100% style="border: none">
        <video controls loop src="https://github.com/user-attachments/assets/f33edb30-66b1-484b-8be0-a5df20a44f3b" muted="false"></video>
    </td>
</tr>
<tr>
    <td width=100% style="border: none">
        <video controls loop src="https://github.com/user-attachments/assets/056105d8-47cd-4a78-8ec2-328ceaf95a5a" muted="false"></video>
    </td>
</tr>
</table>

### Chinese Driven Audio
<table class="center">
<tr>
    <td width=25% style="border: none">
        <video controls loop src="https://github.com/user-attachments/assets/fc1ebae4-b571-43eb-a13a-7d6d05b74082" muted="false"></video>
    </td>
    <td width=25% style="border: none">
        <video controls loop src="https://github.com/user-attachments/assets/54607cc7-944c-4529-9bef-715862ba330d" muted="false"></video>
    </td>
    <td width=25% style="border: none">
        <video controls loop src="https://github.com/user-attachments/assets/4d1de999-cce2-47ab-89ed-f2fa11c838fe" muted="false"></video>
    </td>
    <td width=25% style="border: none">
        <video controls loop src="https://github.com/user-attachments/assets/41e701cc-ac3e-4dd8-b94c-859261f17344" muted="false"></video>
    </td>
</tr>
</table>

For more demo videos, please refer to the [project page](https://antgroup.github.io/ai/echomimic_v3/)

## 🚀 Quick Start

### Environment Setup

*   **OS:** Centos 7.2/Ubuntu 22.04
*   **CUDA:**  \>= 12.1
*   **GPUs:** A100(80G) / RTX4090D (24G) / V100(16G)
*   **Python:** 3.10 / 3.11

### 🛠️ Installation

#### Windows (Quantified Version)

*   Use the [one-click installation package](https://pan.baidu.com/share/init?surl=cV7i2V0wF4exDtKjJrAUeA) (passport: glut)

#### Linux

1.  **Create Conda Environment:**
    ```bash
    conda create -n echomimic_v3 python=3.10
    conda activate echomimic_v3
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### 🧱 Model Preparation

| Model                             | Download Link                                                               | Notes                     |
| :-------------------------------- | :-------------------------------------------------------------------------- | :------------------------ |
| Wan2.1-Fun-V1.1-1.3B-InP          | 🤗 [Huggingface](https://huggingface.co/alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP) | Base model                |
| wav2vec2-base                     | 🤗 [Huggingface](https://huggingface.co/facebook/wav2vec2-base-960h)          | Audio encoder             |
| EchoMimicV3-preview               | 🤗 [Huggingface](https://huggingface.co/BadToBest/EchoMimicV3)              | EchoMimicV3 Weights       |
| EchoMimicV3-preview               | 🤗 [ModelScope](https://modelscope.cn/models/BadToBest/EchoMimicV3)              | EchoMimicV3 Weights       |

-- **Weights Organization:**

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

For Quantified GradioUI:

```bash
python app_mm.py
```

**Sample Data:**  Images, audios, masks, and prompts are available in `datasets/echomimicv3_demos`.

#### Tips
> - **Audio CFG:**  `audio_guidance_scale` (2~3) for lip sync.  Increase for lip-sync, decrease for quality.
> - **Text CFG:**  `guidance_scale` (3~6) for prompt following. Increase for prompt following, decrease for quality.
> - **TeaCache:**  `teacache_threshold` (0~0.1)
> - **Sampling steps:** 5 steps for talking head, 15~25 steps for talking body.
> - **Long video generation:** Use Long Video CFG for videos longer than 138 frames.
> - Set `partial_video_length` to 81, 65 or smaller to reduce VRAM usage.

## 📝 TODO List

| Status | Milestone                                                               |
| :----: | :---------------------------------------------------------------------- |
|   ✅   | The inference code of EchoMimicV3 on GitHub                        |
|   ✅   | EchoMimicV3-preview model on HuggingFace                   |
|   ✅   | EchoMimicV3-preview model on ModelScope                   |
|   ✅   | ModelScope Space                   |
|   🚀   | 720P Pretrained models                                       |
|   🚀   | The training code of EchoMimicV3 on GitHub                       |

## &#x1F680; EchoMimic Series

*   EchoMimicV3: 1.3B Parameters are All You Need for Unified Multi-Modal and Multi-Task Human Animation. [GitHub](https://github.com/antgroup/echomimic_v3)
*   EchoMimicV2: Towards Striking, Simplified, and Semi-Body Human Animation. [GitHub](https://github.com/antgroup/echomimic_v2)
*   EchoMimicV1: Lifelike Audio-Driven Portrait Animations through Editable Landmark Conditioning. [GitHub](https://github.com/antgroup/echomimic)

## &#x1F4D2; Citation

```bibtex
@misc{meng2025echomimicv3,
  title={EchoMimicV3: 1.3B Parameters are All You Need for Unified Multi-Modal and Multi-Task Human Animation},
  author={Rang Meng, Yan Wang, Weipeng Wu, Ruobing Zheng, Yuming Li, Chenguang Ma},
  year={2025},
  eprint={2507.03905},
  archivePrefix={arXiv}
}
```

## 📜 License

The models in this repository are licensed under the Apache 2.0 License.  You are responsible for your use of the models.

## &#x1F31F; Star History

[![Star History Chart](https://api.star-history.com/svg?repos=antgroup/echomimic_v3&type=Date)](https://www.star-history.com/#antgroup/echomimic_v3&Date)