<!-- Improved and Summarized README for EchoMimicV3 -->

# EchoMimicV3: Revolutionizing Human Animation with a Compact 1.3B Parameter Model

**EchoMimicV3** offers a groundbreaking approach to unified multi-modal and multi-task human animation, all while maintaining an impressively compact model size of just 1.3 billion parameters. This repository contains the code and resources for this innovative project. ([Original Repo](https://github.com/antgroup/echomimic_v3))

<div align="center">
  <img src="asset/EchoMimicV3_logo.png.jpg"  height=60>
</div>

**Key Features:**

*   **Unified Approach:** Handles multiple modalities (audio, text) and tasks (talking head, talking body) within a single model.
*   **Compact Size:**  Achieves state-of-the-art results with only 1.3 billion parameters.
*   **Easy to Use:**  Ready-to-use models and straightforward installation instructions.
*   **Flexible:**  Supports various hardware configurations, from 16GB to 24GB of VRAM and beyond.
*   **Community Driven:** Active development with contributions and community support.

<div align="center">
    <a href='https://github.com/antgroup/echomimic_v3'><img src='https://img.shields.io/github/stars/antgroup/echomimic_v3?style=social'></a>
    <a href='https://antgroup.github.io/ai/echomimic_v3/'><img src='https://img.shields.io/badge/Project-Page-blue'></a>
    <a href='https://arxiv.org/abs/2507.03905'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
    <a href='https://huggingface.co/BadToBest/EchoMimicV3'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-yellow'></a>
    <a href='https://modelscope.cn/models/BadToBest/EchoMimicV3'><img src='https://img.shields.io/badge/ModelScope-Model-purple'></a>
    <a href='https://github.com/antgroup/echomimic_v3/blob/main/asset/wechat_group.png'><img src='https://badges.aleen42.com/src/wechat.svg'></a>
    <a href='https://github.com/antgroup/echomimic_v3/discussions/18'><img src='https://img.shields.io/badge/中文版-常见问题汇总-orange'></a>
</div>

<div align="center">
  <img src="asset/algo_framework.jpg"  height=600>
</div>

## 🚀 Recent Updates

*   **[2025.08.21]** 🔥 EchoMimicV3 Gradio Demo on [ModelScope](https://modelscope.cn/studios/BadToBest/EchoMimicV3) is LIVE!
*   **[2025.08.12]** 🔥 Generate videos with **12GB VRAM** using the [Gradio UI](https://github.com/antgroup/echomimic_v3/blob/main/app_mm.py) and see the tutorial from @[gluttony-10](https://github.com/gluttony-10).
*   **[2025.08.12]** 🔥 Run EchoMimicV3 on **16GB VRAM** using [ComfyUI](https://github.com/smthemex/ComfyUI_EchoMimic) - thanks @[smthemex](https://github.com/smthemex)!
*   **[2025.08.09]** 🔥 Models released on [ModelScope](https://modelscope.cn/models/BadToBest/EchoMimicV3).
*   **[2025.08.08]** 🔥 Code released on GitHub and models on [Hugging Face](https://huggingface.co/BadToBest/EchoMimicV3).
*   **[2025.07.08]** 🔥 Paper is now available on arXiv: [https://arxiv.org/abs/2507.03905](https://arxiv.org/abs/2507.03905).

## ✨ Gallery

<div align="center">
  <img src="asset/echomimicv3.jpg"  height=1000>
</div>
<table class="center">
  <tr>
    <td width="100%" style="border: none">
      <video controls loop src="https://github.com/user-attachments/assets/f33edb30-66b1-484b-8be0-a5df20a44f3b" muted="false"></video>
    </td>
  </tr>
  <tr>
    <td width="100%" style="border: none">
      <video controls loop src="https://github.com/user-attachments/assets/056105d8-47cd-4a78-8ec2-328ceaf95a5a" muted="false"></video>
    </td>
  </tr>
</table>

### Chinese Driven Audio
<table class="center">
  <tr>
    <td width="25%" style="border: none">
      <video controls loop src="https://github.com/user-attachments/assets/fc1ebae4-b571-43eb-a13a-7d6d05b74082" muted="false"></video>
    </td>
    <td width="25%" style="border: none">
      <video controls loop src="https://github.com/user-attachments/assets/54607cc7-944c-4529-9bef-715862ba330d" muted="false"></video>
    </td>
    <td width="25%" style="border: none">
      <video controls loop src="https://github.com/user-attachments/assets/4d1de999-cce2-47ab-89ed-f2fa11c838fe" muted="false"></video>
    </td>
    <td width="25%" style="border: none">
      <video controls loop src="https://github.com/user-attachments/assets/41e701cc-ac3e-4dd8-b94c-859261f17344" muted="false"></video>
    </td>
  </tr>
</table>
For more demo videos, please refer to the [project page](https://antgroup.github.io/ai/echomimic_v3/)

## 💻 Quick Start

### Environment Setup

*   **Tested System:** Centos 7.2/Ubuntu 22.04, Cuda >= 12.1
*   **Tested GPUs:** A100(80G) / RTX4090D (24G) / V100(16G)
*   **Tested Python:** 3.10 / 3.11

### 🛠️ Installation

#### 1.  Windows (Simplified)

*   Use the [one-click installation package](https://pan.baidu.com/share/init?surl=cV7i2V0wF4exDtKjJrAUeA) (passport: glut) for a quick start.

#### 2. Linux

##### a. Create a Conda Environment

```bash
conda create -n echomimic_v3 python=3.10
conda activate echomimic_v3
```

##### b. Install Dependencies

```bash
pip install -r requirements.txt
```

### 🧱 Model Preparation

| Model                                      | Download Link                                                                      | Notes                                |
| ------------------------------------------ | ---------------------------------------------------------------------------------- | ------------------------------------ |
| Wan2.1-Fun-V1.1-1.3B-InP                  | 🤗 [Hugging Face](https://huggingface.co/alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP)    | Base Model                           |
| wav2vec2-base                              | 🤗 [Hugging Face](https://huggingface.co/facebook/wav2vec2-base-960h)           | Audio Encoder                        |
| EchoMimicV3-preview                         | 🤗 [Hugging Face](https://huggingface.co/BadToBest/EchoMimicV3)                  | Our Weights                          |
| EchoMimicV3-preview                         | 🤗 [ModelScope](https://modelscope.cn/models/BadToBest/EchoMimicV3)              | Our Weights                          |

---
**Organized Weights:**
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

For the Quantified GradioUI version:

```bash
python app_mm.py
```
**Images, audios, masks, and prompts are provided in `datasets/echomimicv3_demos`.**

#### Tips
> - Audio CFG: `audio_guidance_scale` (2~3) for lip sync, higher for lip sync quality, lower for visual quality.
> - Text CFG: `guidance_scale` (3~6) for prompt following, higher for prompt following, lower for visual quality.
> - TeaCache: `teacache_threshold` (0~0.1) is optimal.
> - Sampling steps: 5 steps for talking head, 15~25 steps for talking body.
> - Long video generation: Use Long Video CFG.
> - `partial_video_length` (81, 65 or smaller) to reduce VRAM.


## 📝 TODO List
| Status | Milestone                                                                |     
|:--------:|:-------------------------------------------------------------------------|
|    ✅    | The inference code of EchoMimicV3 meet everyone on GitHub   | 
|    ✅   | EchoMimicV3-preview model on HuggingFace | 
|    ✅   | EchoMimicV3-preview model on ModelScope | 
|    ✅  | ModelScope Space | 
|    🚀    | 720P Pretrained models | 
|    🚀    | The training code of EchoMimicV3 meet everyone on GitHub   | 


## 🔗 EchoMimic Series

*   **EchoMimicV3:** 1.3B Parameters are All You Need for Unified Multi-Modal and Multi-Task Human Animation. [GitHub](https://github.com/antgroup/echomimic_v3)
*   EchoMimicV2: Towards Striking, Simplified, and Semi-Body Human Animation. [GitHub](https://github.com/antgroup/echomimic_v2)
*   EchoMimicV1: Lifelike Audio-Driven Portrait Animations through Editable Landmark Conditioning. [GitHub](https://github.com/antgroup/echomimic)

## 📚 Citation

If you use our work, please cite the following paper:

```
@misc{meng2025echomimicv3,
  title={EchoMimicV3: 1.3B Parameters are All You Need for Unified Multi-Modal and Multi-Task Human Animation},
  author={Rang Meng, Yan Wang, Weipeng Wu, Ruobing Zheng, Yuming Li, Chenguang Ma},
  year={2025},
  eprint={2507.03905},
  archivePrefix={arXiv}
}
```

## 🤝 Reference

*   Wan2.1: [https://github.com/Wan-Video/Wan2.1/](https://github.com/Wan-Video/Wan2.1/)
*   VideoX-Fun: [https://github.com/aigc-apps/VideoX-Fun/](https://github.com/aigc-apps/VideoX-Fun/)

## 📜 License

The models in this repository are licensed under the Apache 2.0 License.  You are responsible for your usage and ensuring compliance with the license.

## ✨ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=antgroup/echomimic_v3&type=Date)](https://www.star-history.com/#antgroup/echomimic_v3&Date)