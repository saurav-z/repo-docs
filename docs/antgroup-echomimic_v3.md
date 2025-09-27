<!-- Improved & Summarized README for EchoMimicV3 -->

# EchoMimicV3: Revolutionizing Human Animation with Just 1.3 Billion Parameters

**EchoMimicV3** is a cutting-edge model that uses just 1.3 billion parameters to unify multi-modal and multi-task human animation. Explore the future of animation and bring your creative visions to life!  [Access the original repository here](https://github.com/antgroup/echomimic_v3).

<div align="center">
  <img src="asset/EchoMimicV3_logo.png.jpg" height=60>
</div>

## Key Features

*   **Unified Approach:** Supports multi-modal and multi-task human animation.
*   **Compact Architecture:** Achieves impressive results with only 1.3 billion parameters.
*   **User-Friendly Demos:** Ready-to-use demos for quick experimentation.
*   **Community-Driven:**  Active development with contributions from the community, including:
    *   GradioUI demo on ModelScope (12G VRAM needed!)
    *   ComfyUI support (runs on 16G VRAM)
*   **Easy to Use:**  Clear installation and quick inference guides.
*   **Open Source:**  Models and code are available under the Apache 2.0 License.

##  What's New

*   **[2025.08.21]** 🔥 EchoMimicV3 gradio demo on [modelscope](https://modelscope.cn/studios/BadToBest/EchoMimicV3) is ready.
*   **[2025.08.12]** 🔥🚀 **12G VRAM is All YOU NEED to Generate Video**. Please use this [GradioUI](https://github.com/antgroup/echomimic_v3/blob/main/app_mm.py). Check the [tutorial](https://www.bilibili.com/video/BV1W8tdzEEVN) from @[gluttony-10](https://github.com/gluttony-10). Thanks for the contribution.
*   **[2025.08.12]** 🔥 EchoMimicV3 can run on **16G VRAM** using [ComfyUI](https://github.com/smthemex/ComfyUI_EchoMimic). Thanks @[smthemex](https://github.com/smthemex) for the contribution.
*   **[2025.08.09]** 🔥 We release our [models](https://modelscope.cn/models/BadToBest/EchoMimicV3) on ModelScope.
*   **[2025.08.08]** 🔥 We release our [codes](https://github.com/antgroup/echomimic_v3) on GitHub and [models](https://huggingface.co/BadToBest/EchoMimicV3) on Huggingface.
*   **[2025.07.08]** 🔥 Our [paper](https://arxiv.org/abs/2507.03905) is in public on arxiv.

## Gallery

<p align="center">
  <img src="asset/echomimicv3.jpg"  height=700>
</p>

### Video Examples

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
        <video controls loop src="https://github.com/user-attachments/assets/4d1de1999-cce2-47ab-89ed-f2fa11c838fe" muted="false"></video>
    </td>
    <td width=25% style="border: none">
        <video controls loop src="https://github.com/user-attachments/assets/41e701cc-ac3e-4dd8-b94c-859261f17344" muted="false"></video>
    </td>
</tr>
</table>

For more demo videos, please refer to the [project page](https://antgroup.github.io/ai/echomimic_v3/)

## Quick Start

### Prerequisites
*   Centos 7.2/Ubuntu 22.04, Cuda >= 12.1
*   A100(80G) / RTX4090D (24G) / V100(16G)
*   Python: 3.10 / 3.11

### Installation

#### 🛠️ Windows (Recommended)
##### Please use the [one-click installation package](https://pan.baidu.com/share/init?surl=cV7i2V0wF4exDtKjJrAUeA) (passport: glut) to get started quickly for Quantified version.

#### 🛠️ Linux

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

Download the necessary model weights:

| Models        |                       Download Link                                           |    Notes                      |
| --------------|-------------------------------------------------------------------------------|-------------------------------|
| Wan2.1-Fun-V1.1-1.3B-InP  |      🤗 [Huggingface](https://huggingface.co/alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP)       | Base model
| wav2vec2-base |      🤗 [Huggingface](https://huggingface.co/facebook/wav2vec2-base-960h)          | Audio encoder
| EchoMimicV3-preview      |      🤗 [Huggingface](https://huggingface.co/BadToBest/EchoMimicV3)              | Our weights
| EchoMimicV3-preview      |      🤗 [ModelScope](https://modelscope.cn/models/BadToBest/EchoMimicV3)              | Our weights

- The weights should be organized as follows:

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

*   Audio CFG:  Values between 2 and 3 are optimal for audio guidance.
*   Text CFG:  Values between 3 and 6 are optimal for text guidance.
*   TeaCache:  A range between 0~0.1.
*   Sampling Steps:  5 steps for talking head, 15~25 steps for talking body.
*   Long Video Generation: Use Long Video CFG for videos exceeding 138 frames.
*   VRAM Usage: Try setting `partial_video_length` to 81, 65, or smaller.

## 📝 TODO List

| Status | Milestone                                 |
|:--------:|:-----------------------------------------|
|    ✅    | Inference code meets everyone          |
|    ✅   | EchoMimicV3-preview model (HuggingFace) |
|    ✅   | EchoMimicV3-preview model (ModelScope) |
|    ✅  | ModelScope Space |
|    🚀    | 720P Pretrained models |
|    🚀    | Training code release |

##  &#x1F680; EchoMimic Series

*   **EchoMimicV3:** 1.3B Parameters are All You Need for Unified Multi-Modal and Multi-Task Human Animation. [GitHub](https://github.com/antgroup/echomimic_v3)
*   EchoMimicV2: Towards Striking, Simplified, and Semi-Body Human Animation. [GitHub](https://github.com/antgroup/echomimic_v2)
*   EchoMimicV1: Lifelike Audio-Driven Portrait Animations through Editable Landmark Conditioning. [GitHub](https://github.com/antgroup/echomimic)

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