# EchoMimicV3: Unleashing Unified Multi-Modal Human Animation with Just 1.3B Parameters

EchoMimicV3 is a powerful AI model that enables stunning, unified human animation across multiple modalities and tasks, all with a compact 1.3 billion parameters. For the original repo, see [EchoMimicV3 on GitHub](https://github.com/antgroup/echomimic_v3).

**Key Features:**

*   **Multi-Modal Animation:** Supports animation from text, audio, and other inputs.
*   **Multi-Task Capabilities:** Handles a variety of animation tasks, from talking heads to full-body movements.
*   **Efficient Parameterization:** Achieves impressive results with only 1.3B parameters, making it efficient and accessible.
*   **Gradio and ComfyUI Integration:**  Easily generate videos with available Gradio demo and ComfyUI.
*   **Open Source:** Code and models are available on GitHub and Hugging Face.

## What's New

*   **[2025.08.21]**:  Gradio demo on [ModelScope](https://modelscope.cn/studios/BadToBest/EchoMimicV3) is available.
*   **[2025.08.12]**: Video generation can be performed with **12G VRAM**. 
*   **[2025.08.12]**: EchoMimicV3 can be run on **16G VRAM** using [ComfyUI](https://github.com/smthemex/ComfyUI_EchoMimic).
*   **[2025.08.09]**: Models released on [ModelScope](https://modelscope.cn/models/BadToBest/EchoMimicV3).
*   **[2025.08.08]**: Codes released on GitHub and models released on [Hugging Face](https://huggingface.co/BadToBest/EchoMimicV3).
*   **[2025.07.08]**: Paper released on arXiv ([https://arxiv.org/abs/2507.03905](https://arxiv.org/abs/2507.03905)).

## Gallery of Animated Results

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

### Chinese Audio Driven Examples

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

## Quick Start Guide

### Prerequisites:

*   **Operating System:**  Tested on CentOS 7.2/Ubuntu 22.04
*   **GPU:**  A100(80G) / RTX4090D (24G) / V100(16G) or higher.
*   **Python:**  3.10 / 3.11

### 🛠️ Installation
#### 1. Windows - One-Click Installation
*   Use the [one-click installation package](https://pan.baidu.com/share/init?surl=cV7i2V0wF4exDtKjJrAUeA) for a quick start.

#### 2. Linux Installation

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

Download the required models from the links below.

| Models        |                       Download Link                                           |    Notes                      |
| --------------|-------------------------------------------------------------------------------|-------------------------------|
| Wan2.1-Fun-V1.1-1.3B-InP  |      🤗 [Huggingface](https://huggingface.co/alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP)       | Base model
| wav2vec2-base |      🤗 [Huggingface](https://huggingface.co/facebook/wav2vec2-base-960h)          | Audio encoder
| EchoMimicV3-preview      |      🤗 [Huggingface](https://huggingface.co/BadToBest/EchoMimicV3)              | Our weights
| EchoMimicV3-preview      |      🤗 [ModelScope](https://modelscope.cn/models/BadToBest/EchoMimicV3)              | Our weights

The model weights should be organized as follows:

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

For GradioUI:

```bash
python app_mm.py
```

*   **Datasets:** Images, audio, masks, and prompts are provided in the `datasets/echomimicv3_demos` directory.

#### Inference Tips

*   **Audio CFG:** `audio_guidance_scale` - 2~3 is optimal for lip sync. Increase for better sync, decrease for visual quality.
*   **Text CFG:** `guidance_scale` - 3~6 is optimal for prompt following. Increase for better prompt following, decrease for visual quality.
*   **TeaCache:** `teacache_threshold` -  0~0.1 is optimal.
*   **Sampling Steps:** 5 steps for talking head, 15~25 steps for talking body.
*   **Long Video Generation:**  Use Long Video CFG for videos longer than 138 frames.
*   **VRAM Optimization:** Set `partial_video_length` to 81, 65, or lower to reduce VRAM usage.

## 📝 TODO List

| Status  | Milestone                                                         |
| :------ | :---------------------------------------------------------------- |
| ✅       | Release Inference code on GitHub                                |
| ✅       | Release EchoMimicV3-preview model on HuggingFace                  |
| ✅       | Release EchoMimicV3-preview model on ModelScope                   |
| 🚀       | ModelScope Space                                                 |
| 🚀       | Preview models (English and Chinese) on ModelScope                 |
| 🚀       | 720P models (English and Chinese) on HuggingFace                 |
| 🚀       | 720P models (English and Chinese) on ModelScope                  |
| 🚀       | Training code on GitHub                                           |

## 🚀 EchoMimic Series

*   **EchoMimicV3:** This project - Unified Multi-Modal and Multi-Task Human Animation. [GitHub](https://github.com/antgroup/echomimic_v3)
*   **EchoMimicV2:** Towards Striking, Simplified, and Semi-Body Human Animation. [GitHub](https://github.com/antgroup/echomimic_v2)
*   **EchoMimicV1:** Lifelike Audio-Driven Portrait Animations. [GitHub](https://github.com/antgroup/echomimic)

## 📝 Citation

```bibtex
@misc{meng2025echomimicv3,
  title={EchoMimicV3: 1.3B Parameters are All You Need for Unified Multi-Modal and Multi-Task Human Animation},
  author={Rang Meng, Yan Wang, Weipeng Wu, Ruobing Zheng, Yuming Li, Chenguang Ma},
  year={2025},
  eprint={2507.03905},
  archivePrefix={arXiv}
}
```

## 📚 References

*   Wan2.1: [https://github.com/Wan-Video/Wan2.1/](https://github.com/Wan-Video/Wan2.1/)
*   VideoX-Fun: [https://github.com/aigc-apps/VideoX-Fun/](https://github.com/aigc-apps/VideoX-Fun/)

## 📜 License

The models are licensed under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0).  You are responsible for your use of the models, and usage should comply with all applicable laws.

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=antgroup/echomimic_v3&type=Date)](https://www.star-history.com/#antgroup/echomimic_v3&Date)