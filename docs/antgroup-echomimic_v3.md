# EchoMimicV3: Animate Humans with Just 1.3 Billion Parameters!

> Unlock the power of unified multi-modal and multi-task human animation with the EchoMimicV3 model.

[View the original repository on GitHub](https://github.com/antgroup/echomimic_v3)

<p align="center">
  <img src="asset/EchoMimicV3_logo.png.jpg"  height=60>
</p>

## Key Features

*   **Unified Multi-Modal & Multi-Task Animation:**  EchoMimicV3 excels at generating human animations from diverse inputs.
*   **Efficient Parameterization:** Achieve high-quality results with only 1.3 billion parameters.
*   **Easy to Use:** Includes a Gradio demo for quick experimentation and ComfyUI support.
*   **Active Development:**  Ongoing updates and improvements are constantly being released.
*   **Community-Driven:**  Join the community on GitHub and ModelScope and find solutions to common problems.

## What's New

*   **[2025.08.21]**  Gradio demo on [modelscope](https://modelscope.cn/studios/BadToBest/EchoMimicV3) is ready.
*   **[2025.08.12]**  Generate videos using 12GB VRAM! Use this [GradioUI](https://github.com/antgroup/echomimic_v3/blob/main/app_mm.py).  Check the [tutorial](https://www.bilibili.com/video/BV1W8tdzEEVN).
*   **[2025.08.12]**  Run EchoMimicV3 on **16G VRAM** using [ComfyUI](https://github.com/smthemex/ComfyUI_EchoMimic).
*   **[2025.08.09]**  Models are released on [ModelScope](https://modelscope.cn/models/BadToBest/EchoMimicV3).
*   **[2025.08.08]**  Codes released on GitHub and models on [HuggingFace](https://huggingface.co/BadToBest/EchoMimicV3).
*   **[2025.07.08]**  Paper published on [arXiv](https://arxiv.org/abs/2507.03905).

## Gallery

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

## Quick Start

### Environment Setup

*   **OS:** Centos 7.2/Ubuntu 22.04
*   **CUDA:** >= 12.1
*   **GPUs:** A100(80G) / RTX4090D (24G) / V100(16G)
*   **Python:** 3.10 / 3.11

### 🛠️ Installation
#### 1.  **Windows (Quantified Version):**
    *   Use the [one-click installation package](https://pan.baidu.com/share/init?surl=cV7i2V0wF4exDtKjJrAUeA) (passport: glut).
#### 2.  **Linux:**

    *   **Create Conda Environment:**
        ```bash
        conda create -n echomimic_v3 python=3.10
        conda activate echomimic_v3
        ```
    *   **Install Dependencies:**
        ```bash
        pip install -r requirements.txt
        ```

### 🧱 Model Preparation

| Models                      | Download Link                                                     | Notes                      |
| --------------------------- | ----------------------------------------------------------------- | -------------------------- |
| Wan2.1-Fun-V1.1-1.3B-InP   | 🤗 [Huggingface](https://huggingface.co/alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP)       | Base model
| wav2vec2-base              | 🤗 [Huggingface](https://huggingface.co/facebook/wav2vec2-base-960h)          | Audio encoder
| EchoMimicV3-preview       | 🤗 [Huggingface](https://huggingface.co/BadToBest/EchoMimicV3)              | Our weights
| EchoMimicV3-preview       | 🤗 [ModelScope](https://modelscope.cn/models/BadToBest/EchoMimicV3)              | Our weights

-- The **weights** is organized as follows.

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

**images, audios, masks and prompts are provided in `datasets/echomimicv3_demos`**

#### Tips

*   **Audio CFG:** `audio_guidance_scale` (2~3) for better lip sync and visual quality.
*   **Text CFG:** `guidance_scale` (3~6) for prompt following and visual quality.
*   **TeaCache:** `teacache_threshold` (0~0.1) for optimization.
*   **Sampling steps:** 5 steps for talking head, 15~25 steps for talking body.
*   **Long Video Generation:** Use Long Video CFG for videos longer than 138 frames.
*   **Reduce VRAM Usage:** Set `partial_video_length` to 81, 65, or smaller.

## 📝 TODO List

| Status | Milestone                                                                |
| :----: | :------------------------------------------------------------------------- |
|   ✅   | The inference code of EchoMimicV3 meet everyone on GitHub                |
|   ✅   | EchoMimicV3-preview model on HuggingFace                                   |
|   ✅   | EchoMimicV3-preview model on ModelScope                                   |
|   ✅  | ModelScope Space |
|   🚀   | 720P Pretrained models                                                   |
|   🚀   | The training code of EchoMimicV3 meet everyone on GitHub                |

## &#x1F680; EchoMimic Series

*   **EchoMimicV3:** (This Repository)
*   **EchoMimicV2:** [GitHub](https://github.com/antgroup/echomimic_v2)
*   **EchoMimicV1:** [GitHub](https://github.com/antgroup/echomimic)

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

*   Wan2.1: [https://github.com/Wan-Video/Wan2.1/](https://github.com/Wan-Video/Wan2.1/)
*   VideoX-Fun: [https://github.com/aigc-apps/VideoX-Fun/](https://github.com/aigc-apps/VideoX-Fun/)

## 📜 License

This project is licensed under the Apache 2.0 License.

## &#x1F31F; Star History

[![Star History Chart](https://api.star-history.com/svg?repos=antgroup/echomimic_v3&type=Date)](https://www.star-history.com/#antgroup/echomimic_v3&Date)