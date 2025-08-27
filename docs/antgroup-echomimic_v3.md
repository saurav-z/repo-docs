# EchoMimicV3: Revolutionizing Human Animation with a 1.3B Parameter Model

**EchoMimicV3** is a cutting-edge model that unifies multi-modal and multi-task human animation, achieving remarkable results with only 1.3 billion parameters. Check out the original repo for the full details: [antgroup/echomimic_v3](https://github.com/antgroup/echomimic_v3)

<div align="center">
  <img src="asset/EchoMimicV3_logo.png.jpg"  height=60>
</div>

**Key Features:**

*   **Unified Approach:** A single model for various multi-modal and multi-task human animation tasks.
*   **Compact Architecture:**  Achieves impressive performance with only 1.3B parameters.
*   **Gradio Demo:** Experiment with the model using the Gradio demo on ModelScope.
*   **Reduced VRAM Usage:** Generate video with only **12GB VRAM** using the GradioUI, and it can also run on **16G VRAM** using ComfyUI.
*   **Pre-trained Models:** Access pre-trained models on Hugging Face and ModelScope.
*   **Open Source:** Comprehensive code and models are available on GitHub.

## What's New

*   **Model Availability:** Released models on ModelScope and Hugging Face.
*   **Gradio Demo:** Gradio demo ready on [modelscope](https://modelscope.cn/studios/BadToBest/EchoMimicV3).
*   **Low VRAM Requirements:** Can generate video on **12G VRAM** using [GradioUI](https://github.com/antgroup/echomimic_v3/blob/main/app_mm.py) and even **16G VRAM** using [ComfyUI](https://github.com/smthemex/ComfyUI_EchoMimic).

<p align="center">
  <img src="asset/algo_framework.jpg"  height=700>
</p>

## Demo Gallery

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
*   **CUDA:** \>= 12.1
*   **GPUs:** A100(80G) / RTX4090D (24G) / V100(16G)
*   **Python:** 3.10 / 3.11

### Installation

#### 🛠️ For Windows

*   Use the [one-click installation package](https://pan.baidu.com/share/init?surl=cV7i2V0wF4exDtKjJrAUeA) for a quick start.

#### 🛠️ For Linux

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

| Models                     | Download Link                                                                      | Notes                      |
| -------------------------- | ---------------------------------------------------------------------------------- | -------------------------- |
| Wan2.1-Fun-V1.1-1.3B-InP   | [Huggingface](https://huggingface.co/alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP)       | Base model                 |
| wav2vec2-base              | [Huggingface](https://huggingface.co/facebook/wav2vec2-base-960h)                  | Audio encoder              |
| EchoMimicV3-preview        | [Huggingface](https://huggingface.co/BadToBest/EchoMimicV3)                         | Our weights                |
| EchoMimicV3-preview        | [ModelScope](https://modelscope.cn/models/BadToBest/EchoMimicV3)                         | Our weights                |

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

*   **Audio CFG:** `audio_guidance_scale` (2~3 recommended)
*   **Text CFG:** `guidance_scale` (3~6 recommended)
*   **TeaCache:** `teacache_threshold` (0~0.1 recommended)
*   **Sampling Steps:** 5 steps for talking head, 15~25 steps for talking body.
*   **Long Video Generation:** Use Long Video CFG.
*   **Reduce VRAM:** Set `partial_video_length` to 81, 65 or smaller.

## 📝 TODO List

| Status    | Milestone                                                                |
| :--------: | :------------------------------------------------------------------------- |
|   ✅       | The inference code of EchoMimicV3 meet everyone on GitHub |
|   ✅      | EchoMimicV3-preview model on HuggingFace     |
|   ✅      | EchoMimicV3-preview model on ModelScope   |
|   🚀       | ModelScope Space  |
|   🚀       | Preview version Pretrained models trained on English and Chinese on ModelScope   |
|   🚀       | 720P Pretrained models trained on English and Chinese on HuggingFace |
|   🚀       | 720P Pretrained models trained on English and Chinese on ModelScope   |
|   🚀       | The training code of EchoMimicV3 meet everyone on GitHub  |

## &#x1F680; EchoMimic Series

*   EchoMimicV3: 1.3B Parameters are All You Need for Unified Multi-Modal and Multi-Task Human Animation. [GitHub](https://github.com/antgroup/echomimic_v3)
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

*   Wan2.1: [https://github.com/Wan-Video/Wan2.1/](https://github.com/Wan-Video/Wan2.1/)
*   VideoX-Fun: [https://github.com/aigc-apps/VideoX-Fun/](https://github.com/aigc-apps/VideoX-Fun/)

## 📜 License

The models in this repository are licensed under the Apache 2.0 License.  You are fully accountable for your use of the models.

## &#x1F31F; Star History

[![Star History Chart](https://api.star-history.com/svg?repos=antgroup/echomimic_v3&type=Date)](https://www.star-history.com/#antgroup/echomimic_v3&Date)