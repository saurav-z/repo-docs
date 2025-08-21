# EchoMimicV3: Generate Stunning Multi-Modal Human Animation with Just 1.3 Billion Parameters

EchoMimicV3 empowers you to create lifelike human animations from various inputs, offering a unified solution for multi-modal and multi-task animation with a streamlined 1.3B parameter model. Explore the power of this innovative technology on the [original GitHub repository](https://github.com/antgroup/echomimic_v3).

**Key Features:**

*   ✨ **Unified Approach:** Seamlessly handles multiple modalities (audio, text, etc.) and diverse animation tasks.
*   🚀 **Efficient & Powerful:** Achieves impressive results with only 1.3 billion parameters.
*   🖼️ **Versatile Applications:** Generate talking head videos, animated bodies, and more.
*   🛠️ **Easy to Use:** Includes straightforward installation and inference instructions.
*   💻 **Flexible Deployment:** Supports various environments (Linux, Windows) and hardware configurations.

**What's New:**

*   **[2025.08.21]** 🔥 EchoMimicV3 demo on [ModelScope](https://modelscope.cn/studios/BadToBest/EchoMimicV3) is ready.
*   **[2025.08.12]** 🔥🚀 **12G VRAM is All YOU NEED to Generate Video**. Please use this [GradioUI](https://github.com/antgroup/echomimic_v3/blob/main/app_mm.py). Check the [tutorial](https://www.bilibili.com/video/BV1W8tdzEEVN) from @[gluttony-10](https://github.com/gluttony-10). Thanks for the contribution.
*   **[2025.08.12]** 🔥 EchoMimicV3 can run on **16G VRAM** using [ComfyUI](https://github.com/smthemex/ComfyUI_EchoMimic). Thanks @[smthemex](https://github.com/smthemex) for the contribution.
*   **[2025.08.09]** 🔥 We release our [models](https://modelscope.cn/models/BadToBest/EchoMimicV3) on ModelScope.
*   **[2025.08.08]** 🔥 We release our [codes](https://github.com/antgroup/echomimic_v3) on GitHub and [models](https://huggingface.co/BadToBest/EchoMimicV3) on Huggingface.
*   **[2025.07.08]** 🔥 Our [paper](https://arxiv.org/abs/2507.03905) is in public on arxiv.

## Gallery

<p align="center">
  <img src="asset/echomimicv3.jpg"  height=500>
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

*   **Tested Systems:** Centos 7.2/Ubuntu 22.04, Cuda >= 12.1
*   **Tested GPUs:** A100(80G) / RTX4090D (24G) / V100(16G)
*   **Tested Python Versions:** 3.10 / 3.11

### Installation

#### 🛠️ Windows

*   For a quick start, use the [one-click installation package](https://pan.baidu.com/share/init?surl=cV7i2V0wF4exDtKjJrAUeA) (Quantified version).

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

| Model                               | Download Link                                                                    | Notes                      |
| ----------------------------------- | -------------------------------------------------------------------------------- | -------------------------- |
| Wan2.1-Fun-V1.1-1.3B-InP             | 🤗 [Huggingface](https://huggingface.co/alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP)     | Base model                  |
| wav2vec2-base                       | 🤗 [Huggingface](https://huggingface.co/facebook/wav2vec2-base-960h)            | Audio encoder              |
| EchoMimicV3-preview                 | 🤗 [Huggingface](https://huggingface.co/BadToBest/EchoMimicV3)                   | Our weights                 |
| EchoMimicV3-preview                 | 🤗 [ModelScope](https://modelscope.cn/models/BadToBest/EchoMimicV3)              | Our weights                 |

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

**Example datasets (images, audios, masks, prompts) are in `datasets/echomimicv3_demos`.**

#### Tips

*   **Audio CFG:** Optimal `audio_guidance_scale` range is 2-3. Increase for lip sync, decrease for visual quality.
*   **Text CFG:** Optimal `guidance_scale` range is 3-6. Increase for prompt following, decrease for visual quality.
*   **TeaCache:** Optimal `teacache_threshold` is between 0-0.1.
*   **Sampling steps:** Use 5 steps for talking head, 15-25 steps for talking body.
*   **Long Video Generation:** Use Long Video CFG for videos longer than 138 frames.
*   Set `partial_video_length` to 81, 65 or smaller to reduce VRAM.

## 📝 TODO List

| Status | Milestone                                                                |
| :----: | :------------------------------------------------------------------------- |
|   ✅   | The inference code of EchoMimicV3 meet everyone on GitHub   |
|   ✅  | EchoMimicV3-preview model on HuggingFace |
|   ✅  | EchoMimicV3-preview model on ModelScope |
|  🚀  | ModelScope Space |
|  🚀   | Preview version Pretrained models trained on English and Chinese on ModelScope   |
|  🚀   | 720P Pretrained models trained on English and Chinese on HuggingFace |
|  🚀   | 720P Pretrained models trained on English and Chinese on ModelScope   |
|  🚀   | The training code of EchoMimicV3 meet everyone on GitHub   |

## ⚙️ EchoMimic Series

*   **EchoMimicV3:** (This Repository)
*   **EchoMimicV2:** [GitHub](https://github.com/antgroup/echomimic_v2)
*   **EchoMimicV1:** [GitHub](https://github.com/antgroup/echomimic)

## 📚 Citation

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

*   Wan2.1: [GitHub](https://github.com/Wan-Video/Wan2.1/)
*   VideoX-Fun: [GitHub](https://github.com/aigc-apps/VideoX-Fun/)

## 📜 License

The models in this repository are licensed under the Apache 2.0 License.  You are fully accountable for your use of the models, which must not involve sharing any content that violates applicable laws.

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=antgroup/echomimic_v3&type=Date)](https://www.star-history.com/#antgroup/echomimic_v3&Date)