# EchoMimicV3: Revolutionizing Human Animation with 1.3 Billion Parameters

EchoMimicV3 is a groundbreaking advancement in human animation, offering unified multi-modal and multi-task capabilities with only 1.3 billion parameters.  [Explore the original repository on GitHub](https://github.com/antgroup/echomimic_v3) for more details.

**Key Features:**

*   **Unified Approach:** Seamlessly handles multiple modalities (audio, text) and tasks (talking head, talking body) in a single model.
*   **Efficient Architecture:** Achieves impressive results with a compact 1.3B parameter model.
*   **Easy to Use:**  Simplified installation and inference steps for quick experimentation.
*   **High-Quality Results:**  Produces realistic and engaging human animations.
*   **Open Source:**  Available for research and commercial use under the Apache 2.0 License.

**What's New:**

*   **[2025.08.21]** 🔥 EchoMimicV3 demo on [ModelScope](https://modelscope.cn/studios/BadToBest/EchoMimicV3) is ready.
*   **[2025.08.12]** 🔥🚀 **12G VRAM is All YOU NEED to Generate Video**. Please use this [GradioUI](https://github.com/antgroup/echomimic_v3/blob/main/app_mm.py). Check the [tutorial](https://www.bilibili.com/video/BV1W8tdzEEVN) from @[gluttony-10](https://github.com/gluttony-10). Thanks for the contribution.
*   **[2025.08.12]** 🔥 EchoMimicV3 can run on **16G VRAM** using [ComfyUI](https://github.com/smthemex/ComfyUI_EchoMimic). Thanks @[smthemex](https://github.com/smthemex) for the contribution.
*   **[2025.08.09]** 🔥 Models released on [ModelScope](https://modelscope.cn/models/BadToBest/EchoMimicV3).
*   **[2025.08.08]** 🔥 Codes released on GitHub and models released on [Hugging Face](https://huggingface.co/BadToBest/EchoMimicV3).
*   **[2025.07.08]** 🔥 Paper released on [arXiv](https://arxiv.org/abs/2507.03905).

## Showcase

<p align="center">
  <img src="asset/echomimicv3.jpg"  height=1000>
</p>

**Example Videos:**

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

## Getting Started

### Environment Setup

*   **Tested System Environment:** Centos 7.2/Ubuntu 22.04, Cuda >= 12.1
*   **Tested GPUs:** A100(80G) / RTX4090D (24G) / V100(16G)
*   **Tested Python Version:** 3.10 / 3.11

### Installation

#### 🛠️ Windows (Simplified)

*   Use the [one-click installation package](https://pan.baidu.com/share/init?surl=cV7i2V0wF4exDtKjJrAUeA) (passport: glut) for a quick start.

#### 🛠️ Linux

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

| Model                         | Download Link                                                                   | Notes                |
| :---------------------------- | :------------------------------------------------------------------------------ | :------------------- |
| Wan2.1-Fun-V1.1-1.3B-InP      | [Hugging Face](https://huggingface.co/alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP)        | Base model           |
| wav2vec2-base                 | [Hugging Face](https://huggingface.co/facebook/wav2vec2-base-960h)               | Audio encoder        |
| EchoMimicV3-preview           | [Hugging Face](https://huggingface.co/BadToBest/EchoMimicV3)                    | Our weights          |
| EchoMimicV3-preview           | [ModelScope](https://modelscope.cn/models/BadToBest/EchoMimicV3)                    | Our weights          |

The model weights should be organized as follows:

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

**Input data (images, audios, masks, and prompts) are provided in `datasets/echomimicv3_demos`.**

#### Tips

*   **Audio CFG:**  `audio_guidance_scale` (2~3) for optimal lip sync and visual quality.
*   **Text CFG:**  `guidance_scale` (3~6) for prompt adherence and visual quality.
*   **TeaCache:**  `teacache_threshold` (0~0.1)
*   **Sampling steps:**  5 steps for talking head, 15~25 steps for talking body.
*   **Long Video Generation:**  Use Long Video CFG.
*   **VRAM Reduction:** Set `partial_video_length` to 81, 65, or smaller.

## 📝 TODO List

| Status | Milestone                                                              |
| :----: | :--------------------------------------------------------------------- |
|   ✅   | Inference code on GitHub                                               |
|   ✅   | EchoMimicV3-preview model on Hugging Face                              |
|   ✅   | EchoMimicV3-preview model on ModelScope                                |
|   ✅   | ModelScope Space                                                       |
|   🚀   | 720P Pretrained models                                                 |
|   🚀   | Training code on GitHub                                                |

## 🚀 EchoMimic Series

*   **EchoMimicV3:** (This Repository)
*   **EchoMimicV2:** [GitHub](https://github.com/antgroup/echomimic_v2)
*   **EchoMimicV1:** [GitHub](https://github.com/antgroup/echomimic)

## 📚 Citation

If you find our work useful, please cite our paper:

```bibtex
@misc{meng2025echomimicv3,
  title={EchoMimicV3: 1.3B Parameters are All You Need for Unified Multi-Modal and Multi-Task Human Animation},
  author={Rang Meng, Yan Wang, Weipeng Wu, Ruobing Zheng, Yuming Li, Chenguang Ma},
  year={2025},
  eprint={2507.03905},
  archivePrefix={arXiv}
}
```

## 🔗 Reference

*   Wan2.1: [GitHub](https://github.com/Wan-Video/Wan2.1/)
*   VideoX-Fun: [GitHub](https://github.com/aigc-apps/VideoX-Fun/)

## 📜 License

The models are licensed under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0).  You are responsible for your use of the models and must comply with the license.

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=antgroup/echomimic_v3&type=Date)](https://www.star-history.com/#antgroup/echomimic_v3&Date)
```
Key improvements and optimizations in this version:

*   **SEO Optimization:** Included relevant keywords throughout the text (e.g., "human animation," "multi-modal," "multi-task").
*   **Clear Headings and Structure:** Used proper headings and bullet points for readability and SEO.
*   **Concise Summary:**  A one-sentence hook to grab attention.
*   **Emphasis on Key Features:** Highlighting the most important aspects (unified approach, efficiency, ease of use).
*   **Call to Action:** Directing the user to the GitHub repository.
*   **Updated Information:**  Included the most recent updates from the original README.
*   **Removed unnecessary content**
*   **Better organization for the quick start**