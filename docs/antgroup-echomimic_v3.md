# EchoMimicV3: Unleash Lifelike Human Animation with Just 1.3 Billion Parameters!

EchoMimicV3 offers a cutting-edge approach to unified multi-modal and multi-task human animation, all with a surprisingly compact model size.  Explore the code and models on [the original GitHub repository](https://github.com/antgroup/echomimic_v3).

## Key Features

*   **Unified Multi-Modal Animation:**  Seamlessly integrates various input modalities like audio, text, and images to generate realistic human animations.
*   **Multi-Task Capabilities:**  Excels at diverse animation tasks, including talking head generation, full-body movement, and lip-syncing.
*   **Compact Model Size:** Achieves impressive results with only 1.3 billion parameters, enabling efficient deployment and faster inference.
*   **Easy to Use:** Simple installation steps with pre-trained models and clear instructions.
*   **Community Support:** Active development and community contributions with ModelScope and HuggingFace demos.

## What's New

*   **[2025.08.21]** 🔥 EchoMimicV3 Gradio demo available on [ModelScope](https://modelscope.cn/studios/BadToBest/EchoMimicV3).
*   **[2025.08.12]** 🔥 **Generate videos with only 12GB VRAM** using the [GradioUI](https://github.com/antgroup/echomimic_v3/blob/main/app_mm.py) and check out the tutorial from @[gluttony-10](https://github.com/gluttony-10)
*   **[2025.08.12]** 🔥 Run EchoMimicV3 with **16G VRAM** using [ComfyUI](https://github.com/smthemex/ComfyUI_EchoMimic) - thanks @[smthemex](https://github.com/smthemex)!
*   **[2025.08.09]** 🔥 Models released on [ModelScope](https://modelscope.cn/models/BadToBest/EchoMimicV3).
*   **[2025.08.08]** 🔥 Codes released on GitHub and models released on [HuggingFace](https://huggingface.co/BadToBest/EchoMimicV3).
*   **[2025.07.08]** 🔥 Paper released on arXiv: [https://arxiv.org/abs/2507.03905](https://arxiv.org/abs/2507.03905).

## Gallery: See EchoMimicV3 in Action!

*   [Interactive Demo](https://antgroup.github.io/ai/echomimic_v3/)

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

## Quick Start

### Environment Setup

*   **Tested System Environment:** Centos 7.2/Ubuntu 22.04, Cuda >= 12.1
*   **Tested GPUs:** A100(80G) / RTX4090D (24G) / V100(16G)
*   **Tested Python Version:** 3.10 / 3.11

### Installation

#### 🛠️ Installation for Windows

##### Use the [one-click installation package](https://pan.baidu.com/share/init?surl=cV7i2V0wF4exDtKjJrAUeA) (passport: glut) for a quick start with the Quantified version.

#### 🛠️ Installation for Linux

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

| Models        |                       Download Link                                           |    Notes                      |
| --------------|-------------------------------------------------------------------------------|-------------------------------|
| Wan2.1-Fun-V1.1-1.3B-InP  |      🤗 [Huggingface](https://huggingface.co/alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP)       | Base model
| wav2vec2-base |      🤗 [Huggingface](https://huggingface.co/facebook/wav2vec2-base-960h)          | Audio encoder
| EchoMimicV3-preview      |      🤗 [Huggingface](https://huggingface.co/BadToBest/EchoMimicV3)              | Our weights
| EchoMimicV3-preview      |      🤗 [ModelScope](https://modelscope.cn/models/BadToBest/EchoMimicV3)              | Our weights

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

**Images, audios, masks, and prompts are provided in `datasets/echomimicv3_demos`**

#### Tips

*   **Audio CFG:** `audio_guidance_scale` (2~3 optimal).
*   **Text CFG:** `guidance_scale` (3~6 optimal).
*   **TeaCache:** `teacache_threshold` (0~0.1 optimal).
*   **Sampling steps:** 5 steps (talking head), 15~25 steps (talking body).
*   **Long Video Generation:** Use Long Video CFG for videos longer than 138 frames.
*   **Reduce VRAM Usage:** Try setting `partial_video_length` to 81, 65, or smaller.

## 📝 TODO List

| Status | Milestone                                                                |
|:--------:|:-------------------------------------------------------------------------|
|    ✅    | The inference code of EchoMimicV3 meet everyone on GitHub   |
|    ✅   | EchoMimicV3-preview model on HuggingFace |
|    ✅   | EchoMimicV3-preview model on ModelScope |
|    ✅  | ModelScope Space |
|    🚀    | 720P Pretrained models |
|    🚀    | The training code of EchoMimicV3 meet everyone on GitHub   |

## &#x1F680; EchoMimic Series

*   **EchoMimicV3:** 1.3B Parameters are All You Need for Unified Multi-Modal and Multi-Task Human Animation. ([GitHub](https://github.com/antgroup/echomimic_v3))
*   EchoMimicV2: Towards Striking, Simplified, and Semi-Body Human Animation. ([GitHub](https://github.com/antgroup/echomimic_v2))
*   EchoMimicV1: Lifelike Audio-Driven Portrait Animations through Editable Landmark Conditioning. ([GitHub](https://github.com/antgroup/echomimic))

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

## Reference

*   Wan2.1: [https://github.com/Wan-Video/Wan2.1/](https://github.com/Wan-Video/Wan2.1/)
*   VideoX-Fun: [https://github.com/aigc-apps/VideoX-Fun/](https://github.com/aigc-apps/VideoX-Fun/)

## 📜 License

The models in this repository are licensed under the Apache 2.0 License.  You are responsible for your use of the models and must adhere to the license terms.

## &#x1F31F; Star History

[![Star History Chart](https://api.star-history.com/svg?repos=antgroup/echomimic_v3&type=Date)](https://www.star-history.com/#antgroup/echomimic_v3&Date)