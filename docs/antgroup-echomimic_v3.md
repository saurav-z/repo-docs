# EchoMimicV3: Revolutionizing Human Animation with Just 1.3 Billion Parameters

**EchoMimicV3** offers a groundbreaking approach to unified multi-modal and multi-task human animation, and you can explore its capabilities in this repository: [https://github.com/antgroup/echomimic_v3](https://github.com/antgroup/echomimic_v3).

## Key Features

*   **Unified Approach:** Achieve seamless multi-modal and multi-task human animation within a single, efficient model.
*   **Compact Architecture:**  Leveraging only 1.3 billion parameters, making it accessible and efficient.
*   **Versatile Applications:** Generate lifelike animations from audio, text prompts, and more.
*   **User-Friendly:** Quick start with a provided conda environment and clear installation steps.
*   **Model Availability:** Access pre-trained models on Hugging Face and ModelScope.
*   **Active Community:** Explore discussions and FAQs to troubleshoot common issues.
*   **Performance Boosts:** Runs with as little as 12GB of VRAM.

## What's New

*   **Gradio Demo:** Interactive demo available on ModelScope.
*   **12G VRAM Support:** Generate videos with only 12 GB VRAM, thanks to GradioUI (see `app_mm.py`).
*   **ComfyUI Integration:**  Run EchoMimicV3 on 16G VRAM using ComfyUI.
*   **Model Releases:** Models now available on ModelScope and Hugging Face.
*   **Paper Publication:** Read the EchoMimicV3 paper on arXiv.

## Quick Start

### System Requirements

*   **Operating System:** Centos 7.2 / Ubuntu 22.04
*   **CUDA:** \>= 12.1
*   **GPU:**  A100(80G) / RTX4090D (24G) / V100(16G) (tested)
*   **Python:** 3.10 / 3.11 (tested)

### 🛠️ Installation

1.  **Conda Environment:**

    ```bash
    conda create -n echomimic_v3 python=3.10
    conda activate echomimic_v3
    ```

2.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

### 🧱 Model Preparation

Download the required models and place them in the appropriate directory.

| Models                  |  Download Link                                                                                        | Notes          |
| :----------------------- | :--------------------------------------------------------------------------------------------------- | :------------- |
| Wan2.1-Fun-V1.1-1.3B-InP  | [Huggingface](https://huggingface.co/alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP) | Base model     |
| wav2vec2-base           | [Huggingface](https://huggingface.co/facebook/wav2vec2-base-960h)                    | Audio encoder  |
| EchoMimicV3-preview       | [Huggingface](https://huggingface.co/BadToBest/EchoMimicV3)          | Our weights    |
| EchoMimicV3-preview       | [ModelScope](https://modelscope.cn/models/BadToBest/EchoMimicV3)          | Our weights    |

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

For the Quantified GradioUI version:

```bash
python app_mm.py
```

**Sample Data:** Images, audio, masks, and prompts are provided in `datasets/echomimicv3_demos`.

### Tips

*   **Audio CFG:**  `audio_guidance_scale` works best between 2-3 (for lip-sync).
*   **Text CFG:** `guidance_scale` works best between 3-6 (for prompt following).
*   **TeaCache:**  `teacache_threshold` between 0-0.1.
*   **Sampling Steps:** 5 steps for talking head; 15-25 steps for talking body.
*   **Long Video Generation:** Use Long Video CFG for videos > 138 frames.

## 📝 TODO List

*   ✅ The inference code of EchoMimicV3 meet everyone on GitHub
*   ✅ EchoMimicV3-preview model on HuggingFace
*   ✅ EchoMimicV3-preview model on ModelScope
*   ✅ ModelScope Space
*   🚀 720P Pretrained models
*   🚀 The training code of EchoMimicV3 meet everyone on GitHub

## &#x1F680; EchoMimic Series

*   **EchoMimicV3:** [GitHub](https://github.com/antgroup/echomimic_v3)
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
*   Wan2.1: https://github.com/Wan-Video/Wan2.1/
*   VideoX-Fun: https://github.com/aigc-apps/VideoX-Fun/

## 📜 License

This project is licensed under the Apache 2.0 License.