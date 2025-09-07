# EchoMimicV3: Animate Humans with Just 1.3 Billion Parameters

EchoMimicV3 is a cutting-edge model that revolutionizes human animation, offering unified multi-modal and multi-task capabilities with an efficient 1.3 billion parameter architecture.  Dive into the future of animation and explore the possibilities at the original repository: [https://github.com/antgroup/echomimic_v3](https://github.com/antgroup/echomimic_v3).

**Key Features:**

*   **Unified Multi-Modal Animation:**  Integrates seamlessly with audio, text, and more for diverse animation tasks.
*   **Multi-Task Capabilities:**  Handles a range of animation challenges within a single model.
*   **Efficient Parameterization:** Achieves high-quality results with only 1.3 billion parameters.
*   **Easy to Use:** Ready to run on 12G VRAM, get started quickly with the GradioUI demo.
*   **ComfyUI Support:** Leverage ComfyUI for execution on even smaller VRAM setups.
*   **Model Availability:** Access pretrained models on Hugging Face and ModelScope.

## 🚀 What's New

*   **[2025.08.21]** 🔥 EchoMimicV3 Gradio demo available on [ModelScope](https://modelscope.cn/studios/BadToBest/EchoMimicV3).
*   **[2025.08.12]** 🔥 **12G VRAM is All YOU NEED to Generate Video** with the [GradioUI](https://github.com/antgroup/echomimic_v3/blob/main/app_mm.py). Check the [tutorial](https://www.bilibili.com/video/BV1W8tdzEEVN).
*   **[2025.08.12]** 🔥 EchoMimicV3 runs on **16G VRAM** using [ComfyUI](https://github.com/smthemex/ComfyUI_EchoMimic).
*   **[2025.08.09]** 🔥 Models released on [ModelScope](https://modelscope.cn/models/BadToBest/EchoMimicV3).
*   **[2025.08.08]** 🔥 Codes released on GitHub and models on [HuggingFace](https://huggingface.co/BadToBest/EchoMimicV3).
*   **[2025.07.08]** 🔥 [Paper](https://arxiv.org/abs/2507.03905) available on arXiv.

## ✨ Gallery

[Include the image and video examples from the original README here, re-formatting them for better visual appeal.  Consider using a responsive image gallery if possible.]

## 快速开始 / Quick Start

### Environment Setup

*   **Operating Systems:** Centos 7.2/Ubuntu 22.04, Cuda >= 12.1
*   **GPUs:** A100 (80G) / RTX4090D (24G) / V100 (16G)
*   **Python Version:** 3.10 / 3.11

### 🛠️ Installation

*   **Windows:** Use the [one-click installation package](https://pan.baidu.com/share/init?surl=cV7i2V0wF4exDtKjJrAUeA) (passport: glut) to get started.
*   **Linux:**

    1.  Create a Conda environment:

        ```bash
        conda create -n echomimic_v3 python=3.10
        conda activate echomimic_v3
        ```

    2.  Install Dependencies:

        ```bash
        pip install -r requirements.txt
        ```

### 🧱 Model Preparation

Download the following models and place them in the specified directory structure:

| Model                                  | Download Link                                                                  | Notes              |
| :------------------------------------- | :----------------------------------------------------------------------------- | :----------------- |
| Wan2.1-Fun-V1.1-1.3B-InP            | [Huggingface](https://huggingface.co/alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP) | Base model         |
| wav2vec2-base                        | [Huggingface](https://huggingface.co/facebook/wav2vec2-base-960h)          | Audio encoder      |
| EchoMimicV3-preview                  | [Huggingface](https://huggingface.co/BadToBest/EchoMimicV3)                   | Our weights        |
| EchoMimicV3-preview                  | [ModelScope](https://modelscope.cn/models/BadToBest/EchoMimicV3)                  | Our weights        |

Model directory structure:

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

For the GradioUI version:

```bash
python app_mm.py
```

**Input Data:** Images, audios, masks, and prompts are available in `datasets/echomimicv3_demos`.

#### Tips:

*   **Audio CFG:** Optimal `audio_guidance_scale` between 2~3. Increase for lip sync, decrease for visual quality.
*   **Text CFG:** Optimal `guidance_scale` between 3~6. Increase for prompt following, decrease for visual quality.
*   **TeaCache:**  Optimal range for `teacache_threshold` is between 0~0.1.
*   **Sampling Steps:** 5 steps for talking head, 15~25 steps for talking body.
*   **Long Video Generation:** Use Long Video CFG for videos longer than 138 frames.
*   **VRAM Optimization:** Try setting `partial_video_length` to 81, 65, or lower to reduce VRAM usage.

## 📝 Roadmap

| Status | Milestone                                                                |
| :----: | :------------------------------------------------------------------------- |
|   ✅   | Inference code on GitHub                                                 |
|   ✅   | EchoMimicV3-preview model on HuggingFace                                  |
|   ✅   | EchoMimicV3-preview model on ModelScope                                   |
|   ✅   | ModelScope Space                                                         |
|   🚀   | 720P Pretrained models                                                 |
|   🚀   | Training code on GitHub                                                  |

## 🧬 EchoMimic Series

*   **EchoMimicV3:** (This Repository)
*   [EchoMimicV2](https://github.com/antgroup/echomimic_v2): Towards Striking, Simplified, and Semi-Body Human Animation.
*   [EchoMimicV1](https://github.com/antgroup/echomimic): Lifelike Audio-Driven Portrait Animations through Editable Landmark Conditioning.

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

## 🔗 References

*   Wan2.1: [https://github.com/Wan-Video/Wan2.1/](https://github.com/Wan-Video/Wan2.1/)
*   VideoX-Fun: [https://github.com/aigc-apps/VideoX-Fun/](https://github.com/aigc-apps/VideoX-Fun/)

## 📜 License

This project is licensed under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0).  Your generated content is your responsibility.