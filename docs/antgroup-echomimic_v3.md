# EchoMimicV3: Unleash Unified Multi-Modal Human Animation with Just 1.3 Billion Parameters

**EchoMimicV3** is a groundbreaking model that simplifies and unifies multi-modal and multi-task human animation, achieving impressive results with only 1.3 billion parameters. Explore the power of this innovative approach and revolutionize your animation projects.

[Visit the original repository on GitHub](https://github.com/antgroup/echomimic_v3)

<div align='center'>
    <a href='https://github.com/antgroup/echomimic_v3'><img src='https://img.shields.io/github/stars/antgroup/echomimic_v3?style=social'></a>
    <a href='https://antgroup.github.io/ai/echomimic_v3/'><img src='https://img.shields.io/badge/Project-Page-blue'></a>
    <a href='https://arxiv.org/abs/2507.03905'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
    <a href='https://huggingface.co/BadToBest/EchoMimicV3'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-yellow'></a>
    <a href='https://modelscope.cn/models/BadToBest/EchoMimicV3'><img src='https://img.shields.io/badge/ModelScope-Model-purple'></a>
</div>

## Key Features:

*   **Unified Approach:**  Handles multiple modalities (audio, text, etc.) and animation tasks within a single model.
*   **Compact Size:** Achieves state-of-the-art results with only 1.3 billion parameters, making it efficient and accessible.
*   **High-Quality Results:** Generates lifelike and engaging human animations.
*   **Gradio Demo:** Interactive demo available on ModelScope ([https://modelscope.cn/studios/BadToBest/EchoMimicV3](https://modelscope.cn/studios/BadToBest/EchoMimicV3)).
*   **Flexible Deployment:**  Supports running on 16GB VRAM using ComfyUI.

## Latest Updates:

*   **[2025.08.21]** 🔥 EchoMimicV3 Gradio Demo on ModelScope is Live!
*   **[2025.08.12]** 🔥 12GB VRAM is All YOU NEED for Video Generation using the GradioUI ([app_mm.py](https://github.com/antgroup/echomimic_v3/blob/main/app_mm.py)) and tutorial.
*   **[2025.08.12]** 🔥 Supports ComfyUI for running on 16GB VRAM.
*   **[2025.08.09]** 🔥 Models Released on ModelScope.
*   **[2025.08.08]** 🔥 Code Released on GitHub & Models Released on Hugging Face.
*   **[2025.07.08]** 🔥 Paper available on arXiv ([https://arxiv.org/abs/2507.03905](https://arxiv.org/abs/2507.03905)).

## Gallery:

*(Include the image and video examples from the original README here)*

## Quick Start:

### Environment Setup:

*   **Tested Systems:** Centos 7.2/Ubuntu 22.04, Cuda >= 12.1
*   **Tested GPUs:** A100(80G) / RTX4090D (24G) / V100(16G)
*   **Tested Python Versions:** 3.10 / 3.11

### Installation:

**Windows:**

*   Use the [one-click installation package](https://pan.baidu.com/share/init?surl=cV7i2V0wF4exDtKjJrAUeA) for a quick start (Quantified version).

**Linux:**

1.  Create a Conda environment:

```bash
conda create -n echomimic_v3 python=3.10
conda activate echomimic_v3
```

2.  Install dependencies:

```bash
pip install -r requirements.txt
```

### Model Preparation:

| Model                                   | Download Link                                                                                                    | Notes                  |
| --------------------------------------- | ---------------------------------------------------------------------------------------------------------------- | ---------------------- |
| Wan2.1-Fun-V1.1-1.3B-InP                | [Hugging Face](https://huggingface.co/alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP)                                       | Base Model             |
| wav2vec2-base                           | [Hugging Face](https://huggingface.co/facebook/wav2vec2-base-960h)                                                 | Audio Encoder          |
| EchoMimicV3-preview                     | [Hugging Face](https://huggingface.co/BadToBest/EchoMimicV3)                                                       | Our Weights            |
| EchoMimicV3-preview                     | [ModelScope](https://modelscope.cn/models/BadToBest/EchoMimicV3)                                                    | Our Weights            |

*   Place the downloaded models in the following directory structure:

```
./models/
├── Wan2.1-Fun-V1.1-1.3B-InP
├── wav2vec2-base-960h
└── transformer
    └── diffusion_pytorch_model.safetensors
```

### Quick Inference:

```bash
python infer.py
```

For the Quantified GradioUI version:

```bash
python app_mm.py
```

*   **Datasets:** Example images, audio, masks, and prompts are provided in the `datasets/echomimicv3_demos` directory.

#### Tips:

*   **Audio CFG:**  `audio_guidance_scale` (2~3).
*   **Text CFG:** `guidance_scale` (3~6).
*   **TeaCache:** `teacache_threshold` (0~0.1).
*   **Sampling Steps:** 5 steps for talking head, 15-25 steps for talking body.
*   **Long Video Generation:** Use Long Video CFG.
*   **VRAM Usage:** Set `partial_video_length` to 81, 65, or smaller to reduce VRAM usage.

## TODO List:

*(Include the TODO list from the original README)*

## EchoMimic Series:

*   EchoMimicV3: 1.3B Parameters are All You Need for Unified Multi-Modal and Multi-Task Human Animation. [GitHub](https://github.com/antgroup/echomimic_v3)
*   EchoMimicV2: Towards Striking, Simplified, and Semi-Body Human Animation. [GitHub](https://github.com/antgroup/echomimic_v2)
*   EchoMimicV1: Lifelike Audio-Driven Portrait Animations through Editable Landmark Conditioning. [GitHub](https://github.com/antgroup/echomimic)

## Citation:

*(Include the citation information from the original README)*

## Reference:

*(Include the references from the original README)*

## License:

The models are licensed under the Apache 2.0 License.  Use the models responsibly, complying with all applicable laws.  You are fully accountable for your use.

## Star History:

*(Include the Star History Chart from the original README)*