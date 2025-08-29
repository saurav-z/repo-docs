# EchoMimicV3: Revolutionizing Human Animation with a 1.3B Parameter Model

EchoMimicV3 leverages a compact 1.3 billion parameter model to achieve unified multi-modal and multi-task human animation, opening doors to create captivating visual experiences. Explore the power of efficient and versatile human animation with our cutting-edge technology!  **[Visit the original repository on GitHub](https://github.com/antgroup/echomimic_v3) for detailed information and resources.**

## Key Features

*   **Unified Multi-Modal Animation:**  Generate animations driven by various inputs, including audio and text prompts.
*   **Multi-Task Capabilities:**  Perform a range of animation tasks, from talking head to full-body movements.
*   **Efficient Parameter Usage:**  Achieve high-quality results with a relatively small model size (1.3B parameters).
*   **Gradio Demo:** Try the interactive gradio demo on [modelscope](https://modelscope.cn/studios/BadToBest/EchoMimicV3)
*   **Community Contributions**: Run EchoMimicV3 on 12G VRAM through [GradioUI](https://github.com/antgroup/echomimic_v3/blob/main/app_mm.py). Thanks to the community.
*   **ComfyUI Support:** Run EchoMimicV3 on 16G VRAM using [ComfyUI](https://github.com/smthemex/ComfyUI_EchoMimic). Thanks @[smthemex](https://github.com/smthemex) for the contribution.
*   **Model Availability**: Access our models on [Hugging Face](https://huggingface.co/BadToBest/EchoMimicV3) and [ModelScope](https://modelscope.cn/models/BadToBest/EchoMimicV3)

## What's New

*   **[2025.08.21]**  EchoMimicV3 gradio demo on [modelscope](https://modelscope.cn/studios/BadToBest/EchoMimicV3) is ready.
*   **[2025.08.12]**  12G VRAM is All YOU NEED to Generate Video.
*   **[2025.08.12]**  EchoMimicV3 can run on 16G VRAM using [ComfyUI](https://github.com/smthemex/ComfyUI_EchoMimic).
*   **[2025.08.09]**  Models released on ModelScope.
*   **[2025.08.08]**  Codes released on GitHub and models on Hugging Face.
*   **[2025.07.08]**  Paper released on arXiv.

## Gallery

**(Include impressive video demonstrations and image results from the original README)**

## Quick Start

### Environment Setup

*   **Operating System:** Centos 7.2/Ubuntu 22.04
*   **CUDA:** >= 12.1
*   **GPUs:** A100 (80G), RTX4090D (24G), V100 (16G)
*   **Python:** 3.10 / 3.11

### Installation

#### 1.  For Windows:

*   Use the [one-click installation package](https://pan.baidu.com/share/init?surl=cV7i2V0wF4exDtKjJrAUeA).

#### 2.  For Linux:

```bash
conda create -n echomimic_v3 python=3.10
conda activate echomimic_v3
pip install -r requirements.txt
```

### Model Preparation

| Model                                 | Download Link                                                                   | Notes                       |
| ------------------------------------- | ------------------------------------------------------------------------------- | --------------------------- |
| Wan2.1-Fun-V1.1-1.3B-InP              |  🤗 [Huggingface](https://huggingface.co/alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP)        | Base Model                |
| wav2vec2-base                         |  🤗 [Huggingface](https://huggingface.co/facebook/wav2vec2-base-960h)           | Audio Encoder             |
| EchoMimicV3-preview                   |  🤗 [Huggingface](https://huggingface.co/BadToBest/EchoMimicV3)           | Our Weights               |
| EchoMimicV3-preview                   |  🤗 [ModelScope](https://modelscope.cn/models/BadToBest/EchoMimicV3)           | Our Weights               |

*   **Weights Organization:**

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

*   **For Quantified GradioUI:**

```bash
python app_mm.py
```

*   **Datasets:**  Images, audios, masks, and prompts are provided in the `datasets/echomimicv3_demos` directory.

### Tips

*   **Audio CFG:** Optimize `audio_guidance_scale` between 2~3 for lip-sync and visual quality.
*   **Text CFG:** Optimize `guidance_scale` between 3~6 for prompt following and visual quality.
*   **TeaCache:**  Use `teacache_threshold` in the range of 0~0.1.
*   **Sampling Steps:**  Use 5 steps for talking head, 15~25 steps for talking body.
*   **Long Video Generation:** Use Long Video CFG for videos longer than 138 frames.
*   **Reduce VRAM:** Set `partial_video_length` to 81, 65, or smaller.

## Roadmap

**(Include a concise table of TODO items)**

## EchoMimic Series

*   EchoMimicV3: [GitHub](https://github.com/antgroup/echomimic_v3)
*   EchoMimicV2: [GitHub](https://github.com/antgroup/echomimic_v2)
*   EchoMimicV1: [GitHub](https://github.com/antgroup/echomimic)

## Citation

**(Include the citation information)**

## References

**(Include the references)**

## License

The models are licensed under the Apache 2.0 License.