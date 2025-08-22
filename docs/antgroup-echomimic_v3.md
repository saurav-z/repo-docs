# EchoMimicV3: Unleash Lifelike Human Animation with Just 1.3 Billion Parameters

EchoMimicV3 delivers state-of-the-art human animation with an efficient 1.3B parameter model, enabling unified multi-modal and multi-task capabilities.  Explore the power of this innovative technology on the [EchoMimicV3 GitHub](https://github.com/antgroup/echomimic_v3) repository.

**Key Features:**

*   **Unified Multi-Modal Animation:** Generate realistic animations from audio, text prompts, and more.
*   **Multi-Task Capabilities:**  Handle diverse animation tasks with a single, compact model.
*   **Efficient Parameter Usage:** Achieve high-quality results with only 1.3B parameters.
*   **Gradio Demo:** Experience EchoMimicV3 through an interactive Gradio demo on [ModelScope](https://modelscope.cn/studios/BadToBest/EchoMimicV3).
*   **ComfyUI Integration:** Run EchoMimicV3 on 16G VRAM using [ComfyUI](https://github.com/smthemex/ComfyUI_EchoMimic).
*   **Model Availability:** Access pre-trained models on [Hugging Face](https://huggingface.co/BadToBest/EchoMimicV3) and [ModelScope](https://modelscope.cn/models/BadToBest/EchoMimicV3).
*   **Comprehensive Resources:** Find project details, demo videos, and more on the [project page](https://antgroup.github.io/ai/echomimic_v3/).

**What's New:**

*   **2025.08.21:** Gradio demo released on ModelScope.
*   **2025.08.12:**  Generate videos on 12G VRAM using the provided GradioUI, and on 16G VRAM with ComfyUI.
*   **2025.08.09:** Model released on ModelScope.
*   **2025.08.08:** Code and models released on GitHub and Hugging Face.
*   **2025.07.08:** Paper released on arXiv.

**[Click here for more demos and project details](https://antgroup.github.io/ai/echomimic_v3/)**

## Quick Start

### Environment Setup

*   **OS:** Centos 7.2/Ubuntu 22.04
*   **CUDA:** \>= 12.1
*   **GPUs:** A100(80G) / RTX4090D (24G) / V100(16G)
*   **Python:** 3.10 / 3.11

### Installation

**For Windows:**

*   Use the [one-click installation package](https://pan.baidu.com/share/init?surl=cV7i2V0wF4exDtKjJrAUeA) for a quick start.

**For Linux:**

1.  Create a conda environment:

    ```bash
    conda create -n echomimic_v3 python=3.10
    conda activate echomimic_v3
    ```

2.  Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

### Model Preparation

Download the necessary models:

| Model                                      | Download Link                                                                                             | Notes          |
| ------------------------------------------ | --------------------------------------------------------------------------------------------------------- | -------------- |
| Wan2.1-Fun-V1.1-1.3B-InP                | [Huggingface](https://huggingface.co/alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP)                                    | Base model     |
| wav2vec2-base                            | [Huggingface](https://huggingface.co/facebook/wav2vec2-base-960h)                                             | Audio encoder  |
| EchoMimicV3-preview                       | [Huggingface](https://huggingface.co/BadToBest/EchoMimicV3)  /  [ModelScope](https://modelscope.cn/models/BadToBest/EchoMimicV3)           | Our weights |

Organize the weights into the following directory structure:

```
./models/
├── Wan2.1-Fun-V1.1-1.3B-InP
├── wav2vec2-base-960h
└── transformer
    └── diffusion_pytorch_model.safetensors
```

### Quick Inference

Run the inference script:

```bash
python infer.py
```

For the GradioUI version:

```bash
python app_mm.py
```

**Datasets:** Sample images, audio, masks, and prompts are available in the `datasets/echomimicv3_demos` directory.

**Tips for Optimal Results:**
*   **Audio CFG:** Experiment with `audio_guidance_scale` values between 2 and 3 for best lip-sync and visual quality.
*   **Text CFG:** Adjust `guidance_scale` between 3 and 6 for prompt following.
*   **TeaCache:** Use a `teacache_threshold` between 0 and 0.1.
*   **Sampling Steps:** 5 steps for talking head animations, 15-25 steps for full-body animations.
*   **Long Video Generation:** Employ Long Video CFG for videos longer than 138 frames.
*   **Reduce VRAM Usage:** Set `partial_video_length` to 81, 65, or smaller.

## Roadmap

| Status | Milestone                                                             |
| :-----: | :-------------------------------------------------------------------- |
|   ✅    | EchoMimicV3 Inference Code Release                                   |
|   ✅   | EchoMimicV3-preview model on Hugging Face                             |
|   ✅   | EchoMimicV3-preview model on ModelScope                               |
|   🚀    | ModelScope Space                                                      |
|   🚀    | Pretrained models trained on English and Chinese on ModelScope       |
|   🚀    | 720P Pretrained models trained on English and Chinese on HuggingFace |
|   🚀    | 720P Pretrained models trained on English and Chinese on ModelScope   |
|   🚀    | EchoMimicV3 Training Code Release                                    |

## EchoMimic Series

*   EchoMimicV3: [GitHub](https://github.com/antgroup/echomimic_v3)
*   EchoMimicV2: [GitHub](https://github.com/antgroup/echomimic_v2)
*   EchoMimicV1: [GitHub](https://github.com/antgroup/echomimic)

## Citation

```
@misc{meng2025echomimicv3,
  title={EchoMimicV3: 1.3B Parameters are All You Need for Unified Multi-Modal and Multi-Task Human Animation},
  author={Rang Meng, Yan Wang, Weipeng Wu, Ruobing Zheng, Yuming Li, Chenguang Ma},
  year={2025},
  eprint={2507.03905},
  archivePrefix={arXiv}
}
```

## References

*   Wan2.1: [Wan2.1 GitHub](https://github.com/Wan-Video/Wan2.1/)
*   VideoX-Fun: [VideoX-Fun GitHub](https://github.com/aigc-apps/VideoX-Fun/)

## License

The models in this repository are licensed under the Apache 2.0 License.  You are responsible for your use of the models and ensuring it complies with the license and all applicable laws.

## Star History
[![Star History Chart](https://api.star-history.com/svg?repos=antgroup/echomimic_v3&type=Date)](https://www.star-history.com/#antgroup/echomimic_v3&Date)