# EchoMimicV3: Create Stunning Human Animations with Just 1.3 Billion Parameters

[Original Repo](https://github.com/antgroup/echomimic_v3) | [中文](https://github.com/antgroup/echomimic_v3/blob/main/README_zh.md)

EchoMimicV3 revolutionizes human animation by offering unified multi-modal and multi-task capabilities with an efficient 1.3B parameter model, making high-quality animation accessible to everyone.

## Key Features

*   **Unified Multi-Modal & Multi-Task:**  Handles various animation tasks with a single model.
*   **Efficient Parameter Size:**  Achieves impressive results with only 1.3 billion parameters, making it more accessible.
*   **Gradio Demo Ready:**  Explore the technology with a user-friendly Gradio demo on [ModelScope](https://modelscope.cn/studios/BadToBest/EchoMimicV3).
*   **Low VRAM Requirements:** Generate videos on just 12G VRAM using the provided GradioUI or even on 16G VRAM using ComfyUI.
*   **Model Availability:** Access the models on [Hugging Face](https://huggingface.co/BadToBest/EchoMimicV3) and [ModelScope](https://modelscope.cn/models/BadToBest/EchoMimicV3).
*   **Flexible Inference:** Includes easy-to-use inference scripts.

## What's New

*   **[2025.08.21]** Gradio demo on [modelscope](https://modelscope.cn/studios/BadToBest/EchoMimicV3) is ready.
*   **[2025.08.12]** **12G VRAM is All YOU NEED to Generate Video**. Please use this [GradioUI](https://github.com/antgroup/echomimic_v3/blob/main/app_mm.py). Check the [tutorial](https://www.bilibili.com/video/BV1W8tdzEEVN) from @[gluttony-10](https://github.com/gluttony-10). Thanks for the contribution.
*   **[2025.08.12]** EchoMimicV3 can run on **16G VRAM** using [ComfyUI](https://github.com/smthemex/ComfyUI_EchoMimic). Thanks @[smthemex](https://github.com/smthemex) for the contribution.
*   **[2025.08.09]** Released [models](https://modelscope.cn/models/BadToBest/EchoMimicV3) on ModelScope.
*   **[2025.08.08]** Released [codes](https://github.com/antgroup/echomimic_v3) on GitHub and [models](https://huggingface.co/BadToBest/EchoMimicV3) on Huggingface.
*   **[2025.07.08]** [Paper](https://arxiv.org/abs/2507.03905) is in public on arxiv.

## Gallery

**(Include a few representative images or videos here.  Keep it concise, but visually engaging.)**

**(Example:  Insert a few key video outputs here, maybe 2-3 key examples.)**

## Quick Start

### Environment Setup

*   **Operating Systems:** Centos 7.2/Ubuntu 22.04
*   **CUDA Version:**  >= 12.1
*   **GPUs:** A100 (80G) / RTX4090D (24G) / V100 (16G)
*   **Python Version:** 3.10 / 3.11

### Installation

#### Windows

Download the [one-click installation package](https://pan.baidu.com/share/init?surl=cV7i2V0wF4exDtKjJrAUeA) (passport: glut).

#### Linux

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

| Model                                   | Download Link                                                                        | Notes                          |
| :-------------------------------------- | :----------------------------------------------------------------------------------- | :----------------------------- |
| Wan2.1-Fun-V1.1-1.3B-InP                | 🤗 [Huggingface](https://huggingface.co/alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP)          | Base model                     |
| wav2vec2-base                           | 🤗 [Huggingface](https://huggingface.co/facebook/wav2vec2-base-960h)                 | Audio encoder                  |
| EchoMimicV3-preview                     | 🤗 [Huggingface](https://huggingface.co/BadToBest/EchoMimicV3)                         | Our weights                    |
| EchoMimicV3-preview                     | 🤗 [ModelScope](https://modelscope.cn/models/BadToBest/EchoMimicV3)                         | Our weights                    |

**Model Structure:**

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

**Sample Data:** images, audios, masks and prompts are provided in `datasets/echomimicv3_demos`

#### Tips for Optimal Results

*   **Audio CFG:** Audio CFG `audio_guidance_scale` (2~3) for improved lip-sync while maintaining visual quality.
*   **Text CFG:** Text CFG `guidance_scale` (3~6) for more accurate prompt following.
*   **TeaCache:** Recommended range for `teacache_threshold`: between 0 and 0.1.
*   **Sampling Steps:** 5 steps for talking head, 15~25 steps for talking body.
*   **Long Video Generation:** Utilize Long Video CFG for videos longer than 138 frames.
*   **VRAM Optimization:** Reduce `partial_video_length` (e.g., to 81 or 65) for reduced VRAM usage.

##  Roadmap

| Status  | Milestone                                                                 |
| :------ | :------------------------------------------------------------------------ |
| ✅      | Release of EchoMimicV3 inference code on GitHub  |
| ✅      | EchoMimicV3-preview model on HuggingFace  |
| ✅      | EchoMimicV3-preview model on ModelScope  |
| ✅      | ModelScope Space   |
| 🚀      | 720P Pretrained models  |
| 🚀      | Release of EchoMimicV3 training code on GitHub  |

## EchoMimic Series

Explore the full EchoMimic family:

*   **EchoMimicV3:** 1.3B Parameters are All You Need for Unified Multi-Modal and Multi-Task Human Animation. [GitHub](https://github.com/antgroup/echomimic_v3)
*   **EchoMimicV2:** Towards Striking, Simplified, and Semi-Body Human Animation. [GitHub](https://github.com/antgroup/echomimic_v2)
*   **EchoMimicV1:** Lifelike Audio-Driven Portrait Animations through Editable Landmark Conditioning. [GitHub](https://github.com/antgroup/echomimic)

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

## Acknowledgements

*   Wan2.1: [https://github.com/Wan-Video/Wan2.1/](https://github.com/Wan-Video/Wan2.1/)
*   VideoX-Fun: [https://github.com/aigc-apps/VideoX-Fun/](https://github.com/aigc-apps/VideoX-Fun/)

## License

This project is licensed under the Apache 2.0 License.  See the LICENSE file for details.  You are responsible for your use of the generated content and must comply with all applicable laws and regulations.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=antgroup/echomimic_v3&type=Date)](https://www.star-history.com/#antgroup/echomimic_v3&Date)
```

Key improvements and explanations:

*   **SEO-Optimized Title and Introduction:** The title now uses the primary keyword ("EchoMimicV3") and a descriptive hook. The introduction concisely summarizes the project's value proposition.
*   **Clear Headings:**  Well-defined headings (Key Features, What's New, Gallery, Quick Start, Roadmap, etc.) improve readability and SEO.
*   **Bulleted Key Features:**  Key features are listed with bullet points for easy scanning and understanding.
*   **Concise Language:** Removed redundant phrases and focused on conveying essential information.
*   **Actionable Quick Start:**  The Quick Start section is clearly organized with environment setup, installation, model preparation, and inference instructions.
*   **Visual Emphasis:**  The gallery section is emphasized with a call to action to view image/video outputs.
*   **Up-to-date Information:** The "What's New" section is preserved and kept current.
*   **Call to Action:** Encourage users to see the outputs (visual).
*   **Simplified, Consistent Formatting:** Consistent use of bolding, code formatting, and bullet points.
*   **Comprehensive:** Keeps all the important information from the original README.
*   **License and Citation:** Included both the license and citation information.
*   **Star History:** Included the star history, adding to the SEO of the README.
*   **Removed redundant links:** Condensed links where possible.