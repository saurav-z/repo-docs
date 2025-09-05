# EchoMimicV3: Revolutionizing Human Animation with 1.3 Billion Parameters

**EchoMimicV3** offers a groundbreaking approach to unified multi-modal and multi-task human animation, all while being remarkably efficient.  Explore the project on [GitHub](https://github.com/antgroup/echomimic_v3).

## Key Features:

*   **Unified Approach:** Achieve multi-modal and multi-task human animation with a single model.
*   **Efficient Architecture:** Leverages only 1.3 billion parameters, making it resource-friendly.
*   **Versatile Capabilities:**  Generate high-quality animations from audio, text, and more.
*   **Gradio Demo:** Interactive demonstration available on [ModelScope](https://modelscope.cn/studios/BadToBest/EchoMimicV3).
*   **ComfyUI Support:**  Integrates with ComfyUI for expanded functionality ([smthemex/ComfyUI_EchoMimic](https://github.com/smthemex/ComfyUI_EchoMimic)).
*   **Community Resources:** Access pre-trained models on Hugging Face and ModelScope, along with a supportive community.

## What's New:

*   **[2025.08.21]** 🔥 EchoMimicV3 Gradio demo launched on [ModelScope](https://modelscope.cn/studios/BadToBest/EchoMimicV3).
*   **[2025.08.12]** 🔥🚀 Generate video with just 12G VRAM.  Check out the [GradioUI](https://github.com/antgroup/echomimic_v3/blob/main/app_mm.py) and [tutorial](https://www.bilibili.com/video/BV1W8tdzEEVN).
*   **[2025.08.12]** 🔥 Runs on 16G VRAM using [ComfyUI](https://github.com/smthemex/ComfyUI_EchoMimic).
*   **[2025.08.09]** 🔥 Models released on [ModelScope](https://modelscope.cn/models/BadToBest/EchoMimicV3).
*   **[2025.08.08]** 🔥 Codes released on GitHub and models on [HuggingFace](https://huggingface.co/BadToBest/EchoMimicV3).
*   **[2025.07.08]** 🔥 Paper available on arXiv ([https://arxiv.org/abs/2507.03905](https://arxiv.org/abs/2507.03905)).

## Gallery

<!-- Replace these with the existing image and video tags, optimized for size and loading -->
<p align="center">
  <img src="asset/echomimicv3.jpg"  height=400 alt="EchoMimicV3 Example">
</p>

<!-- Videos should be optimized for web display -->
<table class="center">
<tr>
    <td width=50% style="border: none">
        <video controls loop muted="false"  src="https://github.com/user-attachments/assets/f33edb30-66b1-484b-8be0-a5df20a44f3b"  style="width:100%;"></video>
    </td>
        <td width=50% style="border: none">
        <video controls loop muted="false" src="https://github.com/user-attachments/assets/056105d8-47cd-4a78-8ec2-328ceaf95a5a"  style="width:100%;"></video>
    </td>
</tr>
</table>

### Chinese Driven Audio

<table class="center">
<tr>
    <td width=25% style="border: none">
        <video controls loop muted="false" src="https://github.com/user-attachments/assets/fc1ebae4-b571-43eb-a13a-7d6d05b74082"  style="width:100%;"></video>
    </td>
    <td width=25% style="border: none">
        <video controls loop muted="false" src="https://github.com/user-attachments/assets/54607cc7-944c-4529-9bef-715862ba330d"  style="width:100%;"></video>
    </td>
    <td width=25% style="border: none">
        <video controls loop muted="false" src="https://github.com/user-attachments/assets/4d1de999-cce2-47ab-89ed-f2fa11c838fe"  style="width:100%;"></video>
    </td>
    <td width=25% style="border: none">
        <video controls loop muted="false" src="https://github.com/user-attachments/assets/41e701cc-ac3e-4dd8-b94c-859261f17344"  style="width:100%;"></video>
    </td>
</tr>
</table>

For more video demos, visit the [project page](https://antgroup.github.io/ai/echomimic_v3/).

## Quick Start

### Environment Setup

*   **Operating Systems:** Centos 7.2/Ubuntu 22.04
*   **CUDA:** >= 12.1
*   **GPUs:** Tested on A100 (80G), RTX 4090D (24G), V100 (16G)
*   **Python:** 3.10 / 3.11

### Installation

#### For Windows:

*   Use the [one-click installation package](https://pan.baidu.com/share/init?surl=cV7i2V0wF4exDtKjJrAUeA) (passport: glut) for a quick start (Quantified version).

#### For Linux:

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

| Model                               | Download Link                                                                 | Notes                    |
| ----------------------------------- | ----------------------------------------------------------------------------- | ------------------------ |
| Wan2.1-Fun-V1.1-1.3B-InP             | 🤗 [Huggingface](https://huggingface.co/alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP)   | Base model                |
| wav2vec2-base                        | 🤗 [Huggingface](https://huggingface.co/facebook/wav2vec2-base-960h)          | Audio encoder            |
| EchoMimicV3-preview                  | 🤗 [Huggingface](https://huggingface.co/BadToBest/EchoMimicV3)                 | Our weights               |
| EchoMimicV3-preview                  | 🤗 [ModelScope](https://modelscope.cn/models/BadToBest/EchoMimicV3)             | Our weights               |

*The weights should be organized as follows:*

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

*   **Input Data:** Images, audios, masks, and prompts are located in `datasets/echomimicv3_demos`.

#### Tips:

*   **Audio CFG:**  `audio_guidance_scale` (2~3) for optimal lip-sync.
*   **Text CFG:** `guidance_scale` (3~6) for prompt following.
*   **TeaCache:** `teacache_threshold` (0~0.1)
*   **Sampling steps:** Talking head (5 steps), talking body (15-25 steps).
*   **Long Video Generation:** Use Long Video CFG for videos exceeding 138 frames.
*   **Reduce VRAM:** Set `partial_video_length` to 81, 65 or smaller.

## TODO List

| Status | Milestone                                                                |
| :-----: | :------------------------------------------------------------------------- |
|    ✅    | EchoMimicV3 inference code on GitHub                                 |
|    ✅   | EchoMimicV3-preview model on HuggingFace                             |
|    ✅   | EchoMimicV3-preview model on ModelScope                             |
|    ✅  | ModelScope Space                                                       |
|    🚀    | 720P Pretrained models                                                   |
|    🚀    | EchoMimicV3 training code on GitHub                                  |

## EchoMimic Series

*   **EchoMimicV3:** Unified Multi-Modal and Multi-Task Human Animation (1.3B Parameters) - [GitHub](https://github.com/antgroup/echomimic_v3)
*   EchoMimicV2: Towards Striking, Simplified, and Semi-Body Human Animation - [GitHub](https://github.com/antgroup/echomimic_v2)
*   EchoMimicV1: Lifelike Audio-Driven Portrait Animations through Editable Landmark Conditioning - [GitHub](https://github.com/antgroup/echomimic)

## Citation

If you use EchoMimicV3 in your research, please cite the following:

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

*   Wan2.1: [https://github.com/Wan-Video/Wan2.1/](https://github.com/Wan-Video/Wan2.1/)
*   VideoX-Fun: [https://github.com/aigc-apps/VideoX-Fun/](https://github.com/aigc-apps/VideoX-Fun/)

## License

This project is licensed under the Apache 2.0 License.  You are free to use the generated content as long as your use complies with the license. You are fully responsible for your use of the models.

## Star History
[![Star History Chart](https://api.star-history.com/svg?repos=antgroup/echomimic_v3&type=Date)](https://www.star-history.com/#antgroup/echomimic_v3&Date)
```
Key improvements and explanations:

*   **SEO Optimization:** Keywords like "human animation," "multi-modal," "multi-task," and "efficient" are included to improve searchability.  The title includes keywords.
*   **Concise Hook:** The opening sentence immediately grabs attention and explains the core benefit.
*   **Clear Headings:**  Uses H2 headings to structure the information and make it scannable.
*   **Bulleted Key Features:**  Highlights the main advantages in an easy-to-read format.
*   **Concise Language:** Avoids overly verbose sentences.
*   **Up-to-Date Information:** Includes the latest updates, including links to new demos and models.
*   **Optimized Image and Video Integration:** Recommends size optimization for images and videos. Added `alt` tags to images for SEO.  Links to the correct video and image assets.
*   **Emphasis on Community & Resources:**  Highlights the availability of models, demos, and community support.
*   **Complete Structure:** Provides clear instructions for setup, model preparation, and quick inference.
*   **Clear Citation and License Information:**  Includes essential information for users.
*   **Removed Irrelevant Content:** Removed the chat links.
*   **Improved Formatting:**  Uses consistent formatting for better readability.
*   **Clearer Quick Start:** Refined the quick start guide for ease of use.
*   **Improved Table Formatting:** Refined the table formatting for enhanced presentation.
*   **Concise TODO List:**  Made the TODO list more focused.
*   **Includes Star History:** Adds a chart to show project activity.

This revised README is more informative, user-friendly, and SEO-optimized.  It efficiently conveys the value of EchoMimicV3 and guides users through getting started.