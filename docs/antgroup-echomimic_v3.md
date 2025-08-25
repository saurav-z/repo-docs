# EchoMimicV3: Unleash Unified Multi-Modal Human Animation with Just 1.3 Billion Parameters

**[Explore the EchoMimicV3 Repository on GitHub](https://github.com/antgroup/echomimic_v3)**

EchoMimicV3 revolutionizes human animation, offering a powerful, unified approach to multi-modal and multi-task animation with only 1.3B parameters. This innovative model enables high-quality video generation driven by audio, text, and more, all while maintaining efficiency and accessibility.

## Key Features

*   **Unified Multi-Modal Animation:** Generate videos from various inputs like audio, text prompts, and more.
*   **Efficient Performance:** Achieves impressive results with only 1.3 billion parameters, making it accessible.
*   **Multi-Task Capabilities:** Supports diverse animation tasks, including talking heads and body animation.
*   **Gradio and ComfyUI Support:** Easily generate videos with the provided Gradio demo and the ComfyUI implementation.
*   **Model Availability:**  Pre-trained models are available on Hugging Face and ModelScope.

## What's New

*   **[2025.08.21]** 🔥 EchoMimicV3 gradio demo on [modelscope](https://modelscope.cn/studios/BadToBest/EchoMimicV3) is ready.
*   **[2025.08.12]** 🔥🚀 **12G VRAM is All YOU NEED to Generate Video**. Please use this [GradioUI](https://github.com/antgroup/echomimic_v3/blob/main/app_mm.py). Check the [tutorial](https://www.bilibili.com/video/BV1W8tdzEEVN) from @[gluttony-10](https://github.com/gluttony-10). Thanks for the contribution.
*   **[2025.08.12]** 🔥 EchoMimicV3 can run on **16G VRAM** using [ComfyUI](https://github.com/smthemex/ComfyUI_EchoMimic). Thanks @[smthemex](https://github.com/smthemex) for the contribution.
*   **[2025.08.09]** 🔥 We release our [models](https://modelscope.cn/models/BadToBest/EchoMimicV3) on ModelScope.
*   **[2025.08.08]** 🔥 We release our [codes](https://github.com/antgroup/echomimic_v3) on GitHub and [models](https://huggingface.co/BadToBest/EchoMimicV3) on Huggingface.
*   **[2025.07.08]** 🔥 Our [paper](https://arxiv.org/abs/2507.03905) is in public on arxiv.

## Gallery

<!-- Example of how to use the gallery section -->
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

For more demo videos, please refer to the [project page](https://antgroup.github.io/ai/echomimic_v3/)

## Quick Start

### Environment Setup

*   **Tested System Environment:** Centos 7.2/Ubuntu 22.04, Cuda >= 12.1
*   **Tested GPUs:** A100 (80G) / RTX4090D (24G) / V100 (16G)
*   **Tested Python Version:** 3.10 / 3.11

### Installation

#### 🛠️ Installation for Windows

##### Please use the [one-click installation package](https://pan.baidu.com/share/init?surl=cV7i2V0wF4exDtKjJrAUeA) to get started quickly for Quantified version.

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

### Model Preparation

| Models                     | Download Link                                                                | Notes            |
| -------------------------- | ---------------------------------------------------------------------------- | ---------------- |
| Wan2.1-Fun-V1.1-1.3B-InP   | 🤗 [Huggingface](https://huggingface.co/alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP) | Base model       |
| wav2vec2-base              | 🤗 [Huggingface](https://huggingface.co/facebook/wav2vec2-base-960h)           | Audio encoder    |
| EchoMimicV3-preview        | 🤗 [Huggingface](https://huggingface.co/BadToBest/EchoMimicV3)                | Our weights      |
| EchoMimicV3-preview        | 🤗 [ModelScope](https://modelscope.cn/models/BadToBest/EchoMimicV3)                | Our weights      |

**Model Organization:**

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

**For Quantified GradioUI Version:**

```bash
python app_mm.py
```

**Example Data:**

*   Images, audios, masks, and prompts are provided in the `datasets/echomimicv3_demos` directory.

#### Tips

*   **Audio CFG:** Adjust the `audio_guidance_scale` between 2 and 3 for optimal lip-sync.
*   **Text CFG:** Use a `guidance_scale` between 3 and 6 for better prompt following.
*   **TeaCache:**  The optimal range for `teacache_threshold` is between 0~0.1.
*   **Sampling steps:** Use 5 steps for talking head, 15~25 steps for talking body.
*   **Long video generation:** To generate videos longer than 138 frames, use the Long Video CFG.
*   **VRAM optimization:** Try setting `partial_video_length` to 81, 65, or smaller.

## TODO List

| Status | Milestone                                                              |
| :----: | :--------------------------------------------------------------------- |
|   ✅    | The inference code of EchoMimicV3 meet everyone on GitHub            |
|   ✅   | EchoMimicV3-preview model on HuggingFace                           |
|   ✅   | EchoMimicV3-preview model on ModelScope                           |
|   🚀   | ModelScope Space                                                     |
|   🚀   | Preview version Pretrained models trained on English and Chinese on ModelScope |
|   🚀   | 720P Pretrained models trained on English and Chinese on HuggingFace  |
|   🚀   | 720P Pretrained models trained on English and Chinese on ModelScope  |
|   🚀   | The training code of EchoMimicV3 meet everyone on GitHub            |

## EchoMimic Series

*   EchoMimicV3: 1.3B Parameters are All You Need for Unified Multi-Modal and Multi-Task Human Animation. [GitHub](https://github.com/antgroup/echomimic_v3)
*   EchoMimicV2: Towards Striking, Simplified, and Semi-Body Human Animation. [GitHub](https://github.com/antgroup/echomimic_v2)
*   EchoMimicV1: Lifelike Audio-Driven Portrait Animations through Editable Landmark Conditioning. [GitHub](https://github.com/antgroup/echomimic)

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

*   Wan2.1: [https://github.com/Wan-Video/Wan2.1/](https://github.com/Wan-Video/Wan2.1/)
*   VideoX-Fun: [https://github.com/aigc-apps/VideoX-Fun/](https://github.com/aigc-apps/VideoX-Fun/)

## License

The models in this repository are licensed under the Apache 2.0 License.  You are responsible for your usage of the models, which must not involve sharing any content that violates applicable laws, causes harm, or spreads misinformation.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=antgroup/echomimic_v3&type=Date)](https://www.star-history.com/#antgroup/echomimic_v3&Date)