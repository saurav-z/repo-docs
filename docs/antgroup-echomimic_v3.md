<!-- Improved README.md for EchoMimicV3 -->

# EchoMimicV3: Animate Humans with 1.3 Billion Parameters

EchoMimicV3 revolutionizes human animation by offering a unified, multi-modal, and multi-task approach, all within a compact 1.3 billion parameter model. Explore the future of animation with [the original EchoMimicV3 repo](https://github.com/antgroup/echomimic_v3).

<div align="center">
  <img src="asset/EchoMimicV3_logo.png.jpg" alt="EchoMimicV3 Logo" height="60">
</div>

**Key Features:**

*   **Unified Multi-Modal:** Seamlessly integrates text, audio, and visual inputs for rich and dynamic animations.
*   **Multi-Task Capabilities:**  Designed for a variety of animation tasks, from talking heads to full-body movements.
*   **Compact Model Size:** Achieve high-quality results with a streamlined 1.3B parameter model, making it accessible and efficient.
*   **Easy to Use:** Simple installation and inference with provided code and pre-trained models.
*   **Gradio Demo:** Interactive demo available on ModelScope for easy experimentation and exploration.

## What's New

*   **[2025.08.21]** 🔥 EchoMimicV3 Gradio demo launched on [ModelScope](https://modelscope.cn/studios/BadToBest/EchoMimicV3).
*   **[2025.08.12]** 🔥 Generate video with only **12G VRAM** using the provided [GradioUI](https://github.com/antgroup/echomimic_v3/blob/main/app_mm.py).  Check the [tutorial](https://www.bilibili.com/video/BV1W8tdzEEVN).
*   **[2025.08.12]** 🔥  Run EchoMimicV3 on **16G VRAM** with [ComfyUI](https://github.com/smthemex/ComfyUI_EchoMimic).
*   **[2025.08.09]** 🔥 Models released on [ModelScope](https://modelscope.cn/models/BadToBest/EchoMimicV3).
*   **[2025.08.08]** 🔥 Code released on GitHub and models released on [Hugging Face](https://huggingface.co/BadToBest/EchoMimicV3).
*   **[2025.07.08]** 🔥 Paper published on arXiv ([https://arxiv.org/abs/2507.03905](https://arxiv.org/abs/2507.03905)).

## Showcase & Examples

<div align="center">
  <img src="asset/echomimicv3.jpg" alt="EchoMimicV3 Showcase" height="400">
</div>

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

For more demo videos, please refer to the [project page](https://antgroup.github.io/ai/echomimic_v3/).

## Quick Start Guide

### System Requirements
*   **Operating System:** Centos 7.2 / Ubuntu 22.04
*   **CUDA:** >= 12.1
*   **GPUs:** A100(80G) / RTX4090D (24G) / V100(16G)
*   **Python:** 3.10 / 3.11

### Installation

####  Windows:

*   Use the [one-click installation package](https://pan.baidu.com/share/init?surl=cV7i2V0wF4exDtKjJrAUeA) (passport: glut) for a quick start with the quantized version.

#### Linux:

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

Download and place the necessary model files in the appropriate directory.

| Model                    | Download Link                                                                                                                              | Notes                                     |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------- |
| Wan2.1-Fun-V1.1-1.3B-InP | 🤗 [Hugging Face](https://huggingface.co/alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP)                                                             | Base model                                |
| wav2vec2-base            | 🤗 [Hugging Face](https://huggingface.co/facebook/wav2vec2-base-960h)                                                                      | Audio encoder                             |
| EchoMimicV3-preview      | 🤗 [Hugging Face](https://huggingface.co/BadToBest/EchoMimicV3)  /  🤗 [ModelScope](https://modelscope.cn/models/BadToBest/EchoMimicV3)     | EchoMimicV3 Weights                       |

The weights should be organized as follows:
```
./models/
├── Wan2.1-Fun-V1.1-1.3B-InP
├── wav2vec2-base-960h
└── transformer
    └── diffusion_pytorch_model.safetensors
```

### Inference

Run the `infer.py` script for basic inference:

```bash
python infer.py
```

For the GradioUI version:

```bash
python app_mm.py
```

**Sample data (images, audio, masks, prompts) are provided in the `datasets/echomimicv3_demos` directory.**

#### Inference Tips

*   **Audio CFG:**  `audio_guidance_scale` (2~3) - Increase for lip sync, decrease for visual quality.
*   **Text CFG:** `guidance_scale` (3~6) - Increase for prompt following, decrease for visual quality.
*   **TeaCache:** `teacache_threshold` (0~0.1)
*   **Sampling Steps:** 5 steps (talking head), 15-25 steps (talking body).
*   **Long Video Generation:** Use Long Video CFG for videos longer than 138 frames.
*   **Reduce VRAM:** Try `partial_video_length` to 81, 65, or lower.

## Roadmap

| Status   | Milestone                                                                        |
| :------- | :------------------------------------------------------------------------------- |
| ✅       | Release inference code on GitHub                                                  |
| ✅       | EchoMimicV3-preview model on Hugging Face                                        |
| ✅       | EchoMimicV3-preview model on ModelScope                                          |
| ✅       | ModelScope Space                                                                   |
| 🚀       | 720P Pretrained models                                                           |
| 🚀       | Release training code on GitHub                                                   |

## Related Projects

*   [EchoMimicV3](https://github.com/antgroup/echomimic_v3)
*   [EchoMimicV2](https://github.com/antgroup/echomimic_v2)
*   [EchoMimicV1](https://github.com/antgroup/echomimic)

## Citation

If you use our work, please cite our paper:

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

*   [Wan2.1](https://github.com/Wan-Video/Wan2.1/)
*   [VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun/)

## License

This project is licensed under the [Apache 2.0 License](LICENSE).

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=antgroup/echomimic_v3&type=Date)](https://www.star-history.com/#antgroup/echomimic_v3&Date)