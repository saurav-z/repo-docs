# EchoMimicV3: Animate Humans with Just 1.3B Parameters!

EchoMimicV3 revolutionizes human animation, offering unified multi-modal and multi-task capabilities with a compact 1.3 billion parameter model. Explore the cutting-edge technology behind this exciting project and see how you can generate high-quality animated videos.  **[Check out the original repository](https://github.com/antgroup/echomimic_v3) for more details and to contribute!**

## ✨ Key Features

*   **Unified Multi-Modal Animation:**  Handles various input modalities including audio and text.
*   **Multi-Task Capabilities:** Excels at tasks like talking head generation and full-body animation.
*   **Compact Model Size:** Achieves impressive results with only 1.3B parameters, making it efficient for various hardware.
*   **Easy to Use:**  Quick start guide and comprehensive documentation get you up and running quickly.
*   **Open Source:**  Benefit from the open-source nature of the project and contribute to its development.
*   **Multiple Deployment Options:** Ready to use on Hugging Face, ModelScope, and other platforms.
*   **Gradio Demo:** Interactive demo for easy experimentation on ModelScope.

## 🚀 Updates & Recent Developments

*   **[2025.08.21]** Gradio demo is now available on [ModelScope](https://modelscope.cn/studios/BadToBest/EchoMimicV3)!
*   **[2025.08.12]** Generate videos with only 12G VRAM using the [GradioUI](https://github.com/antgroup/echomimic_v3/blob/main/app_mm.py). Check out the tutorial from @[gluttony-10](https://github.com/gluttony-10).
*   **[2025.08.12]** Support for [ComfyUI](https://github.com/smthemex/ComfyUI_EchoMimic) which runs on 16G VRAM is now available. Thanks @[smthemex](https://github.com/smthemex).
*   **[2025.08.09]** Models released on [ModelScope](https://modelscope.cn/models/BadToBest/EchoMimicV3).
*   **[2025.08.08]** Code released on GitHub and [models](https://huggingface.co/BadToBest/EchoMimicV3) on Hugging Face.
*   **[2025.07.08]** Paper available on arXiv:  [https://arxiv.org/abs/2507.03905](https://arxiv.org/abs/2507.03905)

## 🖼️ Gallery of Animated Examples

[Include Gallery of Animated Examples with alt text, consider adding captions.]

*   [Example video 1](video1.mp4) (alt text: A person talking and moving their lips.)
*   [Example video 2](video2.mp4) (alt text: A person full body dancing.)
*   [Example video 3](video3.mp4) (alt text: Animated person reading out loud.)

## 🎬 Demo Videos

[Include demo videos.]

## 🛠️ Quick Start

### Prerequisites
*   **Operating System:** Centos 7.2/Ubuntu 22.04
*   **CUDA:** \>= 12.1
*   **GPUs:**  A100(80G) / RTX4090D (24G) / V100(16G)
*   **Python:** 3.10 / 3.11

### 📦 Installation

**Windows:** Use the [one-click installation package](https://pan.baidu.com/share/init?surl=cV7i2V0wF4exDtKjJrAUeA) (passport: glut).

**Linux:**

1.  Create a Conda environment:
    ```bash
    conda create -n echomimic_v3 python=3.10
    conda activate echomimic_v3
    ```
2.  Install Dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### 💾 Model Preparation

Download the following models:

| Models        |                       Download Link                                           |    Notes                      |
| --------------|-------------------------------------------------------------------------------|-------------------------------|
| Wan2.1-Fun-V1.1-1.3B-InP  |      🤗 [Huggingface](https://huggingface.co/alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP)       | Base model
| wav2vec2-base |      🤗 [Huggingface](https://huggingface.co/facebook/wav2vec2-base-960h)          | Audio encoder
| EchoMimicV3-preview      |      🤗 [Huggingface](https://huggingface.co/BadToBest/EchoMimicV3)              | Our weights
| EchoMimicV3-preview      |      🤗 [ModelScope](https://modelscope.cn/models/BadToBest/EchoMimicV3)              | Our weights

Organize the downloaded weights as follows:

```
./models/
├── Wan2.1-Fun-V1.1-1.3B-InP
├── wav2vec2-base-960h
└── transformer
    └── diffusion_pytorch_model.safetensors
```

### 🏃 Quick Inference

```bash
python infer.py
```
For Quantified GradioUI version:
```bash
python app_mm.py
```
**Images, audios, masks and prompts are provided in `datasets/echomimicv3_demos`**

#### Tips

*   **Audio CFG:**  `audio_guidance_scale` (2-3 for optimal lip sync).
*   **Text CFG:** `guidance_scale` (3-6).
*   **TeaCache:** Use `teacache_threshold` in the range of 0-0.1.
*   **Sampling Steps:** Talking head (5 steps), talking body (15-25 steps).
*   **Long Video Generation:** Use Long Video CFG.
*   **Reduce VRAM:** Set `partial_video_length` to 81, 65, or smaller.

## 📝 TODO List

[Use a table to show TODO list.]

| Status | Milestone                                                                |     
|:--------:|:-------------------------------------------------------------------------|
|    ✅    | The inference code of EchoMimicV3 meet everyone on GitHub   | 
|    ✅   | EchoMimicV3-preview model on HuggingFace | 
|    ✅   | EchoMimicV3-preview model on ModelScope | 
|    ✅  | ModelScope Space | 
|    🚀    | 720P Pretrained models | 
|    🚀    | The training code of EchoMimicV3 meet everyone on GitHub   | 

## 🔗 EchoMimic Series

*   [EchoMimicV3](https://github.com/antgroup/echomimic_v3): 1.3B Parameters are All You Need for Unified Multi-Modal and Multi-Task Human Animation.
*   [EchoMimicV2](https://github.com/antgroup/echomimic_v2): Towards Striking, Simplified, and Semi-Body Human Animation.
*   [EchoMimicV1](https://github.com/antgroup/echomimic): Lifelike Audio-Driven Portrait Animations through Editable Landmark Conditioning.

## 📚 Citation

If you use EchoMimicV3 in your research, please cite our paper:

```
@misc{meng2025echomimicv3,
  title={EchoMimicV3: 1.3B Parameters are All You Need for Unified Multi-Modal and Multi-Task Human Animation},
  author={Rang Meng, Yan Wang, Weipeng Wu, Ruobing Zheng, Yuming Li, Chenguang Ma},
  year={2025},
  eprint={2507.03905},
  archivePrefix={arXiv}
}
```

## 🤝 References

*   Wan2.1: [https://github.com/Wan-Video/Wan2.1/](https://github.com/Wan-Video/Wan2.1/)
*   VideoX-Fun: [https://github.com/aigc-apps/VideoX-Fun/](https://github.com/aigc-apps/VideoX-Fun/)

## 📜 License

EchoMimicV3 is licensed under the Apache 2.0 License.  You are free to use the models, but you are responsible for your use and must comply with the license terms.  Avoid any use that violates laws, harms individuals, or distributes harmful content.

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=antgroup/echomimic_v3&type=Date)](https://www.star-history.com/#antgroup/echomimic_v3&Date)