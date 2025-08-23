# EchoMimicV3: Revolutionizing Human Animation with Just 1.3 Billion Parameters

EchoMimicV3 offers a groundbreaking approach to multi-modal and multi-task human animation, achieving impressive results with a surprisingly compact 1.3 billion parameter model. Explore the cutting-edge capabilities of this innovative technology on [GitHub](https://github.com/antgroup/echomimic_v3).

**Key Features:**

*   **Unified Multi-Modal Animation:** Seamlessly integrates various modalities for a cohesive animation experience.
*   **Multi-Task Capabilities:** Designed to handle diverse animation tasks efficiently.
*   **Compact Parameter Size:** Achieves state-of-the-art performance with only 1.3 billion parameters.
*   **Gradio Demo:** Explore the functionality through the [ModelScope](https://modelscope.cn/studios/BadToBest/EchoMimicV3) demo.
*   **ComfyUI Integration:** Runs on **16G VRAM** using [ComfyUI](https://github.com/smthemex/ComfyUI_EchoMimic).
*   **Model Availability:** Access pre-trained models on [Hugging Face](https://huggingface.co/BadToBest/EchoMimicV3) and [ModelScope](https://modelscope.cn/models/BadToBest/EchoMimicV3).

## 🚀 Updates

*   **[2025.08.21]** 🔥 EchoMimicV3 gradio demo on [modelscope](https://modelscope.cn/studios/BadToBest/EchoMimicV3) is ready.
*   **[2025.08.12]** 🔥🚀 **12G VRAM is All YOU NEED to Generate Video**. Please use this [GradioUI](https://github.com/antgroup/echomimic_v3/blob/main/app_mm.py). Check the [tutorial](https://www.bilibili.com/video/BV1W8tdzEEVN) from @[gluttony-10](https://github.com/gluttony-10). Thanks for the contribution.
*   **[2025.08.12]** 🔥 EchoMimicV3 can run on **16G VRAM** using [ComfyUI](https://github.com/smthemex/ComfyUI_EchoMimic). Thanks @[smthemex](https://github.com/smthemex) for the contribution.
*   **[2025.08.09]** 🔥 We release our [models](https://modelscope.cn/models/BadToBest/EchoMimicV3) on ModelScope.
*   **[2025.08.08]** 🔥 We release our [codes](https://github.com/antgroup/echomimic_v3) on GitHub and [models](https://huggingface.co/BadToBest/EchoMimicV3) on Huggingface.
*   **[2025.07.08]** 🔥 Our [paper](https://arxiv.org/abs/2507.03905) is in public on arxiv.

## 🖼️ Gallery

[Include a selection of the most compelling images and videos from the original README, showcasing the capabilities of EchoMimicV3.]

[Example: Add image previews with captions, and consider including video embeds for dynamic demonstrations. This makes the README more engaging.]

### Chinese Driven Audio

[Include video previews showcasing the Chinese driven audio, like the original README]

For more demo videos, please refer to the [project page](https://antgroup.github.io/ai/echomimic_v3/)

## 🚀 Quick Start

### 🛠️ Installation

*   **Windows:** Use the [one-click installation package](https://pan.baidu.com/share/init?surl=cV7i2V0wF4exDtKjJrAUeA) for a quick start.
*   **Linux:**
    1.  Create a conda environment:
        ```bash
        conda create -n echomimic_v3 python=3.10
        conda activate echomimic_v3
        ```
    2.  Install dependencies:
        ```bash
        pip install -r requirements.txt
        ```

### 🧱 Model Preparation

| Models                       |                       Download Link                                           |    Notes                      |
| -----------------------------|-------------------------------------------------------------------------------|-------------------------------|
| Wan2.1-Fun-V1.1-1.3B-InP  |      🤗 [Huggingface](https://huggingface.co/alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP)       | Base model
| wav2vec2-base |      🤗 [Huggingface](https://huggingface.co/facebook/wav2vec2-base-960h)          | Audio encoder
| EchoMimicV3-preview      |      🤗 [Huggingface](https://huggingface.co/BadToBest/EchoMimicV3)              | Our weights
| EchoMimicV3-preview      |      🤗 [ModelScope](https://modelscope.cn/models/BadToBest/EchoMimicV3)              | Our weights

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

For Quantified GradioUI version:
```bash
python app_mm.py
```
**images, audios, masks and prompts are provided in `datasets/echomimicv3_demos`**

#### Tips

>   -   **Audio CFG:** Adjust `audio_guidance_scale` (2~3) for lip-sync accuracy.
>   -   **Text CFG:** Optimize `guidance_scale` (3~6) for prompt following.
>   -   **TeaCache:** The optimal range for `teacache_threshold` is between 0~0.1.
>   -   **Sampling Steps:** Talking head: 5 steps, talking body: 15~25 steps.
>   -   **Long Video Generation:** Use Long Video CFG for videos exceeding 138 frames.
>   -   **Reduce VRAM:** Set `partial_video_length` to 81, 65, or smaller.

## 📝 TODO List

| Status | Milestone                                                                |
| :----: | :------------------------------------------------------------------------- |
|   ✅   | The inference code of EchoMimicV3 meet everyone on GitHub   |
|   ✅   | EchoMimicV3-preview model on HuggingFace |
|   ✅   | EchoMimicV3-preview model on ModelScope |
|   🚀   | ModelScope Space |
|   🚀   | Preview version Pretrained models trained on English and Chinese on ModelScope |
|   🚀   | 720P Pretrained models trained on English and Chinese on HuggingFace |
|   🚀   | 720P Pretrained models trained on English and Chinese on ModelScope |
|   🚀   | The training code of EchoMimicV3 meet everyone on GitHub |

## 🌐 EchoMimic Series

*   EchoMimicV3: [GitHub](https://github.com/antgroup/echomimic_v3)
*   EchoMimicV2: [GitHub](https://github.com/antgroup/echomimic_v2)
*   EchoMimicV1: [GitHub](https://github.com/antgroup/echomimic)

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

## 🔗 Reference

*   Wan2.1: [https://github.com/Wan-Video/Wan2.1/](https://github.com/Wan-Video/Wan2.1/)
*   VideoX-Fun: [https://github.com/aigc-apps/VideoX-Fun/](https://github.com/aigc-apps/VideoX-Fun/)

## 📜 License

This project is licensed under the [Apache 2.0 License](https://github.com/antgroup/echomimic_v3/blob/main/LICENSE).

## ✨ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=antgroup/echomimic_v3&type=Date)](https://www.star-history.com/#antgroup/echomimic_v3&Date)