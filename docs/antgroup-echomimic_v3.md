# EchoMimicV3: Revolutionizing Human Animation with Just 1.3 Billion Parameters

**EchoMimicV3** offers a groundbreaking approach to unified multi-modal and multi-task human animation, leveraging a compact 1.3B parameter model for impressive results.  Explore the cutting-edge technology behind this project on the [original GitHub repository](https://github.com/antgroup/echomimic_v3).

## Key Features

*   **Unified Multi-Modal & Multi-Task:**  Achieve seamless human animation across various modalities and tasks.
*   **Compact & Efficient:**  Experience high-quality animation with a model size of only 1.3 billion parameters, optimizing for efficiency.
*   **Model Availability:** Pre-trained models available on [Hugging Face](https://huggingface.co/BadToBest/EchoMimicV3) and [ModelScope](https://modelscope.cn/models/BadToBest/EchoMimicV3).
*   **Gradio Demo:** Try out an interactive demonstration on [ModelScope](https://modelscope.cn/studios/BadToBest/EchoMimicV3).
*   **VRAM Optimization:** Supports running on 12GB VRAM using a Gradio UI, and on 16GB VRAM with ComfyUI.
*   **Community Contributions:**  Benefit from community contributions like ComfyUI integration and tutorials, making the project easier to use.
*   **Comprehensive Tutorials:** Explore detailed quick start guides and tips for optimal results.

## Updates

*   **[2025.08.21]** 🔥 EchoMimicV3 gradio demo on [modelscope](https://modelscope.cn/studios/BadToBest/EchoMimicV3) is ready.
*   **[2025.08.12]** 🔥🚀 **12G VRAM is All YOU NEED to Generate Video**. Please use this [GradioUI](https://github.com/antgroup/echomimic_v3/blob/main/app_mm.py). Check the [tutorial](https://www.bilibili.com/video/BV1W8tdzEEVN) from @[gluttony-10](https://github.com/gluttony-10). Thanks for the contribution.
*   **[2025.08.12]** 🔥 EchoMimicV3 can run on **16G VRAM** using [ComfyUI](https://github.com/smthemex/ComfyUI_EchoMimic). Thanks @[smthemex](https://github.com/smthemex) for the contribution.
*   **[2025.08.09]** 🔥 We release our [models](https://modelscope.cn/models/BadToBest/EchoMimicV3) on ModelScope.
*   **[2025.08.08]** 🔥 We release our [codes](https://github.com/antgroup/echomimic_v3) on GitHub and [models](https://huggingface.co/BadToBest/EchoMimicV3) on Huggingface.
*   **[2025.07.08]** 🔥 Our [paper](https://arxiv.org/abs/2507.03905) is in public on arxiv.


## Gallery

[Include the same images/videos as the original README, formatted for markdown.]

## Quick Start

### Environment Setup

*   **Tested System Environment:** Centos 7.2/Ubuntu 22.04, Cuda >= 12.1
*   **Tested GPUs:** A100(80G) / RTX4090D (24G) / V100(16G)
*   **Tested Python Version:** 3.10 / 3.11

### 🛠️Installation for Windows

##### Please use the [one-click installation package](https://pan.baidu.com/share/init?surl=cV7i2V0wF4exDtKjJrAUeA) (passport: glut) to get started quickly for Quantified version.

### 🛠️Installation for Linux

#### 1. Create a conda environment

```bash
conda create -n echomimic_v3 python=3.10
conda activate echomimic_v3
```

#### 2. Other dependencies

```bash
pip install -r requirements.txt
```

### 🧱Model Preparation

| Models        |                       Download Link                                           |    Notes                      |
| --------------|-------------------------------------------------------------------------------|-------------------------------|
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

> - Audio CFG: Audio CFG `audio_guidance_scale` works optimally between 2~3. Increase the audio CFG value for better lip synchronization, while decreasing the audio CFG value can improve the visual quality.
> - Text CFG: Text CFG `guidance_scale` works optimally between 3~6. Increase the text CFG value for better prompt following, while decreasing the text CFG value can improve the visual quality.
> - TeaCache: The optimal range for `teacache_threshold` is between 0~0.1.
> - Sampling steps: 5 steps for talking head, 15~25 steps for talking body. 
> - ​Long video generation: If you want to generate a video longer than 138 frames, you can use Long Video CFG.
> - Try setting `partial_video_length` to 81, 65 or smaller to reduce VRAM usage.

## 📝 TODO List

| Status | Milestone                                                                |     
|:--------:|:-------------------------------------------------------------------------|
|    ✅    | The inference code of EchoMimicV3 meet everyone on GitHub   | 
|    ✅   | EchoMimicV3-preview model on HuggingFace | 
|    ✅   | EchoMimicV3-preview model on ModelScope | 
|    ✅  | ModelScope Space | 
|    🚀    | 720P Pretrained models | 
|    🚀    | The training code of EchoMimicV3 meet everyone on GitHub   | 

## &#x1F680; EchoMimic Series

Explore the evolution of human animation with the EchoMimic series:

*   **EchoMimicV3:** 1.3B Parameters are All You Need for Unified Multi-Modal and Multi-Task Human Animation. [GitHub](https://github.com/antgroup/echomimic_v3)
*   EchoMimicV2: Towards Striking, Simplified, and Semi-Body Human Animation. [GitHub](https://github.com/antgroup/echomimic_v2)
*   EchoMimicV1: Lifelike Audio-Driven Portrait Animations through Editable Landmark Conditioning. [GitHub](https://github.com/antgroup/echomimic)

## &#x1F4D2; Citation

If you find our work useful for your research, please consider citing the paper:

```
@misc{meng2025echomimicv3,
  title={EchoMimicV3: 1.3B Parameters are All You Need for Unified Multi-Modal and Multi-Task Human Animation},
  author={Rang Meng, Yan Wang, Weipeng Wu, Ruobing Zheng, Yuming Li, Chenguang Ma},
  year={2025},
  eprint={2507.03905},
  archivePrefix={arXiv}
}
```

## Reference

*   Wan2.1: [https://github.com/Wan-Video/Wan2.1/](https://github.com/Wan-Video/Wan2.1/)
*   VideoX-Fun: [https://github.com/aigc-apps/VideoX-Fun/](https://github.com/aigc-apps/VideoX-Fun/)

## 📜 License

The models in this repository are licensed under the Apache 2.0 License. We claim no rights over the your generated contents, 
granting you the freedom to use them while ensuring that your usage complies with the provisions of this license. 
You are fully accountable for your use of the models, which must not involve sharing any content that violates applicable laws, 
causes harm to individuals or groups, disseminates personal information intended for harm, spreads misinformation, or targets vulnerable populations. 

## &#x1F31F; Star History

[![Star History Chart](https://api.star-history.com/svg?repos=antgroup/echomimic_v3&type=Date)](https://www.star-history.com/#antgroup/echomimic_v3&Date)