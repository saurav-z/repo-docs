# EchoMimicV3: Revolutionizing Human Animation with 1.3B Parameters

EchoMimicV3 delivers state-of-the-art unified multi-modal and multi-task human animation using just 1.3 billion parameters. Access the code and models on [GitHub](https://github.com/antgroup/echomimic_v3).

## Key Features

*   **Unified Approach:** Combines multi-modal and multi-task capabilities for comprehensive human animation.
*   **Efficient Parameterization:** Achieves high-quality results with only 1.3B parameters, enabling efficient deployment.
*   **Multiple Deployment Options:** Supports deployment on Hugging Face, ModelScope, and via Gradio UI, and ComfyUI.
*   **Simplified Workflow:** Provides a streamlined quick-start guide and example usage.
*   **Community Contributions:** Actively incorporates community contributions like the Gradio UI and ComfyUI integration.

## What's New

*   **[2025.08.21]** 🔥 EchoMimicV3 gradio demo on [modelscope](https://modelscope.cn/studios/BadToBest/EchoMimicV3) is ready.
*   **[2025.08.12]** 🔥🚀 **12G VRAM is All YOU NEED to Generate Video**. Please use this [GradioUI](https://github.com/antgroup/echomimic_v3/blob/main/app_mm.py). Check the [tutorial](https://www.bilibili.com/video/BV1W8tdzEEVN) from @[gluttony-10](https://github.com/gluttony-10). Thanks for the contribution.
*   **[2025.08.12]** 🔥 EchoMimicV3 can run on **16G VRAM** using [ComfyUI](https://github.com/smthemex/ComfyUI_EchoMimic). Thanks @[smthemex](https://github.com/smthemex) for the contribution.
*   **[2025.08.09]** 🔥 We release our [models](https://modelscope.cn/models/BadToBest/EchoMimicV3) on ModelScope.
*   **[2025.08.08]** 🔥 We release our [codes](https://github.com/antgroup/echomimic_v3) on GitHub and [models](https://huggingface.co/BadToBest/EchoMimicV3) on Huggingface.
*   **[2025.07.08]** 🔥 Our [paper](https://arxiv.org/abs/2507.03905) is in public on arxiv.

## Gallery

[Include Images/Videos from the original README here, using HTML table and video tags - ensuring proper display]

## Quick Start

### Environment Setup

*   **Tested Systems:** Centos 7.2/Ubuntu 22.04, Cuda >= 12.1
*   **Supported GPUs:** A100(80G) / RTX4090D (24G) / V100(16G)
*   **Python Versions:** 3.10 / 3.11

### Installation

*   **Windows:** Use the [one-click installation package](https://pan.baidu.com/share/init?surl=cV7i2V0wF4exDtKjJrAUeA) (passport: glut).
*   **Linux:**

    1.  Create a Conda environment:
        ```bash
        conda create -n echomimic_v3 python=3.10
        conda activate echomimic_v3
        ```
    2.  Install dependencies:
        ```bash
        pip install -r requirements.txt
        ```

### Model Preparation

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

### Quick Inference

```bash
python infer.py
```

For Quantified GradioUI version:

```bash
python app_mm.py
```

**Example data available in `datasets/echomimicv3_demos`**

#### Tips

>   - Audio CFG: 2~3 is ideal.
>   - Text CFG: 3~6 is ideal.
>   - TeaCache: 0~0.1 is ideal.
>   - Sampling steps: 5 (talking head), 15~25 (talking body).
>   - Long video generation: Use Long Video CFG.
>   - Reduce VRAM: Try `partial_video_length` of 81, 65, or smaller.

## TODO List

| Status | Milestone                               |
| :----: | :-------------------------------------- |
|   ✅   | Inference code release                  |
|   ✅   | Model release on Hugging Face         |
|   ✅   | Model release on ModelScope           |
|   ✅   | ModelScope Space                    |
|   🚀   | 720P Pretrained models                |
|   🚀   | Training code release              |

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
*   Wan2.1: https://github.com/Wan-Video/Wan2.1/
*   VideoX-Fun: https://github.com/aigc-apps/VideoX-Fun/

## License

The models in this repository are licensed under the Apache 2.0 License.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=antgroup/echomimic_v3&type=Date)](https://www.star-history.com/#antgroup/echomimic_v3&Date)