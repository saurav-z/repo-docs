# EchoMimicV3: Revolutionizing Human Animation with Just 1.3 Billion Parameters

**EchoMimicV3** offers a powerful, unified solution for multi-modal and multi-task human animation, achieving remarkable results with only 1.3 billion parameters.  Check out the original repo for the latest updates: [https://github.com/antgroup/echomimic_v3](https://github.com/antgroup/echomimic_v3)

## Key Features

*   **Unified Multi-Modal & Multi-Task:** A single model handles various animation tasks and input types.
*   **High-Quality Results:** Achieves striking and lifelike animations.
*   **Efficient Parameterization:** Uses a compact 1.3B parameter model for efficient deployment.
*   **Easy to Use:** Simple installation and quick inference steps.
*   **Comprehensive Resources:**  Includes pre-trained models, code, demos, and more.

## What's New
*   **[2025.08.21]** 🔥 EchoMimicV3 Gradio demo on [ModelScope](https://modelscope.cn/studios/BadToBest/EchoMimicV3) is live.
*   **[2025.08.12]** 🔥🚀 Generate videos with **12GB VRAM** using the provided [GradioUI](https://github.com/antgroup/echomimic_v3/blob/main/app_mm.py). Tutorial available [here](https://www.bilibili.com/video/BV1W8tdzEEVN).
*   **[2025.08.12]** 🔥 Run on **16GB VRAM** with [ComfyUI](https://github.com/smthemex/ComfyUI_EchoMimic).
*   **[2025.08.09]** 🔥 Models released on [ModelScope](https://modelscope.cn/models/BadToBest/EchoMimicV3).
*   **[2025.08.08]** 🔥 Code released on GitHub and models on [Hugging Face](https://huggingface.co/BadToBest/EchoMimicV3).
*   **[2025.07.08]** 🔥 Paper published on arXiv: [https://arxiv.org/abs/2507.03905](https://arxiv.org/abs/2507.03905)

## Gallery

[Include your image/video gallery here, formatted using HTML as above.]

## Quick Start

### Environment Setup

*   **Operating System:** Centos 7.2/Ubuntu 22.04 (tested)
*   **CUDA:** >= 12.1
*   **GPUs:** A100(80G) / RTX4090D (24G) / V100(16G) (tested)
*   **Python:** 3.10 / 3.11 (tested)

### Installation

*   **Windows:** Use the [one-click installation package](https://pan.baidu.com/share/init?surl=cV7i2V0wF4exDtKjJrAUeA) (passport: glut).
*   **Linux:**

    1.  Create Conda Environment:
        ```bash
        conda create -n echomimic_v3 python=3.10
        conda activate echomimic_v3
        ```

    2.  Install Dependencies:
        ```bash
        pip install -r requirements.txt
        ```

### Model Preparation

Download the necessary models. Refer to the table for download links. Organize your model files in the following directory structure:

```
./models/
├── Wan2.1-Fun-V1.1-1.3B-InP
├── wav2vec2-base-960h
└── transformer
    └── diffusion_pytorch_model.safetensors
```

| Models        |                       Download Link                                           |    Notes                      |
| --------------|-------------------------------------------------------------------------------|-------------------------------|
| Wan2.1-Fun-V1.1-1.3B-InP  |      🤗 [Huggingface](https://huggingface.co/alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP)       | Base model
| wav2vec2-base |      🤗 [Huggingface](https://huggingface.co/facebook/wav2vec2-base-960h)          | Audio encoder
| EchoMimicV3-preview      |      🤗 [Huggingface](https://huggingface.co/BadToBest/EchoMimicV3)              | Our weights
| EchoMimicV3-preview      |      🤗 [ModelScope](https://modelscope.cn/models/BadToBest/EchoMimicV3)              | Our weights

### Quick Inference

```bash
python infer.py
```

For Quantified GradioUI:
```bash
python app_mm.py
```

**Example data provided in `datasets/echomimicv3_demos`**

#### Tips for Optimal Results
*   **Audio CFG:** Use `audio_guidance_scale` between 2~3.
*   **Text CFG:** Use `guidance_scale` between 3~6.
*   **TeaCache:** Try `teacache_threshold` between 0~0.1.
*   **Sampling Steps:**  Use 5 steps for talking head and 15~25 steps for talking body.
*   **Long Video Generation:** Utilize Long Video CFG for videos longer than 138 frames.
*   **Reduce VRAM:** Adjust `partial_video_length` (e.g., to 81 or 65) to decrease VRAM usage.

##  Roadmap

| Status | Milestone                                                                |     
|:--------:|:-------------------------------------------------------------------------|
|    ✅    | The inference code of EchoMimicV3 meet everyone on GitHub   | 
|    ✅   | EchoMimicV3-preview model on HuggingFace | 
|    ✅   | EchoMimicV3-preview model on ModelScope | 
|    ✅  | ModelScope Space | 
|    🚀    | 720P Pretrained models | 
|    🚀    | The training code of EchoMimicV3 meet everyone on GitHub   | 


## Explore the EchoMimic Family

*   **EchoMimicV3:** [GitHub](https://github.com/antgroup/echomimic_v3)
*   **EchoMimicV2:** [GitHub](https://github.com/antgroup/echomimic_v2)
*   **EchoMimicV1:** [GitHub](https://github.com/antgroup/echomimic)

## Citation

```bibtex
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

This project is licensed under the Apache 2.0 License.  Please review the full license for details on usage and restrictions.

## Star History
[![Star History Chart](https://api.star-history.com/svg?repos=antgroup/echomimic_v3&type=Date)](https://www.star-history.com/#antgroup/echomimic_v3&Date)