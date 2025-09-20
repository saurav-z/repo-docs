# EchoMimicV3: Unleash Realistic Human Animation with Just 1.3 Billion Parameters

EchoMimicV3 revolutionizes human animation by offering unified multi-modal and multi-task capabilities with an efficient 1.3B parameter model.  [Explore the original repository](https://github.com/antgroup/echomimic_v3) for cutting-edge animation techniques.

## Key Features:

*   **Unified Approach:** Handles both multi-modal and multi-task human animation within a single model.
*   **High Efficiency:** Achieves impressive results with only 1.3 billion parameters.
*   **Low VRAM Requirement:**  Generates video with as little as 12GB VRAM!
*   **Flexible Deployment:**  Works with both Gradio UI and ComfyUI for versatile use.
*   **Open Access:**  Models are available on Hugging Face and ModelScope.
*   **Fast & Easy to Use:**  Includes a quick start guide for immediate results.

## What's New:

*   **[2025.08.21]** 🔥 EchoMimicV3 gradio demo on [modelscope](https://modelscope.cn/studios/BadToBest/EchoMimicV3) is ready.
*   **[2025.08.12]** 🔥🚀 **12G VRAM is All YOU NEED to Generate Video**. Please use this [GradioUI](https://github.com/antgroup/echomimic_v3/blob/main/app_mm.py). Check the [tutorial](https://www.bilibili.com/video/BV1W8tdzEEVN) from @[gluttony-10](https://github.com/gluttony-10). Thanks for the contribution.
*   **[2025.08.12]** 🔥 EchoMimicV3 can run on **16G VRAM** using [ComfyUI](https://github.com/smthemex/ComfyUI_EchoMimic). Thanks @[smthemex](https://github.com/smthemex) for the contribution.
*   **[2025.08.09]** 🔥 We release our [models](https://modelscope.cn/models/BadToBest/EchoMimicV3) on ModelScope.
*   **[2025.08.08]** 🔥 We release our [codes](https://github.com/antgroup/echomimic_v3) on GitHub and [models](https://huggingface.co/BadToBest/EchoMimicV3) on Huggingface.
*   **[2025.07.08]** 🔥 Our [paper](https://arxiv.org/abs/2507.03905) is in public on arxiv.

##  Demo Gallery:

[Include images and videos from original README, keeping file names consistent].

_(Include the image and video HTML code here.  Use descriptive alt text for the images.)_

## Quick Start Guide:

### Prerequisites:

*   Tested on: Centos 7.2/Ubuntu 22.04, Cuda >= 12.1
*   Tested GPUs: A100(80G) / RTX4090D (24G) / V100(16G)
*   Tested Python Version: 3.10 / 3.11

### Installation:

#### Windows:

*   Use the [one-click installation package](https://pan.baidu.com/share/init?surl=cV7i2V0wF4exDtKjJrAUeA) (passport: glut).

#### Linux:

1.  Create a Conda environment:

    ```bash
    conda create -n echomimic_v3 python=3.10
    conda activate echomimic_v3
    ```

2.  Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

### Model Preparation:

Download the following models:

| Model                      | Download Link                                                                      | Notes                 |
| -------------------------- | ---------------------------------------------------------------------------------- | --------------------- |
| Wan2.1-Fun-V1.1-1.3B-InP  | [Huggingface](https://huggingface.co/alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP)       | Base Model            |
| wav2vec2-base              | [Huggingface](https://huggingface.co/facebook/wav2vec2-base-960h)                  | Audio Encoder         |
| EchoMimicV3-preview        | [Huggingface](https://huggingface.co/BadToBest/EchoMimicV3)                          | Our Weights           |
| EchoMimicV3-preview        | [ModelScope](https://modelscope.cn/models/BadToBest/EchoMimicV3)                      | Our Weights           |

Organize the weights in the following directory structure:

```
./models/
├── Wan2.1-Fun-V1.1-1.3B-InP
├── wav2vec2-base-960h
└── transformer
    └── diffusion_pytorch_model.safetensors
```

###  Run Inference:

```bash
python infer.py
```
For GradioUI version:
```bash
python app_mm.py
```

**Sample data (images, audios, masks, prompts) are located in the `datasets/echomimicv3_demos` directory.**

####  Inference Tips:

*   **Audio CFG:** Adjust `audio_guidance_scale` (2-3) for lip sync and visual quality.
*   **Text CFG:**  Use `guidance_scale` (3-6) for prompt following.
*   **TeaCache:**  `teacache_threshold` (0-0.1) for best results.
*   **Sampling Steps:** 5 steps for talking head, 15-25 steps for talking body.
*   **Long Video Generation:** Use Long Video CFG for videos longer than 138 frames.
*   **Reduce VRAM Usage:** Set `partial_video_length` to 81, 65 or smaller.

##  Upcoming Features (TODO List):

| Status | Milestone                                                                |     
|:--------:|:-------------------------------------------------------------------------|
|    ✅    | The inference code of EchoMimicV3 meet everyone on GitHub   | 
|    ✅   | EchoMimicV3-preview model on HuggingFace | 
|    ✅   | EchoMimicV3-preview model on ModelScope | 
|    ✅  | ModelScope Space | 
|    🚀    | 720P Pretrained models | 
|    🚀    | The training code of EchoMimicV3 meet everyone on GitHub   | 

## EchoMimic Series:

*   **EchoMimicV3:** (This Repository)
*   [EchoMimicV2: Towards Striking, Simplified, and Semi-Body Human Animation](https://github.com/antgroup/echomimic_v2)
*   [EchoMimicV1: Lifelike Audio-Driven Portrait Animations through Editable Landmark Conditioning](https://github.com/antgroup/echomimic)

## Citation:

```
@misc{meng2025echomimicv3,
  title={EchoMimicV3: 1.3B Parameters are All You Need for Unified Multi-Modal and Multi-Task Human Animation},
  author={Rang Meng, Yan Wang, Weipeng Wu, Ruobing Zheng, Yuming Li, Chenguang Ma},
  year={2025},
  eprint={2507.03905},
  archivePrefix={arXiv}
}
```

## References:

*   [Wan2.1](https://github.com/Wan-Video/Wan2.1/)
*   [VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun/)

## License:

This project is licensed under the Apache 2.0 License.

## Star History:

[Insert the star history chart generated by the provided link.]

```html
<img src="https://api.star-history.com/svg?repos=antgroup/echomimic_v3&type=Date" alt="Star History Chart"/>