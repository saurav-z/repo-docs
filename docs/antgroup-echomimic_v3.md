# EchoMimicV3: Revolutionizing Human Animation with a 1.3B Parameter Model

EchoMimicV3 is a cutting-edge AI model from Ant Group that leverages just 1.3 billion parameters to deliver unified, multi-modal, and multi-task human animation capabilities. **Experience the future of animation by exploring EchoMimicV3, a powerful and efficient solution for creating stunning visual content.** Access the original repository [here](https://github.com/antgroup/echomimic_v3).

## Key Features

*   **Unified Multi-Modal & Multi-Task:** Enables seamless integration of various inputs (audio, text) to generate diverse animation outputs.
*   **High-Quality Animation:** Produces lifelike and engaging animations with impressive visual fidelity.
*   **Efficient Architecture:** Achieves state-of-the-art results with a compact 1.3B parameter model, optimizing for both performance and resource usage.
*   **Versatile Applications:** Suitable for a wide range of applications, including talking heads, full-body animation, and more.
*   **Ease of Use:** Provides quick start guides and readily available models, making it accessible for both researchers and developers.

## What's New

*   **[2025.08.21]** 🔥 EchoMimicV3 Gradio demo now available on [ModelScope](https://modelscope.cn/studios/BadToBest/EchoMimicV3)!
*   **[2025.08.12]** 🔥 Generate videos with just **12GB of VRAM**! Try the [GradioUI](https://github.com/antgroup/echomimic_v3/blob/main/app_mm.py) and check out the tutorial from @[gluttony-10](https://github.com/gluttony-10)!
*   **[2025.08.12]** 🔥 Run EchoMimicV3 on **16GB VRAM** with [ComfyUI](https://github.com/smthemex/ComfyUI_EchoMimic) thanks to @[smthemex](https://github.com/smthemex)!
*   **[2025.08.09]** 🔥 Models released on [ModelScope](https://modelscope.cn/models/BadToBest/EchoMimicV3).
*   **[2025.08.08]** 🔥 Codes released on GitHub and models released on [Hugging Face](https://huggingface.co/BadToBest/EchoMimicV3).
*   **[2025.07.08]** 🔥 Paper released on arXiv: [https://arxiv.org/abs/2507.03905](https://arxiv.org/abs/2507.03905).

## Gallery

**(Include the images and videos from the original README here)**

## Quick Start

### Environment Setup

*   **Tested Systems:** Centos 7.2/Ubuntu 22.04, Cuda >= 12.1
*   **Tested GPUs:** A100(80G) / RTX4090D (24G) / V100(16G)
*   **Tested Python Versions:** 3.10 / 3.11

### Installation

*   **Windows:** Use the [one-click installation package](https://pan.baidu.com/share/init?surl=cV7i2V0wF4exDtKjJrAUeA) (passport: glut).
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

### Model Preparation

**(Include the table of models with download links)**

*   The weights are organized in the following structure:

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

For the GradioUI version:

```bash
python app_mm.py
```

**(Include Tip Section)**

## TODO List

**(Include the TODO table from the original README)**

## Related Projects

*   EchoMimicV2: [GitHub](https://github.com/antgroup/echomimic_v2)
*   EchoMimicV1: [GitHub](https://github.com/antgroup/echomimic)

## Citation

**(Include the citation information from the original README)**

## References

**(Include the References from the original README)**

## License

**(Include the license information from the original README)**

## Star History

**(Include the Star History Chart image from the original README)**