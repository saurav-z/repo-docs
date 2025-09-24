# EchoMimicV3: Revolutionizing Human Animation with 1.3 Billion Parameters

EchoMimicV3 offers a groundbreaking approach to unified multi-modal and multi-task human animation, leveraging only 1.3 billion parameters for impressive results. [Explore the original repository](https://github.com/antgroup/echomimic_v3) for further details.

## Key Features

*   **Unified Approach:**  Handles multiple modalities (audio, text, etc.) and animation tasks with a single model.
*   **Efficient Parameterization:** Achieves high-quality results with a compact 1.3B parameter model.
*   **Multi-Platform Accessibility:** Models available on Hugging Face and ModelScope.
*   **Easy Deployment:** Provides a GradioUI demo on ModelScope and ComfyUI integration for ease of use.
*   **User-Friendly:** Offers quick start instructions and practical tips for optimal results, including adjustable parameters like Audio CFG, Text CFG and Sampling Steps.
*   **Community Resources:** Includes links to project pages, research papers, demos, and community discussions.

## What's New
*   **2024.08.21:** Gradio demo on ModelScope is now available.
*   **2024.08.12:** 12G VRAM is all you need to Generate Video with GradioUI and also compatible with 16G VRAM using ComfyUI.
*   **2024.08.09:** Models released on ModelScope.
*   **2024.08.08:** Codes released on GitHub and models on Huggingface.
*   **2024.07.08:** Paper released on arxiv.

## Showcase

[Include a gallery of videos/images here, similar to the original README. Make sure to include descriptive captions to help with SEO.  e.g., Video demonstrating EchoMimicV3 generating a talking head from audio input. ]

## Getting Started

### Environment Setup

*   **Tested Systems:** Centos 7.2/Ubuntu 22.04, Cuda >= 12.1
*   **Tested GPUs:** A100 (80G) / RTX4090D (24G) / V100 (16G)
*   **Tested Python Versions:** 3.10 / 3.11

### Installation

**Windows Users:**

*   Use the [one-click installation package](https://pan.baidu.com/share/init?surl=cV7i2V0wF4exDtKjJrAUeA) (passport: glut) for quick setup.

**Linux Users:**

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

Download the following pre-trained models and organize your directory as follows:

| Models        |                       Download Link                                           |    Notes                      |
| --------------|-------------------------------------------------------------------------------|-------------------------------|
| Wan2.1-Fun-V1.1-1.3B-InP  |      🤗 [Huggingface](https://huggingface.co/alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP)       | Base model
| wav2vec2-base |      🤗 [Huggingface](https://huggingface.co/facebook/wav2vec2-base-960h)          | Audio encoder
| EchoMimicV3-preview      |      🤗 [Huggingface](https://huggingface.co/BadToBest/EchoMimicV3)              | Our weights
| EchoMimicV3-preview      |      🤗 [ModelScope](https://modelscope.cn/models/BadToBest/EchoMimicV3)              | Our weights

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

**Sample Data:**
`images, audios, masks and prompts are provided in datasets/echomimicv3_demos`

### Tips for Optimal Results
*   **Audio CFG:** Adjust `audio_guidance_scale` (2~3 recommended).
*   **Text CFG:**  Adjust `guidance_scale` (3~6 recommended).
*   **TeaCache:** Adjust `teacache_threshold` (0~0.1 recommended).
*   **Sampling Steps:** Use 5 steps for talking head; 15-25 steps for talking body.
*   **Long Video Generation:** Use Long Video CFG.

## Roadmap

*   [ ] 720P Pretrained models
*   [ ] Training code release

## Related Projects

*   EchoMimicV3:  [GitHub](https://github.com/antgroup/echomimic_v3)
*   EchoMimicV2:  [GitHub](https://github.com/antgroup/echomimic_v2)
*   EchoMimicV1:  [GitHub](https://github.com/antgroup/echomimic)

## Citation

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

## References

*   Wan2.1: [https://github.com/Wan-Video/Wan2.1/](https://github.com/Wan-Video/Wan2.1/)
*   VideoX-Fun: [https://github.com/aigc-apps/VideoX-Fun/](https://github.com/aigc-apps/VideoX-Fun/)

## License

EchoMimicV3 is licensed under the Apache 2.0 License.  Please review the license for full details.  You are responsible for your use of the models and the generated content.

## Star History

[Insert the Star History Chart code here, ensuring it's correctly formatted to display.]

```
[![Star History Chart](https://api.star-history.com/svg?repos=antgroup/echomimic_v3&type=Date)](https://www.star-history.com/#antgroup/echomimic_v3&Date)
```