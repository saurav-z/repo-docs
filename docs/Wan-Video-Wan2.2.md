# Wan2.2: Unleashing Advanced Video Generation with Open-Source Models

**Wan2.2** is a groundbreaking open-source project that advances video generation, allowing users to create stunning and realistic videos from text, images, and audio. Explore the cutting-edge capabilities of [Wan2.2](https://github.com/Wan-Video/Wan2.2) and revolutionize your creative workflow.

*   **Key Features:**

    *   ✅ **Mixture-of-Experts (MoE) Architecture:** Leverages a sophisticated MoE design for enhanced model capacity and efficient inference, resulting in higher-quality videos.
    *   ✅ **Cinematic-Level Aesthetics:** Incorporates curated aesthetic data and fine-grained control over lighting, composition, and color grading for professional-quality results.
    *   ✅ **Enhanced Motion Generation:** Trained on a substantially larger dataset, enabling more complex and dynamic motion generation across various scenarios.
    *   ✅ **Efficient High-Definition Hybrid TI2V:** Offers a high-compression 5B model (TI2V-5B) supporting 720P video generation at 24fps, compatible with consumer-grade GPUs.
    *   ✅ **Text-to-Video (T2V), Image-to-Video (I2V), Text-Image-to-Video (TI2V), and Speech-to-Video (S2V) Support:** Versatile models for creating videos from multiple input modalities.
*   [Learn More](https://github.com/Wan-Video/Wan2.2)

## Core Capabilities

Wan2.2 represents a significant advancement in video generation, offering a range of innovative features and capabilities:

### 1. Mixture-of-Experts (MoE) Architecture

*   **Enhanced Efficiency:** Employs a MoE architecture to increase model parameters while maintaining manageable inference costs.
*   **Specialized Experts:** The model uses a two-expert design tailored to the denoising process of diffusion models: a high-noise expert focuses on overall layout and a low-noise expert refines video details.
*   **Dynamic Activation:** Transitions between experts dynamically based on the signal-to-noise ratio (SNR) during the denoising steps.

### 2. High-Definition Hybrid TI2V

*   **Efficient Deployment:** Provides a 5B model (TI2V-5B) supported by a high-compression Wan2.2-VAE.
*   **Fast Generation:** Capable of generating 720P videos at 24fps, making it one of the fastest models available.
*   **Unified Framework:** Supports both text-to-video and image-to-video tasks within a single model.

### 3. Training Data and Performance

*   **Expanded Dataset:** Trained on a significantly larger dataset compared to Wan2.1.
*   **Superior Performance:** Achieves top performance compared to leading closed-source commercial models.

## Getting Started

### Installation

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/Wan-Video/Wan2.2.git
    cd Wan2.2
    ```

2.  **Install Dependencies:**

    ```bash
    # Ensure torch >= 2.4.0
    # If the installation of `flash_attn` fails, try installing the other packages first and install `flash_attn` last
    pip install -r requirements.txt
    ```

### Model Download

Download the necessary models from Hugging Face or ModelScope:

| Models              | Download Links                                                                                                                              | Description |
|--------------------|---------------------------------------------------------------------------------------------------------------------------------------------|-------------|
| T2V-A14B    | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B)    🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-T2V-A14B)    | Text-to-Video MoE model, supports 480P & 720P |
| I2V-A14B    | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B)    🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-I2V-A14B)    | Image-to-Video MoE model, supports 480P & 720P |
| TI2V-5B     | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B)     🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-TI2V-5B)     | High-compression VAE, T2V+I2V, supports 720P |
| S2V-14B     | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-S2V-14B)     🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-S2V-14B)     | Speech-to-Video model, supports 480P & 720P |

*   **Hugging Face CLI Example:**

    ```bash
    pip install "huggingface_hub[cli]"
    huggingface-cli download Wan-AI/Wan2.2-T2V-A14B --local-dir ./Wan2.2-T2V-A14B
    ```

*   **ModelScope CLI Example:**

    ```bash
    pip install modelscope
    modelscope download Wan-AI/Wan2.2-T2V-A14B --local_dir ./Wan2.2-T2V-A14B
    ```

### Running the Models

Detailed instructions are provided in the original README for running text-to-video, image-to-video, text-image-to-video, and speech-to-video generations. This includes:

*   Single-GPU and Multi-GPU inference options using FSDP + DeepSpeed Ulysses.
*   Prompt extension using Dashscope API or local Qwen models.
*   Examples for various generation tasks.

## Community Works and Integration

Wan2.2 has been integrated into various platforms and projects:

*   **ComfyUI Integration:** Available for enhanced user experience
*   **Diffusers Integration:** Supported for versatile use cases
*   **Third-Party Projects:** Check the original README for community projects like DiffSynth-Studio and Kijai's ComfyUI WanVideoWrapper.

## Additional Information

### Todo List

*   Ongoing development, including further ComfyUI and Diffusers integration.
*   For the latest updates on the project's development, check the original README.

### Citation

If you find this project useful, please cite our work:

```
@article{wan2025,
      title={Wan: Open and Advanced Large-Scale Video Generative Models}, 
      author={Team Wan and Ang Wang and Baole Ai and Bin Wen and Chaojie Mao and Chen-Wei Xie and Di Chen and Feiwu Yu and Haiming Zhao and Jianxiao Yang and Jianyuan Zeng and Jiayu Wang and Jingfeng Zhang and Jingren Zhou and Jinkai Wang and Jixuan Chen and Kai Zhu and Kang Zhao and Keyu Yan and Lianghua Huang and Mengyang Feng and Ningyi Zhang and Pandeng Li and Pingyu Wu and Ruihang Chu and Ruili Feng and Shiwei Zhang and Siyang Sun and Tao Fang and Tianxing Wang and Tianyi Gui and Tingyu Weng and Tong Shen and Wei Lin and Wei Wang and Wei Wang and Wenmeng Zhou and Wente Wang and Wenting Shen and Wenyuan Yu and Xianzhong Shi and Xiaoming Huang and Xin Xu and Yan Kou and Yangyu Lv and Yifei Li and Yijing Liu and Yiming Wang and Yingya Zhang and Yitong Huang and Yong Li and You Wu and Yu Liu and Yulin Pan and Yun Zheng and Yuntao Hong and Yupeng Shi and Yutong Feng and Zeyinzi Jiang and Zhen Han and Zhi-Fan Wu and Ziyu Liu},
      journal = {arXiv preprint arXiv:2503.20314},
      year={2025}
}
```

### License

Wan2.2 is released under the Apache 2.0 License. Please review the [LICENSE.txt](LICENSE.txt) for complete details on usage and limitations.

### Acknowledgements

The project acknowledges the open-source contributions from projects such as SD3, Qwen, umt5-xxl, diffusers, and HuggingFace.

### Contact Us

*   **Discord:** [Discord](https://discord.gg/AKNgpMK4Yj)
*   **WeChat:** [WeChat](https://gw.alicdn.com/imgextra/i2/O1CN01tqjWFi1ByuyehkTSB_!!6000000000015-0-tps-611-1279.jpg)

By leveraging its advanced features and models, **Wan2.2** gives users the tools to generate stunning, high-quality videos, paving the way for new creative possibilities.