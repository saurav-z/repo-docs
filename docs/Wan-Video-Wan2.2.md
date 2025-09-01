# Wan2.2: Unleashing Cinematic Video Generation with Open-Source AI

**Experience the future of video creation with Wan2.2, a cutting-edge open-source video generation model pushing the boundaries of cinematic quality. Check out the original repo at [https://github.com/Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2)**

## Key Features

*   **MoE Architecture:** Leveraging Mixture-of-Experts (MoE) for enhanced model capacity and efficiency.
*   **Cinematic Aesthetics:** Generate videos with precise control over lighting, composition, and color.
*   **Advanced Motion Generation:**  Trained on a significantly expanded dataset for improved generalization across diverse motions, semantics, and aesthetics.
*   **Efficient High-Definition Hybrid TI2V:**  Generate 720P videos at 24fps with a 5B model, optimized for consumer-grade GPUs.
*   **Text-to-Video, Image-to-Video, and Speech-to-Video Support:** Versatile generation capabilities with a focus on cinematic quality.
*   **Community-Driven:** Supports integration with ComfyUI and Diffusers.

## Latest News

*   **August 26, 2025:** Launched **Wan2.2-S2V-14B**, an audio-driven cinematic video generation model.
*   **July 28, 2025:** Released TI2V-5B model on HF Space and integrated into ComfyUI and Diffusers.

## Getting Started

### Installation

1.  Clone the repository:

    ```bash
    git clone https://github.com/Wan-Video/Wan2.2.git
    cd Wan2.2
    ```

2.  Install dependencies:

    ```bash
    # Ensure torch >= 2.4.0
    # If the installation of `flash_attn` fails, try installing the other packages first and install `flash_attn` last
    pip install -r requirements.txt
    ```

### Model Download

| Model             | Download Links                                                                                                                            | Description                                              |
| :---------------- | :---------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------- |
| T2V-A14B          | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B) 🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-T2V-A14B)         | Text-to-Video MoE model, supports 480P & 720P           |
| I2V-A14B          | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B) 🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-I2V-A14B)         | Image-to-Video MoE model, supports 480P & 720P           |
| TI2V-5B           | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B)  🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-TI2V-5B)          | High-compression VAE, T2V+I2V, supports 720P           |
| S2V-14B           | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-S2V-14B)  🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-S2V-14B)          | Speech-to-Video model, supports 480P & 720P           |

Download models using huggingface-cli or modelscope-cli, as described in the original README.

### Run Generation

Detailed instructions for running text-to-video, image-to-video, text-image-to-video, and speech-to-video generation can be found in the original README.  These include example code snippets and command-line arguments, along with explanations for prompt extension.

## Introduction of Wan2.2

Wan2.2 builds upon the foundation of Wan2.1, incorporating several key advancements:

*   **Mixture-of-Experts (MoE) Architecture:** The A14B model uses a MoE design with two experts. This increases the total parameters while maintaining inference cost.
*   **Efficient High-Definition Hybrid TI2V:** A 5B model enables efficient generation on consumer-grade GPUs.

## Performance and Comparisons

Wan2.2 is shown to outperform leading commercial models on the Wan-Bench 2.0.

## Citation

If you find our work helpful, please cite us:

```
@article{wan2025,
      title={Wan: Open and Advanced Large-Scale Video Generative Models},
      author={Team Wan and Ang Wang and Baole Ai and Bin Wen and Chaojie Mao and Chen-Wei Xie and Di Chen and Feiwu Yu and Haiming Zhao and Jianxiao Yang and Jianyuan Zeng and Jiayu Wang and Jingfeng Zhang and Jingren Zhou and Jinkai Wang and Jixuan Chen and Kai Zhu and Kang Zhao and Keyu Yan and Lianghua Huang and Mengyang Feng and Ningyi Zhang and Pandeng Li and Pingyu Wu and Ruihang Chu and Ruili Feng and Shiwei Zhang and Siyang Sun and Tao Fang and Tianxing Wang and Tianyi Gui and Tingyu Weng and Tong Shen and Wei Lin and Wei Wang and Wei Wang and Wenmeng Zhou and Wente Wang and Wenting Shen and Wenyuan Yu and Xianzhong Shi and Xiaoming Huang and Xin Xu and Yan Kou and Yangyu Lv and Yifei Li and Yijing Liu and Yiming Wang and Yingya Zhang and Yitong Huang and Yong Li and You Wu and Yu Liu and Yulin Pan and Yun Zheng and Yuntao Hong and Yupeng Shi and Yutong Feng and Zeyinzi Jiang and Zhen Han and Zhi-Fan Wu and Ziyu Liu},
      journal = {arXiv preprint arXiv:2503.20314},
      year={2025}
}
```

## License

Wan2.2 is licensed under the Apache 2.0 License.

## Acknowledgements

We thank the contributors to the [SD3](https://huggingface.co/stabilityai/stable-diffusion-3-medium), [Qwen](https://huggingface.co/Qwen), [umt5-xxl](https://huggingface.co/google/umt5-xxl), [diffusers](https://github.com/huggingface/diffusers) and [HuggingFace](https://huggingface.co) repositories, for their open research.

## Contact Us

Join our [Discord](https://discord.gg/AKNgpMK4Yj) or [WeChat groups](https://gw.alicdn.com/imgextra/i2/O1CN01tqjWFi1ByuyehkTSB_!!6000000000015-0-tps-611-1279.jpg)!