# Wan2.2: Unleashing Cinematic Video Generation (Official Repo)

**Wan2.2** is a cutting-edge, open-source video generation model that brings cinematic-quality video creation to everyone.  Built upon advanced techniques like Mixture-of-Experts (MoE) architecture and trained on a vast dataset, **Wan2.2** sets a new standard for video generation.  Explore the power of **Wan2.2** and create stunning videos with ease.  [**Check out the original repository on GitHub**](https://github.com/Wan-Video/Wan2.2).

**Key Features:**

*   👍 **Mixture-of-Experts (MoE) Architecture:** Efficiently scales model capacity without increasing inference cost, leading to higher quality video generation.
*   👍 **Cinematic Aesthetics:** Generate videos with meticulously curated aesthetics, including control over lighting, composition, color tones, and more.
*   👍 **Enhanced Motion and Detail:** Trained on an extensive dataset (+65.6% images, +83.2% videos), Wan2.2 excels at generating complex and detailed motions.
*   👍 **High-Definition Hybrid TI2V:** Experience Text-to-Video and Image-to-Video generation at 720P/24fps on consumer-grade GPUs with the highly efficient 5B model.
*   👍 **Speech-to-Video:** Transform audio inputs into stunning visuals with our new S2V-14B model.

## Latest News and Updates:

*   **August 26, 2025:** Released **Wan2.2-S2V-14B**, an audio-driven cinematic video generation model! ([Inference Code]([inference code](#run-speech-to-video-generation)), [Model Weights](#model-download), [Technical Report](https://humanaigc.github.io/wan-s2v-webpage/content/wan-s2v.pdf)) Try it out at [wan.video](https://wan.video/), [ModelScope Gradio](https://www.modelscope.cn/studios/Wan-AI/Wan2.2-S2V) or [HuggingFace Gradio](https://huggingface.co/spaces/Wan-AI/Wan2.2-S2V)!
*   **July 28, 2025:**
    *   HF space ([HF space](https://huggingface.co/spaces/Wan-AI/Wan-2.2-5B) using the TI2V-5B model.
    *   Wan2.2 integrated into ComfyUI ([CN](https://docs.comfy.org/zh-CN/tutorials/video/wan/wan2_2) | [EN](https://docs.comfy.org/tutorials/video/wan/wan2_2)).
    *   Wan2.2 T2V, I2V and TI2V integrated into Diffusers ([T2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) | [I2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers) | [TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers)).
    *   Released the inference code and model weights of **Wan2.2**.

## Community Works

Discover how researchers and developers are leveraging Wan2.2:

*   **DiffSynth-Studio:** Provides comprehensive support, including low-GPU-memory offload, FP8 quantization, and more.
*   **Kijai's ComfyUI WanVideoWrapper:** An alternative implementation for ComfyUI offering cutting-edge optimizations.

## Getting Started with Wan2.2

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

Choose from the following models:

| Model             | Download Links                                                                                                                              | Description                                                               |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| T2V-A14B          | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B)  🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-T2V-A14B)    | Text-to-Video MoE model, supports 480P & 720P                            |
| I2V-A14B          | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B)  🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-I2V-A14B)    | Image-to-Video MoE model, supports 480P & 720P                            |
| TI2V-5B           | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B)  🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-TI2V-5B)    | High-compression VAE, T2V+I2V, supports 720P                              |
| S2V-14B           | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-S2V-14B)  🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-S2V-14B)    | Speech-to-Video model, supports 480P & 720P                              |

>   **Note:** The TI2V-5B model supports 720P video generation at 24 FPS.

Use `huggingface-cli` or `modelscope-cli` to download the desired model:

```bash
# Example: Downloading T2V-A14B using huggingface-cli
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.2-T2V-A14B --local-dir ./Wan2.2-T2V-A14B

# Example: Downloading T2V-A14B using modelscope-cli
pip install modelscope
modelscope download Wan-AI/Wan2.2-T2V-A14B --local_dir ./Wan2.2-T2V-A14B
```

### Running Wan2.2 Generation

The repository provides comprehensive instructions and example code for running Text-to-Video, Image-to-Video, Text-Image-to-Video, and Speech-to-Video generation. Refer to the original README for detailed commands and parameters.

*   [Run Text-to-Video Generation](#run-text-to-video-generation)
*   [Run Image-to-Video Generation](#run-image-to-video-generation)
*   [Run Text-Image-to-Video Generation](#run-text-image-to-video-generation)
*   [Run Speech-to-Video Generation](#run-speech-to-video-generation)

### Computational Efficiency

See the provided table in the original README for GPU memory and processing time comparisons of Wan2.2 models.

## Introduction of Wan2.2

Wan2.2 represents a significant leap forward in video generation, building upon the foundation of Wan2.1 with innovations such as:

*   **Mixture-of-Experts (MoE) Architecture:**  Enhances model capacity while maintaining efficient inference.
*   **Efficient High-Definition Hybrid TI2V:**  Provides high-quality video at 720P/24fps on consumer hardware.

## Citation

If you find this work useful, please cite us:

```
@article{wan2025,
      title={Wan: Open and Advanced Large-Scale Video Generative Models},
      author={Team Wan and Ang Wang and Baole Ai and Bin Wen and Chaojie Mao and Chen-Wei Xie and Di Chen and Feiwu Yu and Haiming Zhao and Jianxiao Yang and Jianyuan Zeng and Jiayu Wang and Jingfeng Zhang and Jingren Zhou and Jinkai Wang and Jixuan Chen and Kai Zhu and Kang Zhao and Keyu Yan and Lianghua Huang and Mengyang Feng and Ningyi Zhang and Pandeng Li and Pingyu Wu and Ruihang Chu and Ruili Feng and Shiwei Zhang and Siyang Sun and Tao Fang and Tianxing Wang and Tianyi Gui and Tingyu Weng and Tong Shen and Wei Lin and Wei Wang and Wei Wang and Wenmeng Zhou and Wente Wang and Wenting Shen and Wenyuan Yu and Xianzhong Shi and Xiaoming Huang and Xin Xu and Yan Kou and Yangyu Lv and Yifei Li and Yijing Liu and Yiming Wang and Yingya Zhang and Yitong Huang and Yong Li and You Wu and Yu Liu and Yulin Pan and Yun Zheng and Yuntao Hong and Yupeng Shi and Yutong Feng and Zeyinzi Jiang and Zhen Han and Zhi-Fan Wu and Ziyu Liu},
      journal = {arXiv preprint arXiv:2503.20314},
      year={2025}
}
```

## License Agreement

Wan2.2 is licensed under the Apache 2.0 License.

## Acknowledgements

The project acknowledges contributions from SD3, Qwen, umt5-xxl, diffusers, and HuggingFace.

## Contact Us

Join our [Discord](https://discord.gg/AKNgpMK4Yj) or [WeChat groups](https://gw.alicdn.com/imgextra/i2/O1CN01tqjWFi1ByuyehkTSB_!!6000000000015-0-tps-611-1279.jpg) to connect with the research and product teams.