# Wan2.2: Revolutionizing Video Generation with State-of-the-Art AI

**Wan2.2** is a cutting-edge, open-source video generation model, offering unparalleled cinematic quality and efficiency for both research and practical applications.  Explore the technology and download the models on [GitHub](https://github.com/Wan-Video/Wan2.2).

[<img src="assets/logo.png" width="400" alt="Wan2.2 Logo"/>](https://github.com/Wan-Video/Wan2.2)

**Key Features & Highlights:**

*   **Mixture-of-Experts (MoE) Architecture:**  Leverages a novel MoE design to boost model capacity without increasing computational costs, achieving superior video quality.
*   **Cinematic Aesthetics:**  Trained on curated aesthetic data with detailed labels, enabling precise control over lighting, composition, and color grading for professional-grade video outputs.
*   **Advanced Motion Generation:**  Trained on significantly expanded datasets (+65.6% images, +83.2% videos) enhancing the model's ability to generate complex, realistic, and diverse motions, achieving top performance.
*   **Efficient High-Definition Hybrid TI2V:**  Includes an open-source TI2V-5B model with a 16x16x4 compression ratio, supporting 720P video generation at 24fps on consumer-grade GPUs (e.g., 4090), for fast and accessible video creation.
*   **Text-to-Video, Image-to-Video, and Text-Image-to-Video Capabilities:** Offers flexible video generation from text prompts, images, or combined text-image inputs.

**Recent Updates:**

*   **July 28, 2025:**  TI2V-5B model now available on [Hugging Face Spaces](https://huggingface.co/spaces/Wan-AI/Wan-2.2-5B)!
*   **July 28, 2025:**  Wan2.2 integrated into ComfyUI ( [CN](https://docs.comfy.org/zh-CN/tutorials/video/wan/wan2_2) | [EN](https://docs.comfy.org/tutorials/video/wan/wan2_2))
*   **July 28, 2025:**  Diffusers integration of Wan2.2 ([T2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) | [I2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers) | [TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers)).
*   **July 28, 2025:** Inference code and model weights of Wan2.2 have been released.

**Video Demos**

<div align="center">
  <video src="https://github.com/user-attachments/assets/b63bfa58-d5d7-4de6-a1a2-98970b06d9a7" width="70%" poster=""> </video>
</div>

**Community Contributions:**

*   [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio): Comprehensive support for Wan 2.2, including low-GPU-memory offload and FP8 quantization.
*   [Kijai's ComfyUI WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper): Alternative implementation of Wan models for ComfyUI with optimizations.

**Getting Started**

**1. Installation:**

```bash
git clone https://github.com/Wan-Video/Wan2.2.git
cd Wan2.2
pip install -r requirements.txt
```

**2. Model Downloads:**

| Model          | Download Links                                                                                                                            | Description                                         |
| :------------- | :---------------------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------- |
| T2V-A14B       | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B) 🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-T2V-A14B)        | Text-to-Video, 480P & 720P                         |
| I2V-A14B       | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B) 🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-I2V-A14B)        | Image-to-Video, 480P & 720P                         |
| TI2V-5B        | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B)  🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-TI2V-5B)         | Text-Image-to-Video, High Compression, 720P, 24FPS |

*   Download models using Hugging Face CLI:
    ```bash
    pip install "huggingface_hub[cli]"
    huggingface-cli download Wan-AI/Wan2.2-T2V-A14B --local-dir ./Wan2.2-T2V-A14B
    ```
*   Download models using ModelScope CLI:
    ```bash
    pip install modelscope
    modelscope download Wan-AI/Wan2.2-T2V-A14B --local_dir ./Wan2.2-T2V-A14B
    ```

**3. Run Generation**

Detailed instructions for running text-to-video, image-to-video, and text-image-to-video generation, including single and multi-GPU setups, with and without prompt extension can be found in the [original README](https://github.com/Wan-Video/Wan2.2).

**Computational Efficiency**

Benchmark results for performance across different GPUs are available in the [original README](https://github.com/Wan-Video/Wan2.2).

**Technical Details:**

*   **Mixture-of-Experts (MoE) Architecture:**  A core innovation in Wan2.2, utilizing a MoE design for enhanced generation quality.  Refer to the [original README](https://github.com/Wan-Video/Wan2.2) for details on the expert architecture and validation.
*   **Efficient High-Definition Hybrid TI2V:**  The TI2V-5B model utilizes a high-compression Wan2.2-VAE with a compression ratio of 4x16x16 for efficient 720P video generation.
*   **Performance:**  Wan2.2 demonstrates superior performance compared to commercial closed-source models.  See the [original README](https://github.com/Wan-Video/Wan2.2) for performance comparisons.

**Citation:**
```
@article{wan2025,
      title={Wan: Open and Advanced Large-Scale Video Generative Models},
      author={Team Wan and Ang Wang and Baole Ai and Bin Wen and Chaojie Mao and Chen-Wei Xie and Di Chen and Feiwu Yu and Haiming Zhao and Jianxiao Yang and Jianyuan Zeng and Jiayu Wang and Jingfeng Zhang and Jingren Zhou and Jinkai Wang and Jixuan Chen and Kai Zhu and Kang Zhao and Keyu Yan and Lianghua Huang and Mengyang Feng and Ningyi Zhang and Pandeng Li and Pingyu Wu and Ruihang Chu and Ruili Feng and Shiwei Zhang and Siyang Sun and Tao Fang and Tianxing Wang and Tianyi Gui and Tingyu Weng and Tong Shen and Wei Lin and Wei Wang and Wei Wang and Wenmeng Zhou and Wente Wang and Wenting Shen and Wenyuan Yu and Xianzhong Shi and Xiaoming Huang and Xin Xu and Yan Kou and Yangyu Lv and Yifei Li and Yijing Liu and Yiming Wang and Yingya Zhang and Yitong Huang and Yong Li and You Wu and Yu Liu and Yulin Pan and Yun Zheng and Yuntao Hong and Yupeng Shi and Yutong Feng and Zeyinzi Jiang and Zhen Han and Zhi-Fan Wu and Ziyu Liu},
      journal = {arXiv preprint arXiv:2503.20314},
      year={2025}
}
```

**License:**

Wan2.2 is released under the Apache 2.0 License.  See [LICENSE.txt](LICENSE.txt) for details.

**Contact & Support:**

*   Join the [Discord](https://discord.gg/AKNgpMK4Yj) for community support and updates.
*   Contact the team via [WeChat](https://gw.alicdn.com/imgextra/i2/O1CN01tqjWFi1ByuyehkTSB_!!6000000000015-0-tps-611-1279.jpg).

**Acknowledgements:**
We thank the contributors to [SD3](https://huggingface.co/stabilityai/stable-diffusion-3-medium), [Qwen](https://huggingface.co/Qwen), [umt5-xxl](https://huggingface.co/google/umt5-xxl), [diffusers](https://github.com/huggingface/diffusers) and [HuggingFace](https://huggingface.co) for their open research.