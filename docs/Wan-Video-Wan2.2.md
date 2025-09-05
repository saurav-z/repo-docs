# Wan2.2: Generate Stunning Videos with Advanced AI

**Unleash the power of AI to create high-quality videos with Wan2.2, the next-generation video generation model.**  ([Original Repo](https://github.com/Wan-Video/Wan2.2))

[<img src="assets/logo.png" width="400" alt="Wan2.2 Logo">](https://github.com/Wan-Video/Wan2.2)

**Key Features:**

*   **Mixture-of-Experts (MoE) Architecture:**  Leverages a cutting-edge MoE design for enhanced model capacity and performance, optimizing video denoising across different stages for improved quality.
*   **Cinematic-Level Aesthetics:**  Built upon meticulously curated aesthetic data, providing precise control over lighting, composition, and color grading to generate videos with customizable cinematic styles.
*   **Enhanced Motion Generation:** Trained on a significantly expanded dataset (+65.6% images, +83.2% videos), resulting in superior generalization across diverse motions, semantics, and aesthetics, achieving top performance.
*   **Efficient High-Definition Hybrid TI2V:**  Includes a 5B model with advanced Wan2.2-VAE, supporting text-to-video and image-to-video generation at 720P (24fps) on consumer-grade GPUs (e.g., 4090).
*   **Speech-to-Video Support:** Supports speech to video with an example available.

**Latest Updates:**

*   **Wan2.2-S2V-14B:**  Audio-driven cinematic video generation model released, including inference code, model weights, and technical report. Try it out on [wan.video](https://wan.video/), [ModelScope Gradio](https://www.modelscope.cn/studios/Wan-AI/Wan2.2-S2V) or [HuggingFace Gradio](https://huggingface.co/spaces/Wan-AI/Wan2.2-S2V)! (August 26, 2025)
*   **HF space for TI2V-5B:** Created a [HF space](https://huggingface.co/spaces/Wan-AI/Wan-2.2-5B) using the TI2V-5B model. (July 28, 2025)
*   **ComfyUI Integration:** Wan2.2 integrated into ComfyUI ([CN](https://docs.comfy.org/zh-CN/tutorials/video/wan/wan2_2) | [EN](https://docs.comfy.org/tutorials/video/wan/wan2_2)). (July 28, 2025)
*   **Diffusers Integration:** Wan2.2's T2V, I2V and TI2V integrated into Diffusers ([T2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) | [I2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers) | [TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers)). (July 28, 2025)
*   **Inference Code & Model Weights:** Inference code and model weights for Wan2.2 released.

**Community Works:**

*   [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)
*   [Kijai's ComfyUI WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper)

**Getting Started:**

*   **Installation:**
    ```bash
    git clone https://github.com/Wan-Video/Wan2.2.git
    cd Wan2.2
    pip install -r requirements.txt
    ```

*   **Model Downloads:**

    | Model         | Download Link                                                                                                   | Description                                   |
    | ------------- | --------------------------------------------------------------------------------------------------------------- | --------------------------------------------- |
    | T2V-A14B      | [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B)  /  [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-T2V-A14B)      | Text-to-Video (480P & 720P)                 |
    | I2V-A14B      | [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B)  /  [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-I2V-A14B)      | Image-to-Video (480P & 720P)                |
    | TI2V-5B       | [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B)  /  [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-TI2V-5B)       | Text-Image-to-Video (720P)                 |
    | S2V-14B       | [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-S2V-14B)  /  [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-S2V-14B)       | Speech-to-Video (480P & 720P)                |

    *   **Download using `huggingface-cli`:**
        ```bash
        pip install "huggingface_hub[cli]"
        huggingface-cli download Wan-AI/Wan2.2-T2V-A14B --local-dir ./Wan2.2-T2V-A14B
        ```
    *   **Download using `modelscope-cli`:**
        ```bash
        pip install modelscope
        modelscope download Wan-AI/Wan2.2-T2V-A14B --local_dir ./Wan2.2-T2V-A14B
        ```

*   **Example Usage:**  See the original README for detailed generation scripts, including Text-to-Video, Image-to-Video, Text-Image-to-Video, and Speech-to-Video examples.

**Citation:**
```
@article{wan2025,
      title={Wan: Open and Advanced Large-Scale Video Generative Models}, 
      author={Team Wan and Ang Wang and Baole Ai and Bin Wen and Chaojie Mao and Chen-Wei Xie and Di Chen and Feiwu Yu and Haiming Zhao and Jianxiao Yang and Jianyuan Zeng and Jiayu Wang and Jingfeng Zhang and Jingren Zhou and Jinkai Wang and Jixuan Chen and Kai Zhu and Kang Zhao and Keyu Yan and Lianghua Huang and Mengyang Feng and Ningyi Zhang and Pandeng Li and Pingyu Wu and Ruihang Chu and Ruili Feng and Shiwei Zhang and Siyang Sun and Tao Fang and Tianxing Wang and Tianyi Gui and Tingyu Weng and Tong Shen and Wei Lin and Wei Wang and Wei Wang and Wenmeng Zhou and Wente Wang and Wenting Shen and Wenyuan Yu and Xianzhong Shi and Xiaoming Huang and Xin Xu and Yan Kou and Yangyu Lv and Yifei Li and Yijing Liu and Yiming Wang and Yingya Zhang and Yitong Huang and Yong Li and You Wu and Yu Liu and Yulin Pan and Yun Zheng and Yuntao Hong and Yupeng Shi and Yutong Feng and Zeyinzi Jiang and Zhen Han and Zhi-Fan Wu and Ziyu Liu},
      journal = {arXiv preprint arXiv:2503.20314},
      year={2025}
}
```

**License:**  [Apache 2.0 License](LICENSE.txt)

**Acknowledgments:**  Thanks to the contributors of SD3, Qwen, umt5-xxl, diffusers and HuggingFace for their open research.

**Contact:**  Join our [Discord](https://discord.gg/AKNgpMK4Yj) or [WeChat groups](https://gw.alicdn.com/imgextra/i2/O1CN01tqjWFi1ByuyehkTSB_!!6000000000015-0-tps-611-1279.jpg) for support.