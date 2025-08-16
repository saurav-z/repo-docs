# Wan2.2: Unleashing Advanced Video Generation with Open-Source Models

**Create stunning videos with Wan2.2, the cutting-edge open-source video generation model, pushing the boundaries of cinematic quality and efficiency.** Explore the power of [Wan2.2 on GitHub](https://github.com/Wan-Video/Wan2.2) and discover a new era of video creation.

<p align="center">
    <img src="assets/logo.png" width="400"/>
</p>

**Key Features:**

*   ✅ **Mixture-of-Experts (MoE) Architecture:** Leverages a sophisticated MoE architecture for enhanced model capacity while maintaining computational efficiency.
*   🎬 **Cinematic Aesthetics:** Designed with curated aesthetic data for precise control over lighting, composition, and color grading, enabling customizable cinematic styles.
*   🚀 **Superior Motion Generation:** Trained on extensive data to generate complex and realistic motions, surpassing existing open-source and closed-source models.
*   ⚡ **Efficient High-Definition Hybrid TI2V:** Features a 5B model with a 16x16x4 compression ratio, supporting both text-to-video and image-to-video generation at 720P and 24fps, making it a fast and accessible solution.

**What is Wan2.2?**

Wan2.2 is a major advancement in video generation, built upon the foundation of its predecessor, Wan2.1. This open-source model offers unparalleled performance and control over video creation. It incorporates groundbreaking innovations in architecture, training data, and compression techniques to deliver high-quality video output with exceptional efficiency.

**Key Innovations:**

*   **MoE Architecture:** The MoE architecture uses two expert models tailored to the denoising process of diffusion models, resulting in a model with 27B parameters but only 14B active parameters per step, keeping inference computation and GPU memory nearly unchanged.
*   **Enhanced Training Data:** Wan2.2 has been trained on significantly expanded datasets, enhancing its ability to generate intricate motion and visual styles.
*   **TI2V-5B Model:** The TI2V-5B model compresses video data, enabling fast 720P video generation on consumer-grade GPUs.

**Video Demos**

<div align="center">
  <video src="https://github.com/user-attachments/assets/b63bfa58-d5d7-4de6-a1a2-98970b06d9a7" width="70%" poster=""> </video>
</div>

**Latest News**

*   July 28, 2025: Released a [Hugging Face space](https://huggingface.co/spaces/Wan-AI/Wan-2.2-5B) for TI2V-5B model.
*   July 28, 2025: Wan2.2 integrated into [ComfyUI](https://docs.comfy.org/zh-CN/tutorials/video/wan/wan2_2).
*   July 28, 2025: Wan2.2 T2V, I2V and TI2V have been integrated into Diffusers ([T2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) | [I2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers) | [TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers)).
*   Released inference code and model weights for Wan2.2.

**Community Works**

If your research or project builds upon [**Wan2.1**](https://github.com/Wan-Video/Wan2.1) or [**Wan2.2**](https://github.com/Wan-Video/Wan2.2), and you would like more people to see it, please inform us.

*   [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) provides comprehensive support for Wan 2.2.
*   [Kijai's ComfyUI WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper) is an alternative implementation of Wan models for ComfyUI.

**Getting Started**

1.  **Installation:**
    ```bash
    git clone https://github.com/Wan-Video/Wan2.2.git
    cd Wan2.2
    pip install -r requirements.txt
    ```

2.  **Model Download:**

    | Model                 | Download Links                                                                                                                              | Description                                 |
    | --------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------- |
    | T2V-A14B              | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B)    🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-T2V-A14B)    | Text-to-Video MoE model, supports 480P & 720P |
    | I2V-A14B              | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B)    🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-I2V-A14B)    | Image-to-Video MoE model, supports 480P & 720P |
    | TI2V-5B               | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B)     🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-TI2V-5B)     | High-compression VAE, T2V+I2V, supports 720P |

    Use `huggingface-cli` or `modelscope-cli` to download the models:

    ```bash
    # Hugging Face
    pip install "huggingface_hub[cli]"
    huggingface-cli download Wan-AI/Wan2.2-T2V-A14B --local-dir ./Wan2.2-T2V-A14B

    # ModelScope
    pip install modelscope
    modelscope download Wan-AI/Wan2.2-T2V-A14B --local_dir ./Wan2.2-T2V-A14B
    ```

3.  **Run Text-to-Video Generation:**

    *   **(1) Without Prompt Extension**

        ```bash
        python generate.py  --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --offload_model True --convert_model_dtype --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
        ```

        *   **(2) Using Prompt Extension:**
            *   **Dashscope API:**

                ```bash
                DASH_API_KEY=your_key torchrun --nproc_per_node=8 generate.py  --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage" --use_prompt_extend --prompt_extend_method 'dashscope' --prompt_extend_target_lang 'zh'
                ```

            *   **Local Model:**

                ```bash
                torchrun --nproc_per_node=8 generate.py  --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage" --use_prompt_extend --prompt_extend_method 'local_qwen' --prompt_extend_target_lang 'zh'
                ```

4.  **Run Image-to-Video Generation:**

    ```bash
    python generate.py --task i2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-I2V-A14B --offload_model True --convert_model_dtype --image examples/i2v_input.JPG --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
    ```

5.  **Run Text-Image-to-Video Generation:**

    ```bash
    python generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ./Wan2.2-TI2V-5B --offload_model True --convert_model_dtype --t5_cpu --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage"
    ```

**Computational Efficiency**

*See computational efficiency tests in the provided image.*

**Introduction of Wan2.2**

*   **MoE Architecture**
    *   The MoE architecture uses two expert models tailored to the denoising process of diffusion models, resulting in a model with 27B parameters but only 14B active parameters per step, keeping inference computation and GPU memory nearly unchanged.

    *   **Efficient High-Definition Hybrid TI2V**
        *   High-compression design with a 5B model.
        *   Wan2.2-VAE achieves a 4x16x16 compression ratio.
        *   TI2V-5B can generate a 5-second 720P video in under 9 minutes on a single consumer-grade GPU.
        *   Supports both text-to-video and image-to-video tasks.

    *   **Comparisons to SOTAs**
        *   Results demonstrate that Wan2.2 achieves superior performance compared to leading models.

**Citation**

```
@article{wan2025,
      title={Wan: Open and Advanced Large-Scale Video Generative Models}, 
      author={Team Wan and Ang Wang and Baole Ai and Bin Wen and Chaojie Mao and Chen-Wei Xie and Di Chen and Feiwu Yu and Haiming Zhao and Jianxiao Yang and Jianyuan Zeng and Jiayu Wang and Jingfeng Zhang and Jingren Zhou and Jinkai Wang and Jixuan Chen and Kai Zhu and Kang Zhao and Keyu Yan and Lianghua Huang and Mengyang Feng and Ningyi Zhang and Pandeng Li and Pingyu Wu and Ruihang Chu and Ruili Feng and Shiwei Zhang and Siyang Sun and Tao Fang and Tianxing Wang and Tianyi Gui and Tingyu Weng and Tong Shen and Wei Lin and Wei Wang and Wei Wang and Wenmeng Zhou and Wente Wang and Wenting Shen and Wenyuan Yu and Xianzhong Shi and Xiaoming Huang and Xin Xu and Yan Kou and Yangyu Lv and Yifei Li and Yijing Liu and Yiming Wang and Yingya Zhang and Yitong Huang and Yong Li and You Wu and Yu Liu and Yulin Pan and Yun Zheng and Yuntao Hong and Yupeng Shi and Yutong Feng and Zeyinzi Jiang and Zhen Han and Zhi-Fan Wu and Ziyu Liu},
      journal = {arXiv preprint arXiv:2503.20314},
      year={2025}
}
```

**License**

Wan2.2 is licensed under the Apache 2.0 License.

**Acknowledgments**

Thanks to the contributors of SD3, Qwen, umt5-xxl, diffusers, and Hugging Face.

**Contact Us**

Join our [Discord](https://discord.gg/AKNgpMK4Yj) or [WeChat groups](https://gw.alicdn.com/imgextra/i2/O1CN01tqjWFi1ByuyehkTSB_!!6000000000015-0-tps-611-1279.jpg) for support and updates.