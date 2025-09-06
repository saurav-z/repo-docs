# Wan2.2: Unleashing Advanced Video Generation with Open-Source Models

**Generate stunning videos with Wan2.2, the open-source video generation model that brings cinematic-level aesthetics and complex motion to your fingertips.** Developed by [Wan-Video](https://github.com/Wan-Video/Wan2.2), this powerful model empowers you to create high-quality videos from text, images, and even audio!

[**View Original Repo**](https://github.com/Wan-Video/Wan2.2)

---

**Key Features of Wan2.2:**

*   ✨ **Mixture-of-Experts (MoE) Architecture:**  Leverages a cutting-edge MoE architecture for enhanced model capacity, leading to improved performance and efficiency.
*   🎬 **Cinematic-Level Aesthetics:** Incorporates carefully curated aesthetic data with detailed labels, enabling the generation of videos with customizable cinematic styles.
*   🚀 **Complex Motion Generation:** Trained on a significantly larger dataset, resulting in enhanced performance in a variety of dimensions like motions and aesthetics.
*   ⚡ **Efficient High-Definition Generation:** Provides a 5B model (TI2V-5B) for both text-to-video and image-to-video generation at 720P resolution (24fps), running on consumer-grade GPUs.
*   🎶 **Speech-to-Video Capabilities:** Introduce Wan2.2-S2V, an audio-driven video generation model for cinematic video generation.

---

## Latest News

*   **[Aug 26, 2025]:** Release of **[Wan2.2-S2V-14B](https://humanaigc.github.io/wan-s2v-webpage)**, an audio-driven cinematic video generation model.
*   **[Jul 28, 2025]:** Integration into ComfyUI.
*   **[Jul 28, 2025]:** Integration of T2V, I2V and TI2V models into Diffusers.
*   **[Jul 28, 2025]:** Release of inference code and model weights.
*   **[Sep 5, 2025]:** Text-to-speech synthesis support with [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) for Speech-to-Video generation task.

## Community Works

Explore community contributions and integrations with Wan2.2:

*   [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)
*   [Kijai's ComfyUI WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper)

---

## Quickstart: Run Wan2.2

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Wan-Video/Wan2.2.git
    cd Wan2.2
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    pip install -r requirements_s2v.txt  # For Speech-to-Video generation
    ```

### Model Download

Download the models using the following links:

| Model       | Description                    | Hugging Face                                                              | ModelScope                                                                 |
| ----------- | ------------------------------ | ------------------------------------------------------------------------ | -------------------------------------------------------------------------- |
| T2V-A14B    | Text-to-Video (MoE)           | [Hugging Face](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B)            | [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-T2V-A14B)          |
| I2V-A14B    | Image-to-Video (MoE)          | [Hugging Face](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B)            | [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-I2V-A14B)          |
| TI2V-5B     | Text-Image-to-Video           | [Hugging Face](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B)             | [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-TI2V-5B)           |
| S2V-14B     | Speech-to-Video                | [Hugging Face](https://huggingface.co/Wan-AI/Wan2.2-S2V-14B)             | [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-S2V-14B)           |

#### Download using Hugging Face Hub:
```bash
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.2-T2V-A14B --local-dir ./Wan2.2-T2V-A14B
```

#### Download using ModelScope:
```bash
pip install modelscope
modelscope download Wan-AI/Wan2.2-T2V-A14B --local_dir ./Wan2.2-T2V-A14B
```

### Run Generation Examples

Detailed instructions and example commands are provided in the original README. Here's a glimpse:

*   **Text-to-Video:**

    ```bash
    python generate.py --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --offload_model True --convert_model_dtype --prompt "Your Text Prompt Here"
    ```

*   **Image-to-Video:**

    ```bash
    python generate.py --task i2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-I2V-A14B --image examples/i2v_input.JPG --prompt "Your Text Prompt Here"
    ```

*   **Text-Image-to-Video:**

    ```bash
    python generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ./Wan2.2-TI2V-5B --image examples/i2v_input.JPG --prompt "Your Text Prompt Here"
    ```

*   **Speech-to-Video:**

    ```bash
    python generate.py  --task s2v-14B --size 1024*704 --ckpt_dir ./Wan2.2-S2V-14B/ --image "examples/i2v_input.JPG" --audio "examples/talk.wav" --prompt "Your Text Prompt Here"
    ```

> Consult the original README for comprehensive options and advanced configurations.

---

## Computational Efficiency
View the Computational Efficiency images in the original README for insights into performance on various GPUs.

---

## Citation

```
@article{wan2025,
      title={Wan: Open and Advanced Large-Scale Video Generative Models},
      author={Team Wan and Ang Wang and Baole Ai and Bin Wen and Chaojie Mao and Chen-Wei Xie and Di Chen and Feiwu Yu and Haiming Zhao and Jianxiao Yang and Jianyuan Zeng and Jiayu Wang and Jingfeng Zhang and Jingren Zhou and Jinkai Wang and Jixuan Chen and Kai Zhu and Kang Zhao and Keyu Yan and Lianghua Huang and Mengyang Feng and Ningyi Zhang and Pandeng Li and Pingyu Wu and Ruihang Chu and Ruili Feng and Shiwei Zhang and Siyang Sun and Tao Fang and Tianxing Wang and Tianyi Gui and Tingyu Weng and Tong Shen and Wei Lin and Wei Wang and Wei Wang and Wenmeng Zhou and Wente Wang and Wenting Shen and Wenyuan Yu and Xianzhong Shi and Xiaoming Huang and Xin Xu and Yan Kou and Yangyu Lv and Yifei Li and Yijing Liu and Yiming Wang and Yingya Zhang and Yitong Huang and Yong Li and You Wu and Yu Liu and Yulin Pan and Yun Zheng and Yuntao Hong and Yupeng Shi and Yutong Feng and Zeyinzi Jiang and Zhen Han and Zhi-Fan Wu and Ziyu Liu},
      journal = {arXiv preprint arXiv:2503.20314},
      year={2025}
}
```

## License

This project is licensed under the Apache 2.0 License. Please refer to [LICENSE.txt](LICENSE.txt) for details.

## Acknowledgements

Special thanks to the contributors of [SD3](https://huggingface.co/stabilityai/stable-diffusion-3-medium), [Qwen](https://huggingface.co/Qwen), [umt5-xxl](https://huggingface.co/google/umt5-xxl), [diffusers](https://github.com/huggingface/diffusers) and [HuggingFace](https://huggingface.co).

## Contact

Join the community! Connect with us on [Discord](https://discord.gg/AKNgpMK4Yj) or [WeChat](https://gw.alicdn.com/imgextra/i2/O1CN01tqjWFi1ByuyehkTSB_!!6000000000015-0-tps-611-1279.jpg).