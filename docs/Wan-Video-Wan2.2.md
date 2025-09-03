# Wan2.2: Unleash Cinematic Video Generation with Open-Source AI 

**Wan2.2** is a cutting-edge, open-source video generation model, offering state-of-the-art performance and innovative features for creating stunning cinematic videos.  Explore the advancements and possibilities at the [original repository](https://github.com/Wan-Video/Wan2.2).

**Key Features:**

*   🎬 **MoE Architecture:** Leverages a Mixture-of-Experts (MoE) architecture to enhance model capacity and maintain computational efficiency.
*   🎨 **Cinematic Aesthetics:** Trained on meticulously curated aesthetic data to enable precise control over lighting, composition, and color for professional-grade video generation.
*   🚀 **Advanced Generation:** Enhanced training data significantly boosts the model's ability to generate complex motions, semantics, and aesthetics.
*   💡 **Efficient High-Definition:** Open-sources a 5B model (TI2V-5B) that supports text-to-video and image-to-video generation at 720P and 24fps.

## Key Highlights

*   **Wan2.2-S2V-14B:** Introducing a new audio-driven cinematic video generation model, including inference code, model weights, and technical reports. Try it out on wan.video, ModelScope Gradio, or HuggingFace Gradio!
*   **Integration with Popular Platforms:** Available on Hugging Face, ModelScope, and integrated into ComfyUI and Diffusers for easy access and use.

## Run Wan2.2

### Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/Wan-Video/Wan2.2.git
    cd Wan2.2
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Model Download

| Model                | Download Links                                                                                                                                    | Description                                                                                                                               |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| T2V-A14B    | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B)    🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-T2V-A14B)    | Text-to-Video MoE model, supports 480P & 720P                                                                                |
| I2V-A14B    | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B)    🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-I2V-A14B)    | Image-to-Video MoE model, supports 480P & 720P                                                                                |
| TI2V-5B     | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B)     🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-TI2V-5B)     | High-compression VAE, T2V+I2V, supports 720P                                                                               |
| S2V-14B     | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-S2V-14B)     🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-S2V-14B)     | Speech-to-Video model, supports 480P & 720P                                                                               |

#### Download using huggingface-cli:
``` sh
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.2-T2V-A14B --local-dir ./Wan2.2-T2V-A14B
```

#### Download models using modelscope-cli:
``` sh
pip install modelscope
modelscope download Wan-AI/Wan2.2-T2V-A14B --local_dir ./Wan2.2-T2V-A14B
```

### Generation Examples
*   **Text-to-Video**
    ```bash
    python generate.py  --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --offload_model True --convert_model_dtype --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
    ```
*   **Image-to-Video**
    ```bash
    python generate.py --task i2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-I2V-A14B --offload_model True --convert_model_dtype --image examples/i2v_input.JPG --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
    ```
*   **Text-Image-to-Video**
    ```bash
    python generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ./Wan2.2-TI2V-5B --offload_model True --convert_model_dtype --t5_cpu --image examples/i2v_input.JPG --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
    ```
*   **Speech-to-Video**
    ```bash
    python generate.py  --task s2v-14B --size 1024*704 --ckpt_dir ./Wan2.2-S2V-14B/ --offload_model True --convert_model_dtype --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard."  --image "examples/i2v_input.JPG" --audio "examples/talk.wav"
    ```

### Computational Efficiency
The computational efficiency of Wan2.2 models varies across different GPUs. Check the table below for estimated times.

<div align="center">
    <img src="assets/comp_effic.png" alt="" style="width: 80%;" />
</div>

## Key Innovations of Wan2.2

*   **Mixture-of-Experts (MoE) Architecture:** Optimized for the denoising process in diffusion models, enhancing model capacity while maintaining low computational cost.
*   **Efficient High-Definition Hybrid TI2V:** Offers a 5B model capable of high-quality 720P video generation, ideal for both research and practical applications.
*   **Advanced Training Data:** Expansive datasets significantly improve the model's ability to create complex motions and detailed aesthetics.

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

Wan2.2 is licensed under the Apache 2.0 License.

## Acknowledgements

Thank you to the contributors of SD3, Qwen, umt5-xxl, diffusers, and HuggingFace.

## Contact

Join the [Discord](https://discord.gg/AKNgpMK4Yj) or [WeChat groups](https://gw.alicdn.com/imgextra/i2/O1CN01tqjWFi1ByuyehkTSB_!!6000000000015-0-tps-611-1279.jpg) for inquiries and support!