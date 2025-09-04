# Wan2.2: State-of-the-Art Video Generation – Open Source & Advanced

**Unleash your creativity with Wan2.2, an open-source video generation model offering cutting-edge features and performance; [explore the original repository](https://github.com/Wan-Video/Wan2.2).**

Wan2.2 is a significant upgrade to the foundational video models, offering advanced video generation capabilities.

**Key Features:**

*   **👍 Mixture-of-Experts (MoE) Architecture:** Boosts model capacity and performance while maintaining low computational cost.
*   **👍 Cinematic-Level Aesthetics:** Generate videos with precise control over lighting, composition, color tones, and more.
*   **👍 Enhanced Motion Generation:** Achieves top performance among open-source and closed-source models through a larger dataset.
*   **👍 Efficient High-Definition Hybrid TI2V:** Generates 720P videos at 24fps on consumer-grade GPUs.

## Model Updates and Integrations

*   **[Wan2.2-S2V-14B](https://humanaigc.github.io/wan-s2v-webpage)**: Audio-driven cinematic video generation.
*   **HF space](https://huggingface.co/spaces/Wan-AI/Wan-2.2-5B) using the TI2V-5B model.
*   **ComfyUI Integration**: Wan2.2 integrated into ComfyUI ([CN](https://docs.comfy.org/zh-CN/tutorials/video/wan/wan2_2) | [EN](https://docs.comfy.org/tutorials/video/wan/wan2_2)).
*   **Diffusers Integration**: Wan2.2's T2V, I2V and TI2V have been integrated into Diffusers ([T2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) | [I2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers) | [TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers)).

## Quick Start Guide

### Installation

```bash
git clone https://github.com/Wan-Video/Wan2.2.git
cd Wan2.2
pip install -r requirements.txt
```

### Model Download

Download the desired model from Hugging Face or ModelScope:

| Model        | Hugging Face                                                                                                | ModelScope                                                                                                 |
| ------------ | ----------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| T2V-A14B     | [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B)                                                   | [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-T2V-A14B)                                            |
| I2V-A14B     | [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B)                                                   | [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-I2V-A14B)                                            |
| TI2V-5B      | [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B)                                                    | [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-TI2V-5B)                                             |
| S2V-14B      | [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-S2V-14B)                                                    | [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-S2V-14B)                                             |

Example download using `huggingface-cli`:

```bash
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.2-T2V-A14B --local-dir ./Wan2.2-T2V-A14B
```

### Generation Examples

**Text-to-Video (T2V):**

```bash
python generate.py --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --offload_model True --convert_model_dtype --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
```

**Image-to-Video (I2V):**

```bash
python generate.py --task i2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-I2V-A14B --offload_model True --convert_model_dtype --image examples/i2v_input.JPG --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
```

**Text-Image-to-Video (TI2V):**

```bash
python generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ./Wan2.2-TI2V-5B --offload_model True --convert_model_dtype --t5_cpu --image examples/i2v_input.JPG --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
```

**Speech-to-Video (S2V):**

```bash
python generate.py  --task s2v-14B --size 1024*704 --ckpt_dir ./Wan2.2-S2V-14B/ --offload_model True --convert_model_dtype --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard."  --image "examples/i2v_input.JPG" --audio "examples/talk.wav"
```

## Performance and Efficiency

[Include the computational efficiency table here]

## Detailed Technical Information

### Mixture-of-Experts (MoE) Architecture
[Describe the MoE architecture]

### Efficient High-Definition Hybrid TI2V
[Describe the TI2V model, VAE, and compression ratios]

### Comparisons to SOTAs
[Include Performance image here]

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

[Include License information]

## Acknowledgements

[Include Acknowledgements]

## Contact

Join our [Discord](https://discord.gg/AKNgpMK4Yj) or [WeChat groups](https://gw.alicdn.com/imgextra/i2/O1CN01tqjWFi1ByuyehkTSB_!!6000000000015-0-tps-611-1279.jpg) for support and updates.