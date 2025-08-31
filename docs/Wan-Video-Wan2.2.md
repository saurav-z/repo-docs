# Wan2.2: Unleash the Power of Advanced Video Generation

**Wan2.2** offers cutting-edge video generation, providing unparalleled quality and control in the realm of AI-driven video creation. Explore advanced features like a Mixture-of-Experts (MoE) architecture, and generate cinematic-quality videos with ease.  Access the original repository on [GitHub](https://github.com/Wan-Video/Wan2.2) for more details.

## Key Features

*   **Mixture-of-Experts (MoE) Architecture:** Enhance model capacity and performance without increasing computational costs.
*   **Cinematic-Level Aesthetics:** Achieve precise control over lighting, composition, color, and more for professional-grade videos.
*   **Enhanced Motion Generation:** Generate dynamic and complex video content thanks to expanded data training.
*   **Efficient High-Definition Hybrid TI2V:** Generate 720P videos at 24fps, running on consumer-grade GPUs, with fast generation speed.
*   **Speech-to-Video Generation**: Create videos driven by audio inputs.

## Getting Started

### Installation
```bash
git clone https://github.com/Wan-Video/Wan2.2.git
cd Wan2.2
pip install -r requirements.txt
```

### Model Download

| Model                                   | Download                                                                                                                            | Description                                                              |
| :-------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------- |
| T2V-A14B                                | [Hugging Face](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B) | Text-to-Video, supports 480P & 720P                    |
| I2V-A14B                                | [Hugging Face](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B)  | Image-to-Video, supports 480P & 720P                     |
| TI2V-5B                                 | [Hugging Face](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B)    | Text-Image-to-Video, supports 720P                         |
| S2V-14B                                 | [Hugging Face](https://huggingface.co/Wan-AI/Wan2.2-S2V-14B)  | Speech-to-Video, supports 480P & 720P                     |

**Download using Hugging Face CLI**
```bash
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.2-T2V-A14B --local-dir ./Wan2.2-T2V-A14B
```

**Download using ModelScope CLI**
```bash
pip install modelscope
modelscope download Wan-AI/Wan2.2-T2V-A14B --local_dir ./Wan2.2-T2V-A14B
```

### Run Examples

**Text-to-Video**
```bash
python generate.py  --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --offload_model True --convert_model_dtype --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
```

**Image-to-Video**
```bash
python generate.py --task i2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-I2V-A14B --offload_model True --convert_model_dtype --image examples/i2v_input.JPG --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
```

**Speech-to-Video**
```bash
python generate.py  --task s2v-14B --size 1024*704 --ckpt_dir ./Wan2.2-S2V-14B/ --offload_model True --convert_model_dtype --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard."  --image "examples/i2v_input.JPG" --audio "examples/talk.wav"
```

## Latest Updates

*   **August 26, 2025**:  Introduction of [Wan2.2-S2V-14B](https://humanaigc.github.io/wan-s2v-webpage), an audio-driven cinematic video generation model.
*   **July 28, 2025**:  Integration into [ComfyUI](https://docs.comfy.org/zh-CN/tutorials/video/wan/wan2_2) and [Diffusers](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers).

## Community Contributions

*   [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio): Support for Wan 2.2, including low-GPU-memory layer-by-layer offload and more.
*   [Kijai's ComfyUI WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper): Alternative implementation of Wan models for ComfyUI.

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
The models in this repository are licensed under the [Apache 2.0 License](LICENSE.txt).

## Acknowledgements
We thank the contributors of [SD3](https://huggingface.co/stabilityai/stable-diffusion-3-medium), [Qwen](https://huggingface.co/Qwen), [umt5-xxl](https://huggingface.co/google/umt5-xxl), [diffusers](https://github.com/huggingface/diffusers) and [HuggingFace](https://huggingface.co) for their open research.

## Contact
Join our [Discord](https://discord.gg/AKNgpMK4Yj) or [WeChat groups](https://gw.alicdn.com/imgextra/i2/O1CN01tqjWFi1ByuyehkTSB_!!6000000000015-0-tps-611-1279.jpg) for questions or suggestions.