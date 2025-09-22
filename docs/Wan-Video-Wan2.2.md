# Wan2.2: Unleash the Power of Advanced Video Generation 

**Create stunning, high-quality videos with ease using Wan2.2, the open-source video generative model, offering cinematic aesthetics and efficient performance.  [Explore the Original Repo](https://github.com/Wan-Video/Wan2.2)**

---

**Key Features:**

*   **Mixture-of-Experts (MoE) Architecture:** Enhance video generation quality while maintaining computational efficiency.
*   **Cinematic Aesthetics:** Generate videos with fine-tuned control over lighting, composition, and color for a professional look.
*   **Advanced Motion Generation:** Achieve superior results with the ability to create intricate and dynamic video content.
*   **Efficient High-Definition Hybrid TI2V:** Generate 720P videos at 24fps on consumer-grade GPUs with the 5B model, ideal for both research and practical applications.
*   **Versatile Generation Modes:** Support text-to-video, image-to-video, text-image-to-video, speech-to-video, and character animation/replacement.

---

## Latest Updates

*   **Wan2.2-Animate-14B:**  Unified model for character animation and replacement. [Model Weights](link to models) and [Inference Code](link to code). Try it on [wan.video](https://wan.video/), [ModelScope Studio](https://www.modelscope.cn/studios/Wan-AI/Wan2.2-Animate) or [HuggingFace Space](https://huggingface.co/spaces/Wan-AI/Wan2.2-Animate)!
*   **Wan2.2-S2V-14B:** Audio-driven cinematic video generation.  [Inference Code](link to code), [Model Weights](link to models), and [Technical Report](link to report). Try it on [wan.video](https://wan.video/),  [ModelScope Gradio](https://www.modelscope.cn/studios/Wan-AI/Wan2.2-S2V) or [HuggingFace Gradio](https://huggingface.co/spaces/Wan-AI/Wan2.2-S2V)!
*   **TI2V-5B Model:** Open-sourced on Hugging Face Spaces.
*   **ComfyUI Integration:** Integrated into ComfyUI.
*   **Diffusers Integration:**  Integrated into Diffusers (T2V, I2V, and TI2V).
*   **Inference Code and Model Weights Released:**  Initial release of inference code and model weights.
*   **Text-to-Speech Support:** CosyVoice integration for Speech-to-Video.

---

## Community Contributions

*   **DiffSynth-Studio:** Comprehensive Wan 2.2 support, including optimization features.
*   **Kijai's ComfyUI WanVideoWrapper:** Alternative implementation for ComfyUI.
*   **Cache-dit:** Fully Cache Acceleration support.
*   **FastVideo:** Distilled Wan models for faster inference.

---

## Quick Start: Run Wan2.2

### Installation

```bash
git clone https://github.com/Wan-Video/Wan2.2.git
cd Wan2.2
pip install -r requirements.txt
pip install -r requirements_s2v.txt  # If using Speech-to-Video
```

### Model Download

| Model              | Download Links                                                                                                                              | Description |
|--------------------|---------------------------------------------------------------------------------------------------------------------------------------------|-------------|
| T2V-A14B    | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B)    🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-T2V-A14B)    | Text-to-Video MoE model, supports 480P & 720P |
| I2V-A14B    | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B)    🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-I2V-A14B)    | Image-to-Video MoE model, supports 480P & 720P |
| TI2V-5B     | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B)     🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-TI2V-5B)     | High-compression VAE, T2V+I2V, supports 720P |
| S2V-14B     | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-S2V-14B)     🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-S2V-14B)     | Speech-to-Video model, supports 480P & 720P |
| Animate-14B | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-Animate-14B) 🤖 [ModelScope](https://www.modelscope.cn/models/Wan-AI/Wan2.2-Animate-14B)  | Character animation and replacement | |

Download models using huggingface-cli:
``` sh
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.2-T2V-A14B --local-dir ./Wan2.2-T2V-A14B
```

Download models using modelscope-cli:
``` sh
pip install modelscope
modelscope download Wan-AI/Wan2.2-T2V-A14B --local_dir ./Wan2.2-T2V-A14B
```


### Text-to-Video Generation
```bash
python generate.py  --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --offload_model True --convert_model_dtype --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
```

### Image-to-Video Generation
```bash
python generate.py --task i2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-I2V-A14B --offload_model True --convert_model_dtype --image examples/i2v_input.JPG --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
```

### Text-Image-to-Video Generation
```bash
python generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ./Wan2.2-TI2V-5B --offload_model True --convert_model_dtype --t5_cpu --image examples/i2v_input.JPG --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
```

### Speech-to-Video Generation
```bash
python generate.py  --task s2v-14B --size 1024*704 --ckpt_dir ./Wan2.2-S2V-14B/ --offload_model True --convert_model_dtype --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard."  --image "examples/i2v_input.JPG" --audio "examples/talk.wav"
```

### Wan-Animate
Please follow the steps described in this section or visit our [project page](https://humanaigc.github.io/wan-animate) to learn more.

---

##  Model Details

*   **Mixture-of-Experts (MoE) Architecture:** An architectural innovation to efficiently scale the model for improved video generation quality.  [See the moer details](#introduction-of-wan22)
*   **Efficient High-Definition Hybrid TI2V:** A high compression design to ensure faster inference speed. [See the vae details](#introduction-of-wan22)

---

## Performance and Comparison

Wan2.2 surpasses leading commercial models in benchmark tests, showcasing its superior capabilities.  [View performance details](#comparisons-to-sotas).

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

---

## License

Wan2.2 is licensed under the [Apache 2.0 License](LICENSE.txt).

---

## Acknowledgements

Thanks to the contributors of [SD3](https://huggingface.co/stabilityai/stable-diffusion-3-medium), [Qwen](https://huggingface.co/Qwen), [umt5-xxl](https://huggingface.co/google/umt5-xxl), [diffusers](https://github.com/huggingface/diffusers) and [HuggingFace](https://huggingface.co).

---

## Contact

Join our [Discord](https://discord.gg/AKNgpMK4Yj) or [WeChat groups](https://gw.alicdn.com/imgextra/i2/O1CN01tqjWFi1ByuyehkTSB_!!6000000000015-0-tps-611-1279.jpg) to connect with our research and product teams.