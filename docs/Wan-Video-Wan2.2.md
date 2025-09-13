# Wan2.2: Unleash Cinematic Video Generation with Open-Source AI

**Wan2.2** is a powerful, open-source video generation model that allows you to create stunning videos from text, images, and audio. Explore the cutting edge of AI-powered video creation, backed by the team at [Wan-Video](https://github.com/Wan-Video/Wan2.2).

*   [**GitHub**](https://github.com/Wan-Video/Wan2.2)
*   [**Hugging Face**](https://huggingface.co/Wan-AI/)
*   [**ModelScope**](https://modelscope.cn/organization/Wan-AI)
*   [**Paper**](https://arxiv.org/abs/2503.20314)
*   [**Blog**](https://wan.video/welcome?spm=a2ty_o02.30011076.0.0.6c9ee41eCcluqg)
*   [**Discord**](https://discord.gg/AKNgpMK4Yj)

---

## Key Features

*   **MoE Architecture:** Utilizing a Mixture-of-Experts (MoE) design, Wan2.2 boosts model capacity without increasing computational cost, leading to improved video generation.
*   **Cinematic Aesthetics:** Leverage detailed aesthetic data including lighting, composition, and color grading to achieve precise control over video style and create cinematic quality results.
*   **Advanced Motion Generation:** Trained on a substantially larger dataset (65.6% more images, 83.2% more videos) compared to Wan2.1, Wan2.2 provides superior performance in a wide variety of motions, semantics, and aesthetics.
*   **Efficient High-Definition Hybrid TI2V:** Generate high-quality 720P videos at 24fps with the open-sourced 5B model and Wan2.2-VAE.
*   **Text, Image, Audio & Pose Input Support:** Generate videos from text prompts, reference images, audio, or pose sequences to create diverse and dynamic content.
*   **ComfyUI & Diffusers Integration:** Easy-to-use integration with ComfyUI and Diffusers allows for straightforward generation and experimentation.

## What's New

### Wan2.2-S2V (Speech-to-Video)
*   Audio-driven cinematic video generation
    *   [Inference Code](https://humanaigc.github.io/wan-s2v-webpage/content/wan-s2v.pdf)
    *   [Model Weights](https://humanaigc.github.io/wan-s2v-webpage)
    *   [Technical Report](https://humanaigc.github.io/wan-s2v-webpage/content/wan-s2v.pdf)!
    *   Try it on [wan.video](https://wan.video/), [ModelScope Gradio](https://www.modelscope.cn/studios/Wan-AI/Wan2.2-S2V) or [HuggingFace Gradio](https://huggingface.co/spaces/Wan-AI/Wan2.2-S2V)!

### Integration and Releases
*   Wan2.2 has been integrated into ComfyUI ([CN](https://docs.comfy.org/zh-CN/tutorials/video/wan/wan2_2) | [EN](https://docs.comfy.org/tutorials/video/wan/wan2_2)).
*   Wan2.2's T2V, I2V and TI2V have been integrated into Diffusers ([T2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) | [I2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers) | [TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers)).

## Getting Started

### Installation

```bash
git clone https://github.com/Wan-Video/Wan2.2.git
cd Wan2.2
pip install -r requirements.txt
pip install -r requirements_s2v.txt # if you need speech-to-video generation support
```

### Model Download

Download the required models from Hugging Face or ModelScope.

| Models              | Download Links                                                                                                                              | Description |
|--------------------|---------------------------------------------------------------------------------------------------------------------------------------------|-------------|
| T2V-A14B    | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B)    🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-T2V-A14B)    | Text-to-Video MoE model, supports 480P & 720P |
| I2V-A14B    | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B)    🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-I2V-A14B)    | Image-to-Video MoE model, supports 480P & 720P |
| TI2V-5B     | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B)     🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-TI2V-5B)     | High-compression VAE, T2V+I2V, supports 720P |
| S2V-14B     | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-S2V-14B)     🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-S2V-14B)     | Speech-to-Video model, supports 480P & 720P |

#### Example Download (Text-to-Video)

```bash
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.2-T2V-A14B --local-dir ./Wan2.2-T2V-A14B
```

or
```bash
pip install modelscope
modelscope download Wan-AI/Wan2.2-T2V-A14B --local_dir ./Wan2.2-T2V-A14B
```

### Run Example Generations

Follow the detailed instructions in the original README for running text-to-video, image-to-video, text-image-to-video, and speech-to-video.  Key commands are shown below.

**Text-to-Video (T2V)**

```bash
python generate.py  --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --offload_model True --convert_model_dtype --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
```

**Image-to-Video (I2V)**

```bash
python generate.py --task i2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-I2V-A14B --offload_model True --convert_model_dtype --image examples/i2v_input.JPG --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
```

**Speech-to-Video (S2V)**

```bash
python generate.py  --task s2v-14B --size 1024*704 --ckpt_dir ./Wan2.2-S2V-14B/ --offload_model True --convert_model_dtype --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard."  --image "examples/i2v_input.JPG" --audio "examples/talk.wav"
```

## Community Works

If you're working with Wan2.2 or Wan2.1, share your project or research with the community!

*   [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)
*   [Kijai's ComfyUI WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper)
*   [Cache-dit](https://github.com/vipshop/cache-dit)
*   [FastVideo](https://github.com/hao-ai-lab/FastVideo)

##  Computational Efficiency

[See the original README for the computational efficiency tables.]

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

This project is licensed under the Apache 2.0 License. Please refer to the [LICENSE.txt](LICENSE.txt) file for full details.

## Acknowledgements

Thank you to the contributors of SD3, Qwen, umt5-xxl, diffusers, and Hugging Face.

## Contact

Join the [Discord](https://discord.gg/AKNgpMK4Yj) or [WeChat groups](https://gw.alicdn.com/imgextra/i2/O1CN01tqjWFi1ByuyehkTSB_!!6000000000015-0-tps-611-1279.jpg) to connect with the research and product teams.