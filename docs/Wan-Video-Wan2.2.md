# Wan2.2: Unleashing Advanced Large-Scale Video Generation (Open-Source)

**Generate stunning videos with cutting-edge AI.**  Wan2.2 is an open-source project that pushes the boundaries of video generation, offering state-of-the-art capabilities for text-to-video, image-to-video, and more.  Explore the power of advanced AI video generation and join our vibrant community!

*   [**GitHub Repository**](https://github.com/Wan-Video/Wan2.2) | [Hugging Face](https://huggingface.co/Wan-AI/) | [Paper](https://arxiv.org/abs/2503.20314) | [Blog](https://wan.video/welcome?spm=a2ty_o02.30011076.0.0.6c9ee41eCcluqg) | [Discord](https://discord.gg/AKNgpMK4Yj)

## Key Features

*   ✅ **Mixture-of-Experts (MoE) Architecture:** Leverage advanced MoE architecture for increased model capacity without sacrificing inference speed.
*   ✅ **Cinematic Aesthetics:** Create videos with precise control over lighting, composition, color, and more, for professional-level results.
*   ✅ **Complex Motion Generation:**  Generate highly dynamic and nuanced videos thanks to a significantly expanded training dataset.
*   ✅ **Efficient High-Definition Hybrid TI2V:** Experience fast 720P@24fps video generation on consumer-grade GPUs with our optimized TI2V-5B model.

## Video Demos

<div align="center">
  <video src="https://github.com/user-attachments/assets/b63bfa58-d5d7-4de6-a1a2-98970b06d9a7" width="70%" poster=""> </video>
</div>

## Latest News

Stay up-to-date with the latest releases and features:

*   **Wan2.2-Animate-14B:** Character animation and replacement model ([Model](https://huggingface.co/Wan-AI/Wan2.2-Animate-14B), [ModelScope](https://www.modelscope.cn/models/Wan-AI/Wan2.2-Animate-14B), [Demo](https://wan.video/), [ModelScope Studio](https://www.modelscope.cn/studios/Wan-AI/Wan2.2-Animate), [Hugging Face Space](https://huggingface.co/spaces/Wan-AI/Wan2.2-Animate))
*   **Wan2.2-S2V-14B:** Audio-driven cinematic video generation ([Inference Code](#run-speech-to-video-generation), [Model](https://huggingface.co/Wan-AI/Wan2.2-S2V-14B), [Technical Report](https://humanaigc.github.io/wan-s2v-webpage/content/wan-s2v.pdf), [Demo](https://wan.video/), [ModelScope](https://www.modelscope.cn/studios/Wan-AI/Wan2.2-S2V), [Hugging Face](https://huggingface.co/spaces/Wan-AI/Wan2.2-S2V))
*   [HF Space](https://huggingface.co/spaces/Wan-AI/Wan-2.2-5B)
*   ComfyUI Integration ([CN](https://docs.comfy.org/zh-CN/tutorials/video/wan/wan2_2) | [EN](https://docs.comfy.org/tutorials/video/wan/wan2_2))
*   Diffusers Integration ([T2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) | [I2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers) | [TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers))
*   Inference code and model weights released
*   Text-to-speech synthesis support with [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) for Speech-to-Video generation

## Community Works

Explore community contributions and integrations:

*   [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)
*   [Kijai's ComfyUI WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper)
*   [Cache-dit](https://github.com/vipshop/cache-dit)
*   [FastVideo](https://github.com/hao-ai-lab/FastVideo)

## Getting Started

### Installation

```bash
git clone https://github.com/Wan-Video/Wan2.2.git
cd Wan2.2
pip install -r requirements.txt
pip install -r requirements_s2v.txt  # for Speech-to-Video (optional)
```

### Model Download

Download the models:

| Model           | Download Links                                                                                                                                                                    | Description                                    |
| --------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| T2V-A14B        | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B) 🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-T2V-A14B)                                                | Text-to-Video MoE, 480P & 720P                |
| I2V-A14B        | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B) 🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-I2V-A14B)                                                | Image-to-Video MoE, 480P & 720P               |
| TI2V-5B         | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) 🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-TI2V-5B)                                                 | Text-Image-to-Video, 720P                      |
| S2V-14B         | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-S2V-14B) 🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-S2V-14B)                                                 | Speech-to-Video, 480P & 720P                  |
| Animate-14B     | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-Animate-14B) 🤖 [ModelScope](https://www.modelscope.cn/models/Wan-AI/Wan2.2-Animate-14B)                                      | Character animation and replacement           |

Download models using `huggingface-cli` or `modelscope-cli`. (See original readme)

### Run Text-to-Video Generation

```bash
python generate.py  --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --offload_model True --convert_model_dtype --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
```

(See the original readme for Multi-GPU and Prompt Extension instructions)

### Run Image-to-Video Generation

```bash
python generate.py --task i2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-I2V-A14B --offload_model True --convert_model_dtype --image examples/i2v_input.JPG --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
```

(See the original readme for Multi-GPU and Image-to-Video without prompt instructions)

### Run Text-Image-to-Video Generation

```bash
python generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ./Wan2.2-TI2V-5B --offload_model True --convert_model_dtype --t5_cpu --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage"
```

(See the original readme for Single-GPU and Multi-GPU instructions)

### Run Speech-to-Video Generation

```bash
python generate.py  --task s2v-14B --size 1024*704 --ckpt_dir ./Wan2.2-S2V-14B/ --offload_model True --convert_model_dtype --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard."  --image "examples/i2v_input.JPG" --audio "examples/talk.wav"
```

(See the original readme for Multi-GPU and Speech-to-Video with pose instructions)

### Run Wan-Animate

(See the original readme for preprocessing, animation and replacement instructions)

## Computational Efficiency on Different GPUs

(See the original readme for the table and information about GPU performance)

## Introduction of Wan2.2

**Wan2.2** builds upon the foundation of Wan2.1 with notable improvements in generation quality and model capability. This upgrade is driven by a series of key technical innovations, mainly including the Mixture-of-Experts (MoE) architecture, upgraded training data, and high-compression video generation.

(See the original readme for the information on Mixture-of-Experts (MoE) Architecture, Efficient High-Definition Hybrid TI2V and Comparisons to SOTAs)

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

We would like to thank the contributors to the [SD3](https://huggingface.co/stabilityai/stable-diffusion-3-medium), [Qwen](https://huggingface.co/Qwen), [umt5-xxl](https://huggingface.co/google/umt5-xxl), [diffusers](https://github.com/huggingface/diffusers) and [HuggingFace](https://huggingface.co) repositories, for their open research.

## Contact

Join our [Discord](https://discord.gg/AKNgpMK4Yj) or [WeChat groups](https://gw.alicdn.com/imgextra/i2/O1CN01tqjWFi1ByuyehkTSB_!!6000000000015-0-tps-611-1279.jpg) to connect with the team and community.