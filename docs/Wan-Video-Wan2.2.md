# Wan2.2: Unleash Cinematic Video Generation with Open-Source AI

**Wan2.2** ([Original Repo](https://github.com/Wan-Video/Wan2.2)) is a cutting-edge open-source video generation model pushing the boundaries of AI-driven video creation.  Generate stunning, high-definition videos from text, images, or audio with unparalleled speed and quality.

**Key Features:**

*   👍 **MoE Architecture:** Utilizes a Mixture-of-Experts (MoE) architecture for increased model capacity while maintaining efficient computation.
*   👍 **Cinematic Aesthetics:** Incorporates curated aesthetic data for precise control over lighting, composition, and color grading, enabling cinematic-quality video generation.
*   👍 **Advanced Motion Generation:** Trained on a significantly expanded dataset (+65.6% images, +83.2% videos) compared to Wan2.1, resulting in improved generalization across motions, semantics, and aesthetics.
*   👍 **Efficient High-Definition Hybrid TI2V:** Open-sources a 5B model with a compression ratio of 16x16x4, supporting 720P@24fps text-to-video and image-to-video generation on consumer-grade GPUs like the 4090.

**Access & Resources:**

*   [**Wan**](https://wan.video)
*   [**GitHub**](https://github.com/Wan-Video/Wan2.2)
*   [**Hugging Face**](https://huggingface.co/Wan-AI/)
*   [**ModelScope**](https://modelscope.cn/organization/Wan-AI)
*   [**Paper**](https://arxiv.org/abs/2503.20314)
*   [**Blog**](https://wan.video/welcome?spm=a2ty_o02.30011076.0.0.6c9ee41eCcluqg)
*   [**Discord**](https://discord.gg/AKNgpMK4Yj)
*   [**User Guide (English)**](https://alidocs.dingtalk.com/i/nodes/EpGBa2Lm8aZxe5myC99MelA2WgN7R35y)
*   [**User Guide (中文)**](https://alidocs.dingtalk.com/i/nodes/jb9Y4gmKWrx9eo4dCql9LlbYJGXn6lpz)

## Latest News & Updates:

*   **[Wan2.2-Animate-14B](https://humanaigc.github.io/wan-animate)**: Unified model for character animation and replacement. (Released September 19, 2025)
*   **[Wan2.2-S2V-14B](https://humanaigc.github.io/wan-s2v-webpage)**: Audio-driven cinematic video generation. (Released August 26, 2025)
*   And more!  See the original README for more announcements.

## Video Demos
[Include Video Demo]

## Community Works
*   [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)
*   [Kijai's ComfyUI WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper)
*   [Cache-dit](https://github.com/vipshop/cache-dit)
*   [FastVideo](https://github.com/hao-ai-lab/FastVideo)

## Installation
```sh
git clone https://github.com/Wan-Video/Wan2.2.git
cd Wan2.2
pip install -r requirements.txt
pip install -r requirements_s2v.txt # If using Speech-to-Video
```

## Model Download

| Models              | Download Links                                                                                                                              | Description |
|--------------------|-----------------------------------------------------------------------------------------------------------------------------|-------------|
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

## Run Wan2.2 - Quickstart Guides

*(Note:  Detailed instructions for each task (T2V, I2V, TI2V, S2V, Animate) are provided in the original README, below this summary. These are not repeated here for brevity, but are essential.)*

**Key steps for all tasks:**

1.  **Install Dependencies:**  (See Installation instructions above)
2.  **Download Models:** (See Model Download instructions above)
3.  **Run Generation:** (Refer to the original README section "Run Wan2.2" for commands.  Examples are provided below.)

### Example: Text-to-Video (T2V) - Single GPU

```sh
python generate.py  --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --offload_model True --convert_model_dtype --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
```

### Example: Image-to-Video (I2V) - Single GPU

```sh
python generate.py --task i2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-I2V-A14B --offload_model True --convert_model_dtype --image examples/i2v_input.JPG --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
```

### Example: Text-Image-to-Video (TI2V) - Single GPU

```sh
python generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ./Wan2.2-TI2V-5B --offload_model True --convert_model_dtype --t5_cpu --image examples/i2v_input.JPG --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
```

### Example: Speech-to-Video (S2V) - Single GPU

```sh
python generate.py  --task s2v-14B --size 1024*704 --ckpt_dir ./Wan2.2-S2V-14B/ --offload_model True --convert_model_dtype --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard."  --image "examples/i2v_input.JPG" --audio "examples/talk.wav"
```

### Example: Wan-Animate (Character Animation/Replacement) - Single GPU

*(Refer to original README for detailed pre-processing steps.)*

```bash
python generate.py --task animate-14B --ckpt_dir ./Wan2.2-Animate-14B/ --src_root_path ./examples/wan_animate/animate/process_results/ --refert_num 1
```

## Computational Efficiency

[Include Image of Computational Efficiency Table from Original README]

## Detailed Introduction

Wan2.2 builds on the advancements of Wan2.1 with significant improvements in video quality and capability. The key innovations include:

### Mixture-of-Experts (MoE) Architecture
[Include MoE architecture graphic from original README]

### Efficient High-Definition Hybrid TI2V
[Include VAE graphic from original README]

### Comparisons to SOTAs
[Include Performance graphic from original README]

## Citation
```
@article{wan2025,
      title={Wan: Open and Advanced Large-Scale Video Generative Models}, 
      author={Team Wan and Ang Wang and Baole Ai and Bin Wen and Chaojie Mao and Chen-Wei Xie and Di Chen and Feiwu Yu and Haiming Zhao and Jianxiao Yang and Jianyuan Zeng and Jiayu Wang and Jingfeng Zhang and Jingren Zhou and Jinkai Wang and Jixuan Chen and Kai Zhu and Kang Zhao and Keyu Yan and Lianghua Huang and Mengyang Feng and Ningyi Zhang and Pandeng Li and Pingyu Wu and Ruihang Chu and Ruili Feng and Shiwei Zhang and Siyang Sun and Tao Fang and Tianxing Wang and Tianyi Gui and Tingyu Weng and Tong Shen and Wei Lin and Wei Wang and Wei Wang and Wenmeng Zhou and Wente Wang and Wenting Shen and Wenyuan Yu and Xianzhong Shi and Xiaoming Huang and Xin Xu and Yan Kou and Yangyu Lv and Yifei Li and Yijing Liu and Yiming Wang and Yingya Zhang and Yitong Huang and Yong Li and You Wu and Yu Liu and Yulin Pan and Yun Zheng and Yuntao Hong and Yupeng Shi and Yutong Feng and Zeyinzi Jiang and Zhen Han and Zhi-Fan Wu and Ziyu Liu},
      journal = {arXiv preprint arXiv:2503.20314},
      year={2025}
}
```

## License Agreement
[Include License Text from original README or provide a link to the LICENSE.txt]

## Acknowledgements
[Include Acknowledgements from original README]

## Contact
[Include Contact Information from original README]

---

*Note: The detailed "Run Wan2.2" instructions, and all content below that section of the original README, are incorporated by reference and are still crucial for users. This summary provides a high-level overview for SEO and ease of navigation.*