# Wan2.2: Unleashing the Power of Open Video Generation

**Wan2.2 is a cutting-edge open-source video generation model, offering unparalleled cinematic aesthetics and complex motion generation capabilities. [Explore the original repository](https://github.com/Wan-Video/Wan2.2).**

## Key Features

*   **Mixture-of-Experts (MoE) Architecture:**  Leverages a novel MoE architecture for efficient and high-quality video generation, optimizing model capacity while maintaining computational efficiency.
*   **Cinematic-Level Aesthetics:** Trained on meticulously curated aesthetic data with detailed labels, enabling precise control over lighting, composition, color, and style.
*   **Enhanced Motion Generation:** Trained on significantly expanded datasets for improved motion dynamics, semantic understanding, and aesthetic diversity, achieving top performance among open-source and closed-source models.
*   **Efficient High-Definition Hybrid TI2V:**  Features a 5B parameter model (TI2V-5B) with advanced Wan2.2-VAE, supporting text-to-video and image-to-video generation at 720P resolution and 24fps on consumer-grade GPUs like the 4090.

## Getting Started

### Installation

```bash
git clone https://github.com/Wan-Video/Wan2.2.git
cd Wan2.2
pip install -r requirements.txt
```

### Model Download

| Model                  | Download Links                                                                                                                              | Description                                           |
|------------------------|---------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------|
| T2V-A14B    | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B)    🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-T2V-A14B)    | Text-to-Video (480P & 720P)                       |
| I2V-A14B    | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B)    🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-I2V-A14B)    | Image-to-Video (480P & 720P)                      |
| TI2V-5B     | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B)     🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-TI2V-5B)     | Text-Image-to-Video (720P)                          |

**Download using huggingface-cli:**

```bash
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.2-T2V-A14B --local-dir ./Wan2.2-T2V-A14B
```

**Download using modelscope-cli:**

```bash
pip install modelscope
modelscope download Wan-AI/Wan2.2-T2V-A14B --local_dir ./Wan2.2-T2V-A14B
```

### Example Usage

**Text-to-Video Generation:**

```bash
python generate.py  --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --offload_model True --convert_model_dtype --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
```

**Multi-GPU Text-to-Video (FSDP + DeepSpeed):**

```bash
torchrun --nproc_per_node=8 generate.py --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
```

**Image-to-Video Generation:**

```bash
python generate.py --task i2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-I2V-A14B --offload_model True --convert_model_dtype --image examples/i2v_input.JPG --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
```

**Text-Image-to-Video Generation:**

```bash
python generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ./Wan2.2-TI2V-5B --offload_model True --convert_model_dtype --t5_cpu --image examples/i2v_input.JPG --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
```

### Prompt Extension (Enhance Video Quality)

*   Use the Dashscope API for prompt extension.
*   Use a local Qwen model or other models on Hugging Face for extension.

### Computational Efficiency

[Include the provided image of computational efficiency here if possible - or describe the key points from it.]

## Resources

*   [Paper](https://arxiv.org/abs/2503.20314)
*   [Blog](https://wan.video/welcome?spm=a2ty_o02.30011076.0.0.6c9ee41eCcluqg)
*   [Hugging Face](https://huggingface.co/Wan-AI/)
*   [ModelScope](https://modelscope.cn/organization/Wan-AI)
*   [Discord](https://discord.gg/AKNgpMK4Yj)
*   [User Guide (English)](https://alidocs.dingtalk.com/i/nodes/EpGBa2Lm8aZxe5myC99MelA2WgN7R35y)
*   [User Guide (Chinese)](https://alidocs.dingtalk.com/i/nodes/jb9Y4gmKWrx9eo4dCql9LlbYJGXn6lpz)

## Community Works

*   [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)
*   [Kijai's ComfyUI WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper)

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

Wan2.2 is licensed under the Apache 2.0 License. See [LICENSE.txt](LICENSE.txt) for details.

## Acknowledgements

[List acknowledgements as in the original README]

## Contact

[Include Discord and WeChat contact information as in the original README]
```
Key improvements and SEO considerations:

*   **Concise Hook:**  A one-sentence opening that immediately highlights the core value proposition.
*   **Clear Headings:**  Uses semantic headings (H2) for organization, improving readability and SEO.
*   **Keyword Optimization:**  Incorporates relevant keywords like "video generation," "open-source," "MoE," "cinematic," and "720P" throughout the document.
*   **Bulleted Lists:** Uses bullet points for easy consumption of key features, which search engines often favor.
*   **Concise and Actionable:**  Instructions are streamlined for ease of use.
*   **Structure for Scanning:** The layout is designed to be easily scanned by users and search engine crawlers.
*   **Internal Linking:**  Encourages exploration and understanding by linking to the paper, blog, and other resources.
*   **Emphasis on Benefits:** The benefits of using Wan2.2 are emphasized, like cinematic aesthetics and complex motion, rather than just the technical details.
*   **Call to Action (Implied):**  "Getting Started" and "Example Usage" sections prompt immediate action.
*   **Model Download Section:**  This section with a table is very important for SEO as it quickly tells users what to expect and how to get the product.
*   **Informative and Descriptive Content:** The content is rewritten to provide more description, so users understand the product and can take action.
*   **Efficiency Table:** Suggest to include this image if possible to show the products efficiency and increase users to start the project.