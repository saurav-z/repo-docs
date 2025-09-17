# Wan2.2: Unleash Cinematic Video Generation with Open-Source AI

**Wan2.2** is a cutting-edge open-source video generation model, revolutionizing how we create high-quality videos from text, images, and speech.  ([Original Repo](https://github.com/Wan-Video/Wan2.2))

**Key Features:**

*   ✨ **Mixture-of-Experts (MoE) Architecture:**  Leverages a novel MoE architecture to boost model capacity and performance without increasing computational cost.
*   🎬 **Cinematic Aesthetics:**  Incorporates meticulously curated data for precise control over lighting, composition, and color, enabling cinematic-quality video generation.
*   🚀 **Enhanced Motion Generation:** Trained on significantly larger datasets, resulting in improved generalization and realistic movement.
*   ⚡ **Efficient High-Definition TI2V:**  Open-sources a fast 720P model that supports both text-to-video and image-to-video generation, running on consumer-grade GPUs.
*   🗣️ **Speech-to-Video Capability:** Introducing Wan2.2-S2V-14B, an audio-driven cinematic video generation model to generate videos from speech.

## Getting Started

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/Wan-Video/Wan2.2.git
    cd Wan2.2
    ```

2.  **Install dependencies:**

    ```bash
    # Ensure torch >= 2.4.0
    # If the installation of `flash_attn` fails, try installing the other packages first and install `flash_attn` last
    pip install -r requirements.txt
    # If you want to use CosyVoice to synthesize speech for Speech-to-Video Generation, please install requirements_s2v.txt additionally
    pip install -r requirements_s2v.txt
    ```

### Model Download

Choose the model that fits your use case.  Models are available on Hugging Face and ModelScope:

| Model                  | Description                      | Download Links                                                                                                                                   |
| ---------------------- | -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| T2V-A14B              | Text-to-Video (MoE)               | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B)    🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-T2V-A14B)    |
| I2V-A14B              | Image-to-Video (MoE)              | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B)    🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-I2V-A14B)    |
| TI2V-5B               | Text-Image-to-Video (720P)         | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B)     🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-TI2V-5B)     |
| S2V-14B               | Speech-to-Video                  | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-S2V-14B)     🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-S2V-14B)     |

**Download using `huggingface-cli`:**

```bash
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.2-T2V-A14B --local-dir ./Wan2.2-T2V-A14B
```

**Download using `modelscope-cli`:**

```bash
pip install modelscope
modelscope download Wan-AI/Wan2.2-T2V-A14B --local_dir ./Wan2.2-T2V-A14B
```

## Running Wan2.2

Detailed instructions for running text-to-video, image-to-video, text-image-to-video and speech-to-video generation can be found in the original README (linked above).  Key steps include:

*   **Specifying Task:** Select the appropriate `--task` parameter (e.g., `t2v-A14B`, `i2v-A14B`, `ti2v-5B`, `s2v-14B`).
*   **Setting Model Path:**  Use the `--ckpt_dir` parameter to point to the downloaded model.
*   **Providing Prompts/Inputs:**  Use `--prompt` for text, `--image` for images, and `--audio` for audio input.
*   **Resolution and GPU Configuration:**  Set the `--size` parameter for the video resolution and explore multi-GPU inference options.

### Example: Text-to-Video Generation

```bash
python generate.py  --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --offload_model True --convert_model_dtype --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
```

## Community and Resources

*   **Paper:** [Wan: Open and Advanced Large-Scale Video Generative Models](https://arxiv.org/abs/2503.20314)
*   **Blog:** [Wan Video Blog](https://wan.video/welcome?spm=a2ty_o02.30011076.0.0.6c9ee41eCcluqg)
*   **Discord:** [Discord](https://discord.gg/AKNgpMK4Yj)
*   **User Guides:** [English User Guide](https://alidocs.dingtalk.com/i/nodes/EpGBa2Lm8aZxe5myC99MelA2WgN7R35y) & [Chinese User Guide](https://alidocs.dingtalk.com/i/nodes/jb9Y4gmKWrx9eo4dCql9LlbYJGXn6lpz)

## Citation

If you use Wan2.2 in your research, please cite our work:

```
@article{wan2025,
      title={Wan: Open and Advanced Large-Scale Video Generative Models}, 
      author={Team Wan and Ang Wang and Baole Ai and Bin Wen and Chaojie Mao and Chen-Wei Xie and Di Chen and Feiwu Yu and Haiming Zhao and Jianxiao Yang and Jianyuan Zeng and Jiayu Wang and Jingfeng Zhang and Jingren Zhou and Jinkai Wang and Jixuan Chen and Kai Zhu and Kang Zhao and Keyu Yan and Lianghua Huang and Mengyang Feng and Ningyi Zhang and Pandeng Li and Pingyu Wu and Ruihang Chu and Ruili Feng and Shiwei Zhang and Siyang Sun and Tao Fang and Tianxing Wang and Tianyi Gui and Tingyu Weng and Tong Shen and Wei Lin and Wei Wang and Wei Wang and Wenmeng Zhou and Wente Wang and Wenting Shen and Wenyuan Yu and Xianzhong Shi and Xiaoming Huang and Xin Xu and Yan Kou and Yangyu Lv and Yifei Li and Yijing Liu and Yiming Wang and Yingya Zhang and Yitong Huang and Yong Li and You Wu and Yu Liu and Yulin Pan and Yun Zheng and Yuntao Hong and Yupeng Shi and Yutong Feng and Zeyinzi Jiang and Zhen Han and Zhi-Fan Wu and Ziyu Liu},
      journal = {arXiv preprint arXiv:2503.20314},
      year={2025}
}