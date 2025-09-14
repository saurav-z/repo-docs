# Wan2.2: Unleash Cinematic Video Generation with Advanced AI

**Create stunning, high-quality videos with Wan2.2, an open-source video generation model featuring cutting-edge architecture and performance enhancements.** ([Original Repo](https://github.com/Wan-Video/Wan2.2))

## Key Features

*   **Mixture-of-Experts (MoE) Architecture:** Achieve enhanced model capacity with minimal computational overhead using specialized expert models.
*   **Cinematic Aesthetics:** Generate videos with precise control over lighting, composition, and color for customizable cinematic styles.
*   **Superior Motion & Detail:** Experience significantly improved realism and complexity in your video creations, trained on a vast dataset.
*   **Efficient HD Generation:** Utilize the TI2V-5B model for fast, 720P@24fps text-to-video and image-to-video generation on consumer-grade GPUs.
*   **Text-to-Video (T2V), Image-to-Video (I2V), and Text-Image-to-Video (TI2V) Support:** Generate videos from text descriptions, images, or a combination of both.
*   **Speech-to-Video (S2V) Support:** Generate videos from audio input, including voiceovers and music.

## What's New

*   **Wan2.2-S2V-14B:** Generate audio-driven cinematic videos with the latest Speech-to-Video model.
*   **Hugging Face Integration:** Utilize pre-trained models on Hugging Face for easy access and integration.
*   **ComfyUI Support:** Seamlessly integrate Wan2.2 into your ComfyUI workflows.
*   **Diffusers Integration:** Utilize the powerful features within the Diffusers library.

## Quick Start

### Installation

1.  Clone the repository:

    ```bash
    git clone https://github.com/Wan-Video/Wan2.2.git
    cd Wan2.2
    ```

2.  Install dependencies:

    ```bash
    # Ensure torch >= 2.4.0
    # If the installation of `flash_attn` fails, try installing the other packages first and install `flash_attn` last
    pip install -r requirements.txt
    # If you want to use CosyVoice to synthesize speech for Speech-to-Video Generation, please install requirements_s2v.txt additionally
    pip install -r requirements_s2v.txt
    ```

### Model Download

Download the desired models from Hugging Face or ModelScope using the links in the original README (see below for a list).

### Generation Examples

*   **Text-to-Video (T2V):**

    ```bash
    python generate.py --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --offload_model True --convert_model_dtype --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
    ```

*   **Image-to-Video (I2V):**

    ```bash
    python generate.py --task i2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-I2V-A14B --offload_model True --convert_model_dtype --image examples/i2v_input.JPG --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
    ```

*   **Text-Image-to-Video (TI2V):**

    ```bash
    python generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ./Wan2.2-TI2V-5B --offload_model True --convert_model_dtype --t5_cpu --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage"
    ```

*   **Speech-to-Video (S2V):**

    ```bash
    python generate.py --task s2v-14B --size 1024*704 --ckpt_dir ./Wan2.2-S2V-14B/ --offload_model True --convert_model_dtype --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard."  --image "examples/i2v_input.JPG" --audio "examples/talk.wav"
    ```

### Model Download Links

| Models              | Download Links                                                                                                                              | Description |
|--------------------|---------------------------------------------------------------------------------------------------------------------------------------------|-------------|
| T2V-A14B    | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B)    🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-T2V-A14B)    | Text-to-Video MoE model, supports 480P & 720P |
| I2V-A14B    | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B)    🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-I2V-A14B)    | Image-to-Video MoE model, supports 480P & 720P |
| TI2V-5B     | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B)     🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-TI2V-5B)     | High-compression VAE, T2V+I2V, supports 720P |
| S2V-14B     | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-S2V-14B)     🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-S2V-14B)     | Speech-to-Video model, supports 480P & 720P |

> **Note:** The TI2V-5B model supports 720P video generation at 24 FPS.

## Community & Support

*   **Discord:** Join our [Discord](https://discord.gg/AKNgpMK4Yj) to connect with the community.
*   **WeChat:**  [WeChat](https://gw.alicdn.com/imgextra/i2/O1CN01tqjWFi1ByuyehkTSB_!!6000000000015-0-tps-611-1279.jpg)
*   Explore community contributions like [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio), [Kijai's ComfyUI WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper), [Cache-dit](https://github.com/vipshop/cache-dit), and [FastVideo](https://github.com/hao-ai-lab/FastVideo).

## License

Wan2.2 is released under the Apache 2.0 License. See the [LICENSE.txt](LICENSE.txt) file for details.

## Citation

```
@article{wan2025,
      title={Wan: Open and Advanced Large-Scale Video Generative Models}, 
      author={Team Wan and Ang Wang and Baole Ai and Bin Wen and Chaojie Mao and Chen-Wei Xie and Di Chen and Feiwu Yu and Haiming Zhao and Jianxiao Yang and Jianyuan Zeng and Jiayu Wang and Jingfeng Zhang and Jingren Zhou and Jinkai Wang and Jixuan Chen and Kai Zhu and Kang Zhao and Keyu Yan and Lianghua Huang and Mengyang Feng and Ningyi Zhang and Pandeng Li and Pingyu Wu and Ruihang Chu and Ruili Feng and Shiwei Zhang and Siyang Sun and Tao Fang and Tianxing Wang and Tianyi Gui and Tingyu Weng and Tong Shen and Wei Lin and Wei Wang and Wei Wang and Wenmeng Zhou and Wente Wang and Wenting Shen and Wenyuan Yu and Xianzhong Shi and Xiaoming Huang and Xin Xu and Yan Kou and Yangyu Lv and Yifei Li and Yijing Liu and Yiming Wang and Yingya Zhang and Yitong Huang and Yong Li and You Wu and Yu Liu and Yulin Pan and Yun Zheng and Yuntao Hong and Yupeng Shi and Yutong Feng and Zeyinzi Jiang and Zhen Han and Zhi-Fan Wu and Ziyu Liu},
      journal = {arXiv preprint arXiv:2503.20314},
      year={2025}
}