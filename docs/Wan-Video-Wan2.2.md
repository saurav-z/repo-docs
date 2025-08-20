# Wan2.2: Unleashing Cinematic-Quality Video Generation with Cutting-Edge AI

**Wan2.2** is a powerful, open-source video generation model pushing the boundaries of AI video creation, enabling users to generate stunning, high-quality videos from text and images. Check out the original repository [here](https://github.com/Wan-Video/Wan2.2).

**Key Features:**

*   🚀 **Mixture-of-Experts (MoE) Architecture:** Boosts model capacity and efficiency by dividing the denoising process between specialized experts, optimizing performance without increasing computational cost.
*   🎬 **Cinematic-Level Aesthetics:** Generates videos with customizable cinematic styles using meticulously curated data, allowing for precise control over lighting, composition, and color.
*   📈 **Enhanced Motion Generation:** Trained on a significantly larger dataset (65.6% more images, 83.2% more videos), Wan2.2 excels at generating complex and diverse motions, achieving top performance among open-source and closed-source models.
*   ⚡ **Efficient High-Definition Hybrid TI2V:**  Generates 720P videos at 24fps with consumer-grade GPUs, making it ideal for both industrial and academic use cases.

**Quick Links:**

*   🌐 [Wan](https://wan.video)
*   🖥️ [GitHub](https://github.com/Wan-Video/Wan2.2)
*   🤗 [Hugging Face](https://huggingface.co/Wan-AI/)
*   🤖 [ModelScope](https://modelscope.cn/organization/Wan-AI)
*   📑 [Paper](https://arxiv.org/abs/2503.20314)
*   📑 [Blog](https://wan.video/welcome?spm=a2ty_o02.30011076.0.0.6c9ee41eCcluqg)
*   💬 [Discord](https://discord.gg/AKNgpMK4Yj)

---

## Latest News

*   **July 28, 2025:** 👋 [HF space](https://huggingface.co/spaces/Wan-AI/Wan-2.2-5B) using the TI2V-5B model is open.
*   **July 28, 2025:** 👋 Wan2.2 is integrated into ComfyUI ([CN](https://docs.comfy.org/zh-CN/tutorials/video/wan/wan2_2) | [EN](https://docs.comfy.org/tutorials/video/wan/wan2_2)).
*   **July 28, 2025:** 👋 Wan2.2's T2V, I2V and TI2V have been integrated into Diffusers ([T2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) | [I2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers) | [TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers)).
*   **July 28, 2025:** 👋 Inference code and model weights of **Wan2.2** have been released.

## Community Works

*   [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) provides comprehensive support for Wan 2.2.
*   [Kijai's ComfyUI WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper) is an alternative implementation of Wan models for ComfyUI.

## Getting Started

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
    ```

### Model Download

Download models from Hugging Face or ModelScope:

| Model           | Download Links                                                                                                                              | Description                                     |
|-----------------|---------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------|
| T2V-A14B        | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B)    🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-T2V-A14B)    | Text-to-Video MoE, supports 480P & 720P      |
| I2V-A14B        | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B)    🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-I2V-A14B)    | Image-to-Video MoE, supports 480P & 720P      |
| TI2V-5B         | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B)     🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-TI2V-5B)     | High-Compression VAE, T2V+I2V, supports 720P |

> **Note:** The TI2V-5B model supports 720P video generation at 24 FPS.

**Download with huggingface-cli:**

```bash
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.2-T2V-A14B --local-dir ./Wan2.2-T2V-A14B
```

**Download with modelscope-cli:**

```bash
pip install modelscope
modelscope download Wan-AI/Wan2.2-T2V-A14B --local_dir ./Wan2.2-T2V-A14B
```

### Run Text-to-Video Generation

**Example:**

*   **Single-GPU Inference:**

    ```bash
    python generate.py --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --offload_model True --convert_model_dtype --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
    ```

    *   **Multi-GPU Inference:**

    ```bash
    torchrun --nproc_per_node=8 generate.py --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
    ```

    *   **With Prompt Extension:**

      *   **Using Dashscope API:**
      ```bash
      DASH_API_KEY=your_key torchrun --nproc_per_node=8 generate.py  --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage" --use_prompt_extend --prompt_extend_method 'dashscope' --prompt_extend_target_lang 'zh'
      ```
      *   **Using a local model:**
      ```bash
      torchrun --nproc_per_node=8 generate.py  --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage" --use_prompt_extend --prompt_extend_method 'local_qwen' --prompt_extend_target_lang 'zh'
      ```

### Run Image-to-Video Generation

**Example:**

*   **Single-GPU Inference:**

    ```bash
    python generate.py --task i2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-I2V-A14B --offload_model True --convert_model_dtype --image examples/i2v_input.JPG --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
    ```

    *   **Multi-GPU Inference:**

    ```bash
    torchrun --nproc_per_node=8 generate.py --task i2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-I2V-A14B --image examples/i2v_input.JPG --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
    ```

    *   **Image-to-Video Generation without prompt:**

      ```bash
      DASH_API_KEY=your_key torchrun --nproc_per_node=8 generate.py --task i2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-I2V-A14B --prompt '' --image examples/i2v_input.JPG --dit_fsdp --t5_fsdp --ulysses_size 8 --use_prompt_extend --prompt_extend_method 'dashscope'
      ```

### Run Text-Image-to-Video Generation

**Example:**

*   **Single-GPU Text-to-Video Inference:**

    ```bash
    python generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ./Wan2.2-TI2V-5B --offload_model True --convert_model_dtype --t5_cpu --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage"
    ```

    *   **Single-GPU Image-to-Video Inference:**

    ```bash
    python generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ./Wan2.2-TI2V-5B --offload_model True --convert_model_dtype --t5_cpu --image examples/i2v_input.JPG --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
    ```

    *   **Multi-GPU Inference:**

    ```bash
    torchrun --nproc_per_node=8 generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ./Wan2.2-TI2V-5B --dit_fsdp --t5_fsdp --ulysses_size 8 --image examples/i2v_input.JPG --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
    ```

## Computational Efficiency

*   **Wan2.2** performance on different GPUs is provided in the image below.

    ```
    <div align="center">
        <img src="assets/comp_effic.png" alt="" style="width: 80%;" />
    </div>
    ```

## Core Technical Innovations

### 1. Mixture-of-Experts (MoE) Architecture

*   Leverages a MoE architecture for increased capacity and efficiency.
*   Employs two experts for denoising: a high-noise expert for overall layout and a low-noise expert for refining details.
*   This keeps inference costs similar to traditional dense models.

### 2. Efficient High-Definition Hybrid TI2V

*   Features a high-compression design, including a 5B dense model (TI2V-5B) with a high-compression Wan2.2-VAE.
*   Offers fast 720P@24fps video generation on consumer-grade GPUs, supporting both text-to-video and image-to-video tasks.

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

*   The models are licensed under the Apache 2.0 License.

## Acknowledgements

*   Thanks to the contributors of SD3, Qwen, umt5-xxl, diffusers and HuggingFace.

## Contact

*   Join the [Discord](https://discord.gg/AKNgpMK4Yj) or [WeChat groups](https://gw.alicdn.com/imgextra/i2/O1CN01tqjWFi1ByuyehkTSB_!!6000000000015-0-tps-611-1279.jpg) for support and updates!