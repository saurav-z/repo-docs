# Wan2.2: Generate Stunning Videos with Advanced AI

**Unleash the power of AI to create high-quality videos with Wan2.2, the open-source video generation model that sets a new standard for cinematic aesthetics and complex motion.** ([Original Repo](https://github.com/Wan-Video/Wan2.2))

[<img src="assets/logo.png" width="400" alt="Wan2.2 Logo"/>](https://github.com/Wan-Video/Wan2.2)

*   **[Paper](https://arxiv.org/abs/2503.20314)** | **[Blog](https://wan.video/welcome?spm=a2ty_o02.30011076.0.0.6c9ee41eCcluqg)** | **[Discord](https://discord.gg/AKNgpMK4Yj)** | **[User Guide (EN)](https://alidocs.dingtalk.com/i/nodes/EpGBa2Lm8aZxe5myC99MelA2WgN7R35y)** | **[User Guide (中文)](https://alidocs.dingtalk.com/i/nodes/jb9Y4gmKWrx9eo4dCql9LlbYJGXn6lpz)**
*   **[Hugging Face](https://huggingface.co/Wan-AI/)** | **[ModelScope](https://modelscope.cn/organization/Wan-AI)**

---

## Key Features of Wan2.2

*   👍 **Mixture-of-Experts (MoE) Architecture:** Leveraging MoE, Wan2.2 increases model capacity while maintaining computational efficiency, leading to improved performance.
*   👍 **Cinematic Aesthetics:** Enhanced with meticulously curated aesthetic data, enabling precise control over lighting, composition, and overall style for professional-looking videos.
*   👍 **Advanced Motion Generation:** Trained on a significantly expanded dataset (+65.6% images, +83.2% videos) to generate a wider range of complex motions and achieve top performance among open-source models.
*   👍 **Efficient High-Definition Hybrid TI2V:**  Includes a 5B model with a 16x16x4 compression ratio, supporting text-to-video and image-to-video generation at 720P/24fps, even on consumer GPUs.

## Video Demos

<div align="center">
  <video src="https://github.com/user-attachments/assets/b63bfa58-d5d7-4de6-a1a2-98970b06d9a7" width="70%" poster=""> </video>
</div>

## What's New

*   **July 28, 2025:** HF space with the TI2V-5B model released!
*   **July 28, 2025:** Wan2.2 integrated into ComfyUI.
*   **July 28, 2025:** Integration of Wan2.2's T2V, I2V, and TI2V into Diffusers.
*   **July 28, 2025:** Inference code and model weights of Wan2.2 released.

## Community Contributions

*   [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio): Provides extensive support for Wan 2.2, including optimization tools.
*   [Kijai's ComfyUI WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper): Alternative implementation for ComfyUI, focusing on cutting-edge optimizations.

## Todo List

A comprehensive list of planned features including:
*   Wan2.2 Text-to-Video
*   Wan2.2 Image-to-Video
*   Wan2.2 Text-Image-to-Video

## Run Wan2.2: Quickstart

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
    ```

### Model Download

| Models              | Download Links                                                                                                                              | Description |
|--------------------|---------------------------------------------------------------------------------------------------------------------------------------------|-------------|
| T2V-A14B    | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B)    🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-T2V-A14B)    | Text-to-Video MoE model, supports 480P & 720P |
| I2V-A14B    | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B)    🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-I2V-A14B)    | Image-to-Video MoE model, supports 480P & 720P |
| TI2V-5B     | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B)     🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-TI2V-5B)     | High-compression VAE, T2V+I2V, supports 720P |

> 💡Note:
> The TI2V-5B model supports 720P video generation at **24 FPS**.

Use huggingface-cli:
``` sh
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.2-T2V-A14B --local-dir ./Wan2.2-T2V-A14B
```

Use modelscope-cli:
``` sh
pip install modelscope
modelscope download Wan-AI/Wan2.2-T2V-A14B --local_dir ./Wan2.2-T2V-A14B
```

### Generate Videos: Basic Usage

#### Text-to-Video

*   **Single-GPU:**
    ```bash
    python generate.py  --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --offload_model True --convert_model_dtype --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
    ```
    > 💡 Requires a GPU with at least 80GB VRAM. Use `--offload_model True`, `--convert_model_dtype` and `--t5_cpu` to reduce VRAM usage.

*   **Multi-GPU (FSDP + DeepSpeed Ulysses):**
    ```bash
    torchrun --nproc_per_node=8 generate.py --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
    ```

#### Image-to-Video

*   **Single-GPU:**
    ```bash
    python generate.py --task i2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-I2V-A14B --offload_model True --convert_model_dtype --image examples/i2v_input.JPG --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
    ```
    > 💡 `size` represents the video area, respecting the aspect ratio of the input image.

*   **Multi-GPU (FSDP + DeepSpeed Ulysses):**
    ```bash
    torchrun --nproc_per_node=8 generate.py --task i2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-I2V-A14B --image examples/i2v_input.JPG --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
    ```
*   **Image-to-Video Generation without prompt:**
```bash
DASH_API_KEY=your_key torchrun --nproc_per_node=8 generate.py --task i2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-I2V-A14B --prompt '' --image examples/i2v_input.JPG --dit_fsdp --t5_fsdp --ulysses_size 8 --use_prompt_extend --prompt_extend_method 'dashscope'
```

> 💡 Model can generate videos solely from the input image. You can use prompt extension to generate prompt from the image.

#### Text-Image-to-Video

*   **Single-GPU (720P):**
    ```bash
    python generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ./Wan2.2-TI2V-5B --offload_model True --convert_model_dtype --t5_cpu --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage"
    ```
    > 💡 720P resolution: `1280*704` or `704*1280`. Requires at least 24GB VRAM (e.g., RTX 4090).  Omit `--offload_model True`, `--convert_model_dtype` and `--t5_cpu` on 80GB+ VRAM.
*   **Single-GPU (Image-to-Video):**
    ```bash
    python generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ./Wan2.2-TI2V-5B --offload_model True --convert_model_dtype --t5_cpu --image examples/i2v_input.JPG --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
    ```
    > 💡  `size` parameter respects the aspect ratio of the original image.

*   **Multi-GPU (FSDP + DeepSpeed Ulysses):**
    ```bash
    torchrun --nproc_per_node=8 generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ./Wan2.2-TI2V-5B --dit_fsdp --t5_fsdp --ulysses_size 8 --image examples/i2v_input.JPG --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
    ```

### Prompt Extension
Extend prompts to enrich details in generated videos. Two methods are provided:
*   **Dashscope API** You can apply for a `dashscope.api_key`. More details please refer to the [dashscope document](https://www.alibabacloud.com/help/en/model-studio/developer-reference/use-qwen-by-calling-api?spm=a2c63.p38356.0.i1).
*   **Local model** Use models like `Qwen/Qwen2.5-14B-Instruct`, `Qwen/Qwen2.5-7B-Instruct` and `Qwen/Qwen2.5-3B-Instruct`

Use the `--use_prompt_extend`, `--prompt_extend_method` and `--prompt_extend_target_lang` parameters to configure the prompt extension.

> For more detailed instructions and options, please refer to the original repository.

## Computational Efficiency

[<img src="assets/comp_effic.png" alt="Computational Efficiency Table" style="width: 80%;" />](assets/comp_effic.png)

> The table shows the total time (s) / peak GPU memory (GB) for different models and configurations.

## Introduction of Wan2.2

Wan2.2 introduces key innovations:

##### (1) Mixture-of-Experts (MoE) Architecture
*   Utilizes MoE to efficiently increase model capacity with specialized experts.
*   High-Noise Expert to focus on the overall layout and Low-Noise Expert to focus on refining the video details.

##### (2) Efficient High-Definition Hybrid TI2V
*   Wan2.2-VAE achieves high compression, making the model computationally efficient.

##### Comparisons to SOTAs

[<img src="assets/performance.png" alt="Performance Comparison" style="width: 90%;" />](assets/performance.png)

Wan2.2 achieves superior performance compared to leading models.

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

Licensed under the Apache 2.0 License. See [LICENSE.txt](LICENSE.txt) for details.

## Acknowledgements

Thanks to the contributors of SD3, Qwen, umt5-xxl, diffusers, and HuggingFace.

## Contact

Join our [Discord](https://discord.gg/AKNgpMK4Yj) or [WeChat groups](https://gw.alicdn.com/imgextra/i2/O1CN01tqjWFi1ByuyehkTSB_!!6000000000015-0-tps-611-1279.jpg) for more information.