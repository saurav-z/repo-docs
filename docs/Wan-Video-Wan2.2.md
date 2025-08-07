# Wan2.2: Unleash Cinematic Video Generation with Open-Source AI

**Wan2.2** is an open-source, advanced video generation model revolutionizing the creation of high-quality videos.  Explore the cutting edge of AI video generation at [Wan2.2's GitHub Repository](https://github.com/Wan-Video/Wan2.2).

<p align="center">
    <img src="assets/logo.png" width="400"/>
</p>

**Key Features:**

*   🚀 **Mixture-of-Experts (MoE) Architecture:** Enhanced performance with a MoE architecture designed for video diffusion models, increasing model capacity while maintaining efficient computation.
*   🎬 **Cinematic-Level Aesthetics:** Generate videos with precise control over lighting, composition, color tones, and more, using meticulously curated aesthetic data.
*   🎥 **Complex Motion Generation:** Improved motion, semantics, and aesthetics thanks to training on significantly expanded datasets (+65.6% more images and +83.2% more videos).
*   ⚡️ **Efficient High-Definition Hybrid TI2V:**  The 5B model supports both text-to-video and image-to-video at 720P resolution with 24fps and runs on consumer-grade GPUs like the 4090.

**Explore Wan2.2:**

*   [**Wan**](https://wan.video)
*   🖥️ [**GitHub**](https://github.com/Wan-Video/Wan2.2)
*   🤗 [**Hugging Face**](https://huggingface.co/Wan-AI/)
*   🤖 [**ModelScope**](https://modelscope.cn/organization/Wan-AI)
*   📑 [**Paper**](https://arxiv.org/abs/2503.20314)
*   📑 [**Blog**](https://wan.video/welcome?spm=a2ty_o02.30011076.0.0.6c9ee41eCcluqg)
*   💬 [**Discord**](https://discord.gg/AKNgpMK4Yj)
*   📕 [**使用指南 (中文)**](https://alidocs.dingtalk.com/i/nodes/jb9Y4gmKWrx9eo4dCql9LlbYJGXn6lpz)
*   📘 [**User Guide (English)**](https://alidocs.dingtalk.com/i/nodes/EpGBa2Lm8aZxe5myC99MelA2WgN7R35y)
*   💬 [**WeChat (微信)**](https://gw.alicdn.com/imgextra/i2/O1CN01tqjWFi1ByuyehkTSB_!!6000000000015-0-tps-611-1279.jpg)

**Video Demos:**

<div align="center">
  <video src="https://github.com/user-attachments/assets/b63bfa58-d5d7-4de6-a1a2-98970b06d9a7" width="70%" poster=""> </video>
</div>

## Key Highlights

*   **Integrated into ComfyUI**:  Wan2.2 has been integrated into ComfyUI ([CN](https://docs.comfy.org/zh-CN/tutorials/video/wan/wan2_2) | [EN](https://docs.comfy.org/tutorials/video/wan/wan2_2))
*   **Diffusers Integration**:  Wan2.2's T2V, I2V and TI2V have been integrated into Diffusers ([T2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) | [I2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers) | [TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers))
*   **Open-Source Inference**: The inference code and model weights of **Wan2.2** have been released.

## Community Contributions

*   **DiffSynth-Studio**:  Provides support for Wan 2.2, including low-GPU-memory layer-by-layer offload, FP8 quantization, sequence parallelism, LoRA training, and full training.
*   **Kijai's ComfyUI WanVideoWrapper**: An alternative implementation of Wan models for ComfyUI.

## Getting Started

### Installation

```bash
git clone https://github.com/Wan-Video/Wan2.2.git
cd Wan2.2
pip install -r requirements.txt # Ensure torch >= 2.4.0.  If flash_attn fails, install other packages first.
```

### Model Download

| Models              | Download Links                                                                                                                              | Description |
|--------------------|---------------------------------------------------------------------------------------------------------------------------------------------|-------------|
| T2V-A14B    | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B)    🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-T2V-A14B)    | Text-to-Video MoE model, supports 480P & 720P |
| I2V-A14B    | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B)    🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-I2V-A14B)    | Image-to-Video MoE model, supports 480P & 720P |
| TI2V-5B     | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B)     🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-TI2V-5B)     | High-compression VAE, T2V+I2V, supports 720P |

> 💡Note: The TI2V-5B model supports 720P video generation at **24 FPS**.

Download models using huggingface-cli:

```bash
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.2-T2V-A14B --local-dir ./Wan2.2-T2V-A14B
```

Download models using modelscope-cli:

```bash
pip install modelscope
modelscope download Wan-AI/Wan2.2-T2V-A14B --local_dir ./Wan2.2-T2V-A14B
```

## Run Text-to-Video Generation

This repository supports the `Wan2.2-T2V-A14B` Text-to-Video model, supporting 480P and 720P resolutions.

### (1) Basic Inference

*   **Single-GPU:**

```bash
python generate.py  --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --offload_model True --convert_model_dtype --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
```

> 💡  Runs on a GPU with at least 80GB VRAM.  Use `--offload_model True`, `--convert_model_dtype`, and `--t5_cpu` to reduce memory usage if needed.

*   **Multi-GPU (FSDP + DeepSpeed Ulysses):**

```bash
torchrun --nproc_per_node=8 generate.py --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
```

### (2) Using Prompt Extension

Extend prompts for richer details.  Choose from:

*   **Dashscope API:** (Requires API key)

    *   Apply for a `dashscope.api_key` ([EN](https://www.alibabacloud.com/help/en/model-studio/getting-started/first-api-call-to-qwen) | [CN](https://help.aliyun.com/zh/model-studio/getting-started/first-api-call-to-qwen))
    *   Set `DASH_API_KEY` and, for Alibaba Cloud international, `DASH_API_URL`.
    *   Use `qwen-plus` (T2V) or `qwen-vl-max` (I2V) models.
    *   Modify model with `--prompt_extend_model`.

    ```bash
    DASH_API_KEY=your_key torchrun --nproc_per_node=8 generate.py  --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage" --use_prompt_extend --prompt_extend_method 'dashscope' --prompt_extend_target_lang 'zh'
    ```

*   **Local Model for Extension:**

    *   Use Qwen or other models based on GPU memory.
    *   Use `Qwen/Qwen2.5-14B-Instruct`, `Qwen/Qwen2.5-7B-Instruct`, `Qwen/Qwen2.5-3B-Instruct` (T2V), or `Qwen/Qwen2.5-VL-7B-Instruct`, `Qwen/Qwen2.5-VL-3B-Instruct` (I2V)
    *   Modify the model with `--prompt_extend_model`.

    ```bash
    torchrun --nproc_per_node=8 generate.py  --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage" --use_prompt_extend --prompt_extend_method 'local_qwen' --prompt_extend_target_lang 'zh'
    ```

## Run Image-to-Video Generation

This repository supports the `Wan2.2-I2V-A14B` Image-to-Video model.

*   **Single-GPU:**

```bash
python generate.py --task i2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-I2V-A14B --offload_model True --convert_model_dtype --image examples/i2v_input.JPG --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
```

> 💡  The `size` parameter maintains the input image's aspect ratio.

*   **Multi-GPU (FSDP + DeepSpeed Ulysses):**

```bash
torchrun --nproc_per_node=8 generate.py --task i2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-I2V-A14B --image examples/i2v_input.JPG --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
```

*   **Image-to-Video Generation without prompt:**

```bash
DASH_API_KEY=your_key torchrun --nproc_per_node=8 generate.py --task i2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-I2V-A14B --prompt '' --image examples/i2v_input.JPG --dit_fsdp --t5_fsdp --ulysses_size 8 --use_prompt_extend --prompt_extend_method 'dashscope'
```

> 💡The model can generate videos solely from the input image.

## Run Text-Image-to-Video Generation

This repository supports the `Wan2.2-TI2V-5B` model at 720P.

*   **Single-GPU Text-to-Video:**

```bash
python generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ./Wan2.2-TI2V-5B --offload_model True --convert_model_dtype --t5_cpu --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage"
```

> 💡  720P resolution for this task is `1280*704` or `704*1280`. Runs on a GPU with at least 24GB VRAM (e.g., RTX 4090).

*   **Single-GPU Image-to-Video:**

```bash
python generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ./Wan2.2-TI2V-5B --offload_model True --convert_model_dtype --t5_cpu --image examples/i2v_input.JPG --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
```

> 💡  If the image parameter is configured, it is an Image-to-Video generation; otherwise, it defaults to a Text-to-Video generation.

*   **Multi-GPU (FSDP + DeepSpeed Ulysses):**

```bash
torchrun --nproc_per_node=8 generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ./Wan2.2-TI2V-5B --dit_fsdp --t5_fsdp --ulysses_size 8 --image examples/i2v_input.JPG --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
```

## Computational Efficiency

See the `assets/comp_effic.png` image in the repository for efficiency tests on different GPUs.

## About Wan2.2

**Wan2.2** builds on Wan2.1 with notable improvements. This includes:

*   **Mixture-of-Experts (MoE) Architecture:** Using MoE to enhance performance.
*   **Efficient High-Definition Hybrid TI2V:** This model is supported by a high-compression Wan2.2-VAE, which achieves a $T\times H\times W$ compression ratio of $4\times16\times16$, increasing the overall compression rate to 64 while maintaining high-quality video reconstruction. With an additional patchification layer, the total compression ratio of TI2V-5B reaches $4\times32\times32$.
*   **Enhanced Data and Training:**  Improved datasets for complex motion generation.
*   **Performance:** Wan2.2 achieves superior performance compared to leading models.

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

The models are licensed under the Apache 2.0 License.  See [LICENSE.txt](LICENSE.txt) for details.

## Acknowledgements

Thanks to the contributors of [SD3](https://huggingface.co/stabilityai/stable-diffusion-3-medium), [Qwen](https://huggingface.co/Qwen), [umt5-xxl](https://huggingface.co/google/umt5-xxl), [diffusers](https://github.com/huggingface/diffusers) and [HuggingFace](https://huggingface.co).

## Contact

Join our [Discord](https://discord.gg/AKNgpMK4Yj) or [WeChat groups](https://gw.alicdn.com/imgextra/i2/O1CN01tqjWFi1ByuyehkTSB_!!6000000000015-0-tps-611-1279.jpg) to connect with the team.