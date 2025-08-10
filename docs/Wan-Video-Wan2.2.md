# Wan2.2: Revolutionizing Video Generation with Open-Source Innovation

**Unlock the power of advanced video generation with Wan2.2, a cutting-edge open-source model pushing the boundaries of cinematic quality.**  Check out the original repo here: [https://github.com/Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2)

<p align="center">
    <img src="assets/logo.png" width="400"/>
</p>

<p align="center">
    💜 [Wan Website](https://wan.video) &nbsp&nbsp ｜ &nbsp&nbsp 🖥️ [GitHub](https://github.com/Wan-Video/Wan2.2) &nbsp&nbsp  | &nbsp&nbsp 🤗 [Hugging Face](https://huggingface.co/Wan-AI/)&nbsp&nbsp | &nbsp&nbsp 🤖 [ModelScope](https://modelscope.cn/organization/Wan-AI) &nbsp&nbsp | &nbsp&nbsp 📑 [Paper](https://arxiv.org/abs/2503.20314) &nbsp&nbsp | &nbsp&nbsp 📑 [Blog](https://wan.video/welcome?spm=a2ty_o02.30011076.0.0.6c9ee41eCcluqg) &nbsp&nbsp |  &nbsp&nbsp 💬  [Discord](https://discord.gg/AKNgpMK4Yj)
    <br>
    📕 [User Guide (中文)](https://alidocs.dingtalk.com/i/nodes/jb9Y4gmKWrx9eo4dCql9LlbYJGXn6lpz) &nbsp&nbsp | &nbsp&nbsp 📘 [User Guide (English)](https://alidocs.dingtalk.com/i/nodes/EpGBa2Lm8aZxe5myC99MelA2WgN7R35y) &nbsp&nbsp | &nbsp&nbsp 💬 [WeChat (微信)](https://gw.alicdn.com/imgextra/i2/O1CN01tqjWFi1ByuyehkTSB_!!6000000000015-0-tps-611-1279.jpg)
<br>

---

## Key Features of Wan2.2

*   **Mixture-of-Experts (MoE) Architecture:** Innovative MoE design enhances model capacity while maintaining efficient computation, improving video quality and detail.
*   **Cinematic Aesthetics:**  Generate videos with customizable cinematic styles using meticulously curated aesthetic data for precise control over lighting, composition, and color.
*   **Enhanced Motion Generation:** Trained on a significantly larger dataset, Wan2.2 excels at generating complex and diverse motions, achieving top performance among open and closed source models.
*   **Efficient High-Definition TI2V:**  Open-source 5B model (TI2V-5B) with advanced compression technology for 720P video generation at 24fps on consumer-grade GPUs (e.g., RTX 4090), suitable for both industrial and academic applications.

## Video Demos

<div align="center">
  <video src="https://github.com/user-attachments/assets/b63bfa58-d5d7-4de6-a1a2-98970b06d9a7" width="70%" poster=""> </video>
</div>

## Latest News

*   **July 28, 2025:**  👋  HF space launched with the TI2V-5B model: [Hugging Face Space](https://huggingface.co/spaces/Wan-AI/Wan-2.2-5B)
*   **July 28, 2025:**  👋  Wan2.2 integrated into ComfyUI ([CN](https://docs.comfy.org/zh-CN/tutorials/video/wan/wan2_2) | [EN](https://docs.comfy.org/tutorials/video/wan/wan2_2)).
*   **July 28, 2025:** 👋  Wan2.2's T2V, I2V, and TI2V integrated into Diffusers ([T2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) | [I2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers) | [TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers)).
*   **July 28, 2025:**  👋  Inference code and model weights for Wan2.2 released.

## Community Works

Showcase your research or projects utilizing Wan2.1 or Wan2.2. Let us know so that we can help you reach more people.

*   [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) - Provides comprehensive support for Wan 2.2, including optimizations.
*   [Kijai's ComfyUI WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper) - Alternative Wan models implementation for ComfyUI.

## Todo List

*   Wan2.2 Text-to-Video
    *   \[x] Multi-GPU Inference code of the A14B and 14B models
    *   \[x] Checkpoints of the A14B and 14B models
    *   \[x] ComfyUI integration
    *   \[x] Diffusers integration
*   Wan2.2 Image-to-Video
    *   \[x] Multi-GPU Inference code of the A14B model
    *   \[x] Checkpoints of the A14B model
    *   \[x] ComfyUI integration
    *   \[x] Diffusers integration
*   Wan2.2 Text-Image-to-Video
    *   \[x] Multi-GPU Inference code of the 5B model
    *   \[x] Checkpoints of the 5B model
    *   \[x] ComfyUI integration
    *   \[x] Diffusers integration

## Getting Started with Wan2.2

### Installation

```bash
git clone https://github.com/Wan-Video/Wan2.2.git
cd Wan2.2
# Ensure torch >= 2.4.0
# If the installation of `flash_attn` fails, try installing the other packages first and install `flash_attn` last
pip install -r requirements.txt
```

### Model Downloads

| Model | Download Links | Description |
|---|---|---|
| T2V-A14B | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B)  🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-T2V-A14B) | Text-to-Video MoE Model (480P & 720P) |
| I2V-A14B | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B)  🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-I2V-A14B) | Image-to-Video MoE Model (480P & 720P) |
| TI2V-5B | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B)  🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-TI2V-5B) | High-Compression VAE (T2V + I2V, 720P) |

> **Note:** The TI2V-5B model supports 720P video generation at 24 FPS.

### Download using Hugging Face CLI:

```bash
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.2-T2V-A14B --local-dir ./Wan2.2-T2V-A14B
```

### Download using ModelScope CLI:

```bash
pip install modelscope
modelscope download Wan-AI/Wan2.2-T2V-A14B --local_dir ./Wan2.2-T2V-A14B
```

## Running Wan2.2

### Text-to-Video Generation

Supports `Wan2.2-T2V-A14B` (480P and 720P).

#### (1) Basic Inference (Without Prompt Extension)

-   **Single-GPU:**

```bash
python generate.py  --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --offload_model True --convert_model_dtype --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
```

>   Runs on GPU with at least 80GB VRAM. Use `--offload_model True`, `--convert_model_dtype`, and `--t5_cpu` for reduced GPU memory usage.

-   **Multi-GPU (FSDP + DeepSpeed Ulysses):**

```bash
torchrun --nproc_per_node=8 generate.py --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
```

#### (2) Prompt Extension (Recommended)

Extends prompts for richer video details. Choose from:

-   **Dashscope API:**
    *   Get a `dashscope.api_key` ([EN](https://www.alibabacloud.com/help/en/model-studio/getting-started/first-api-call-to-qwen) | [CN](https://help.aliyun.com/zh/model-studio/getting-started/first-api-call-to-qwen)).
    *   Set `DASH_API_KEY` environment variable. Set `DASH_API_URL` to 'https://dashscope-intl.aliyuncs.com/api/v1' for Alibaba Cloud international users.
    *   Use `qwen-plus` for text-to-video or `qwen-vl-max` for image-to-video.
    *   Modify prompt extension model with `--prompt_extend_model`.

```bash
DASH_API_KEY=your_key torchrun --nproc_per_node=8 generate.py  --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage" --use_prompt_extend --prompt_extend_method 'dashscope' --prompt_extend_target_lang 'zh'
```

-   **Local Model Extension:**
    *   Use Qwen models from Hugging Face (e.g., `Qwen/Qwen2.5-14B-Instruct`, etc.).  Larger models may provide better extension, but require more GPU memory.
    *   Specify the model with `--prompt_extend_model`.

```bash
torchrun --nproc_per_node=8 generate.py  --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage" --use_prompt_extend --prompt_extend_method 'local_qwen' --prompt_extend_target_lang 'zh'
```

### Image-to-Video Generation

Supports `Wan2.2-I2V-A14B` (480P and 720P).

-   **Single-GPU:**

```bash
python generate.py --task i2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-I2V-A14B --offload_model True --convert_model_dtype --image examples/i2v_input.JPG --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
```

>   Runs on GPU with at least 80GB VRAM.
>  The `size` parameter matches the aspect ratio of the input image.

-   **Multi-GPU (FSDP + DeepSpeed Ulysses):**

```bash
torchrun --nproc_per_node=8 generate.py --task i2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-I2V-A14B --image examples/i2v_input.JPG --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
```

-   **Image-to-Video Generation without prompt:**

```bash
DASH_API_KEY=your_key torchrun --nproc_per_node=8 generate.py --task i2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-I2V-A14B --prompt '' --image examples/i2v_input.JPG --dit_fsdp --t5_fsdp --ulysses_size 8 --use_prompt_extend --prompt_extend_method 'dashscope'
```

>   The model can generate videos solely from the input image. You can use prompt extension to generate a prompt from the image.

### Text-Image-to-Video Generation

Supports `Wan2.2-TI2V-5B` (720P).

-   **Single-GPU (Text-to-Video):**

```bash
python generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ./Wan2.2-TI2V-5B --offload_model True --convert_model_dtype --t5_cpu --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage"
```

>   720P resolution: `1280*704` or `704*1280`. Runs on GPU with at least 24GB VRAM (e.g., RTX 4090).

-   **Single-GPU (Image-to-Video):**

```bash
python generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ./Wan2.2-TI2V-5B --offload_model True --convert_model_dtype --t5_cpu --image examples/i2v_input.JPG --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
```

>   If the image parameter is configured, it is an Image-to-Video generation; otherwise, it defaults to a Text-to-Video generation.  The `size` parameter matches the aspect ratio of the input image.

-   **Multi-GPU (FSDP + DeepSpeed Ulysses):**

```bash
torchrun --nproc_per_node=8 generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ./Wan2.2-TI2V-5B --dit_fsdp --t5_fsdp --ulysses_size 8 --image examples/i2v_input.JPG --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
```

## Computational Efficiency on Different GPUs

<div align="center">
    <img src="assets/comp_effic.png" alt="" style="width: 80%;" />
</div>

>  Results presented in the format: **Total time (s) / peak GPU memory (GB)**.

>   Testing parameters: (1) Multi-GPU: 14B: `--ulysses_size 4/8 --dit_fsdp --t5_fsdp`, 5B: `--ulysses_size 4/8 --offload_model True --convert_model_dtype --t5_cpu`; Single-GPU: 14B: `--offload_model True --convert_model_dtype`, 5B: `--offload_model True --convert_model_dtype --t5_cpu`; (2) Distributed testing utilizes built-in FSDP and Ulysses, FlashAttention3 on Hopper architecture GPUs; (3) Tests run without `--use_prompt_extend`; (4) Results are averaged after warm-up.

---

## Technical Deep Dive: Understanding Wan2.2

Wan2.2 builds upon Wan2.1, introducing key innovations to enhance video generation:

### (1) Mixture-of-Experts (MoE) Architecture

<p align="center">
    <img src="assets/moe_arch.png" alt="" style="width: 90%;" />
</p>

Wan2.2 utilizes MoE, a technique to increase model parameters efficiently. The A14B models feature a two-expert design optimized for diffusion models:

*   **High-Noise Expert:** Handles early stages focusing on the overall layout.
*   **Low-Noise Expert:** Refines video details in later stages.
*   Each expert model contains ~14B parameters resulting in a 27B parameter model which uses 14B active parameters per step.

<p align="center">
    <img src="assets/moe_2.png" alt="" style="width: 90%;" />
</p>

The model dynamically switches between experts based on the Signal-to-Noise Ratio (SNR). The transition point, ${t}_{moe}$, occurs when  $t<{t}_{moe}$.

### (2) Efficient High-Definition Hybrid TI2V

To enable efficient deployment, a 5B dense model, i.e., TI2V-5B, is released. It is supported by a high-compression Wan2.2-VAE, which achieves a $T\times H\times W$ compression ratio of $4\times16\times16$, increasing the overall compression rate to 64 while maintaining high-quality video reconstruction. With an additional patchification layer, the total compression ratio of TI2V-5B reaches $4\times32\times32$. Without specific optimization, TI2V-5B can generate a 5-second 720P video in under 9 minutes on a single consumer-grade GPU, ranking among the fastest 720P@24fps video generation models.

<p align="center">
    <img src="assets/vae.png" alt="" style="width: 80%;" />
</p>

### Performance Benchmarking

<p align="center">
    <img src="assets/performance.png" alt="" style="width: 90%;" />
</p>

Wan2.2 demonstrates superior performance across key dimensions compared to leading commercial models on our new Wan-Bench 2.0.

---

## Citation

If you find Wan2.2 helpful, please cite our paper:

```
@article{wan2025,
      title={Wan: Open and Advanced Large-Scale Video Generative Models},
      author={Team Wan and Ang Wang and Baole Ai and Bin Wen and Chaojie Mao and Chen-Wei Xie and Di Chen and Feiwu Yu and Haiming Zhao and Jianxiao Yang and Jianyuan Zeng and Jiayu Wang and Jingfeng Zhang and Jingren Zhou and Jinkai Wang and Jixuan Chen and Kai Zhu and Kang Zhao and Keyu Yan and Lianghua Huang and Mengyang Feng and Ningyi Zhang and Pandeng Li and Pingyu Wu and Ruihang Chu and Ruili Feng and Shiwei Zhang and Siyang Sun and Tao Fang and Tianxing Wang and Tianyi Gui and Tingyu Weng and Tong Shen and Wei Lin and Wei Wang and Wei Wang and Wenmeng Zhou and Wente Wang and Wenting Shen and Wenyuan Yu and Xianzhong Shi and Xiaoming Huang and Xin Xu and Yan Kou and Yangyu Lv and Yifei Li and Yijing Liu and Yiming Wang and Yingya Zhang and Yitong Huang and Yong Li and You Wu and Yu Liu and Yulin Pan and Yun Zheng and Yuntao Hong and Yupeng Shi and Yutong Feng and Zeyinzi Jiang and Zhen Han and Zhi-Fan Wu and Ziyu Liu},
      journal = {arXiv preprint arXiv:2503.20314},
      year={2025}
}
```

## License

Wan2.2 models are licensed under the Apache 2.0 License. You are free to use the generated content within the bounds of this license.

## Acknowledgements

We thank the contributors of [SD3](https://huggingface.co/stabilityai/stable-diffusion-3-medium), [Qwen](https://huggingface.co/Qwen), [umt5-xxl](https://huggingface.co/google/umt5-xxl), [diffusers](https://github.com/huggingface/diffusers), and [HuggingFace](https://huggingface.co) for their open research.

## Contact

Join our [Discord](https://discord.gg/AKNgpMK4Yj) or [WeChat groups](https://gw.alicdn.com/imgextra/i2/O1CN01tqjWFi1ByuyehkTSB_!!6000000000015-0-tps-611-1279.jpg) to connect with the team.