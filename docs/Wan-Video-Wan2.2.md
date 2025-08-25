# Wan2.2: Unleashing Cinematic Video Generation with Open-Source Power

**Wan2.2 offers cutting-edge, open-source video generation, empowering users to create stunning cinematic visuals with unprecedented control.** ([Original Repo](https://github.com/Wan-Video/Wan2.2))

<p align="center">
    <img src="assets/logo.png" width="400"/>
</p>

**Key Features:**

*   💜 **Mixture-of-Experts (MoE) Architecture:** Utilizes specialized expert models for denoising, increasing model capacity while maintaining computational efficiency.
*   👍 **Cinematic-Level Aesthetics:** Trained on meticulously curated aesthetic data to enable fine-grained control over lighting, composition, color, and style.
*   👍 **Enhanced Complex Motion Generation:** Trained on significantly larger datasets (+65.6% images, +83.2% videos) for improved generalization across motion, semantics, and aesthetics.
*   👍 **Efficient High-Definition Hybrid TI2V:** Open-sources a 5B model (Wan2.2-VAE) achieving a 16x16x4 compression ratio, supporting text-to-video and image-to-video at 720P/24fps on consumer-grade GPUs.

## Key Updates and Integrations:

*   **July 28, 2025:**
    *   🚀 [HF space](https://huggingface.co/spaces/Wan-AI/Wan-2.2-5B) using the TI2V-5B model.
    *   ✅ Integrated into ComfyUI ([CN](https://docs.comfy.org/zh-CN/tutorials/video/wan/wan2_2) | [EN](https://docs.comfy.org/tutorials/video/wan/wan2_2)).
    *   ✅ Integrated into Diffusers ([T2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) | [I2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers) | [TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers)).
    *   🚀 Inference code and model weights of Wan2.2 have been released.

## Video Demos

<div align="center">
  <video src="https://github.com/user-attachments/assets/b63bfa58-d5d7-4de6-a1a2-98970b06d9a7" width="70%" poster=""> </video>
</div>


## Community Works

*   **DiffSynth-Studio:** Supports Wan 2.2, including low-GPU-memory offload, FP8 quantization, sequence parallelism, LoRA training, and full training.
*   **Kijai's ComfyUI WanVideoWrapper:** An alternative implementation of Wan models for ComfyUI, focusing on optimizations.

## 📑  Todo List (Completed)

*   Wan2.2 Text-to-Video (T2V)
    *   [x] Multi-GPU Inference code of the A14B and 14B models
    *   [x] Checkpoints of the A14B and 14B models
    *   [x] ComfyUI integration
    *   [x] Diffusers integration
*   Wan2.2 Image-to-Video (I2V)
    *   [x] Multi-GPU Inference code of the A14B model
    *   [x] Checkpoints of the A14B model
    *   [x] ComfyUI integration
    *   [x] Diffusers integration
*   Wan2.2 Text-Image-to-Video (TI2V)
    *   [x] Multi-GPU Inference code of the 5B model
    *   [x] Checkpoints of the 5B model
    *   [x] ComfyUI integration
    *   [x] Diffusers integration


## Getting Started

### Installation

```bash
git clone https://github.com/Wan-Video/Wan2.2.git
cd Wan2.2
# Ensure torch >= 2.4.0
# If the installation of `flash_attn` fails, try installing the other packages first and install `flash_attn` last
pip install -r requirements.txt
```

### Model Download

| Models      | Download Links                                                                                                 | Description                       |
| ----------- | -------------------------------------------------------------------------------------------------------------- | --------------------------------- |
| T2V-A14B    | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B)    🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-T2V-A14B)    | Text-to-Video MoE, 480P & 720P |
| I2V-A14B    | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B)    🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-I2V-A14B)    | Image-to-Video MoE, 480P & 720P |
| TI2V-5B     | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B)     🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-TI2V-5B)     | T2V+I2V, 720P                   |

>   💡 **Note:** TI2V-5B supports 720P video generation at 24 FPS.

**Download using Hugging Face CLI:**

```bash
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.2-T2V-A14B --local-dir ./Wan2.2-T2V-A14B
```

**Download using ModelScope CLI:**

```bash
pip install modelscope
modelscope download Wan-AI/Wan2.2-T2V-A14B --local_dir ./Wan2.2-T2V-A14B
```

### Run Text-to-Video Generation

#### (1) Basic Inference

```bash
python generate.py  --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --offload_model True --convert_model_dtype --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
```

>   💡  Requires at least 80GB VRAM.

>   💡 Use `--offload_model True`, `--convert_model_dtype`, and `--t5_cpu` for reduced GPU memory usage.

#### (2) Multi-GPU Inference with FSDP + DeepSpeed Ulysses

```bash
torchrun --nproc_per_node=8 generate.py --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
```

#### (3) Using Prompt Extension

**(A) Using Dashscope API**
* Apply for a `dashscope.api_key` ([EN](https://www.alibabacloud.com/help/en/model-studio/getting-started/first-api-call-to-qwen) | [CN](https://help.aliyun.com/zh/model-studio/getting-started/first-api-call-to-qwen)).
*   Set environment variable `DASH_API_KEY`.
*   For Alibaba Cloud's international site, also set `DASH_API_URL`.
*   Use `qwen-plus` (T2V) or `qwen-vl-max` (I2V).
```bash
DASH_API_KEY=your_key torchrun --nproc_per_node=8 generate.py  --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage" --use_prompt_extend --prompt_extend_method 'dashscope' --prompt_extend_target_lang 'zh'
```
** (B) Using Local Model**

*   Use Qwen models or other models based on available GPU memory.
*   T2V: `Qwen/Qwen2.5-14B-Instruct`, `Qwen/Qwen2.5-7B-Instruct`, `Qwen/Qwen2.5-3B-Instruct`.
*   I2V: `Qwen/Qwen2.5-VL-7B-Instruct`, `Qwen/Qwen2.5-VL-3B-Instruct`.
```bash
torchrun --nproc_per_node=8 generate.py  --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage" --use_prompt_extend --prompt_extend_method 'local_qwen' --prompt_extend_target_lang 'zh'
```


### Run Image-to-Video Generation

#### (1) Single-GPU Inference

```bash
python generate.py --task i2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-I2V-A14B --offload_model True --convert_model_dtype --image examples/i2v_input.JPG --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
```

>   💡 Size parameter follows input image aspect ratio.

#### (2) Multi-GPU Inference with FSDP + DeepSpeed Ulysses

```bash
torchrun --nproc_per_node=8 generate.py --task i2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-I2V-A14B --image examples/i2v_input.JPG --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
```

#### (3) Image-to-Video without Prompt (with prompt extension)

```bash
DASH_API_KEY=your_key torchrun --nproc_per_node=8 generate.py --task i2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-I2V-A14B --prompt '' --image examples/i2v_input.JPG --dit_fsdp --t5_fsdp --ulysses_size 8 --use_prompt_extend --prompt_extend_method 'dashscope'
```

### Run Text-Image-to-Video Generation

#### (1) Single-GPU

```bash
python generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ./Wan2.2-TI2V-5B --offload_model True --convert_model_dtype --t5_cpu --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage"
```

>   💡 720P resolution is 1280\*704 or 704\*1280.
>
>   💡 Requires 24GB VRAM (e.g., RTX 4090).
>
>   💡 Remove `--offload_model True`, `--convert_model_dtype`, and `--t5_cpu` for faster execution on 80GB+ VRAM GPUs.

#### (2) Single-GPU (Image-to-Video)

```bash
python generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ./Wan2.2-TI2V-5B --offload_model True --convert_model_dtype --t5_cpu --image examples/i2v_input.JPG --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
```

>   💡  Image-to-Video if the `--image` parameter is used.
>
>   💡  `size` follows input image aspect ratio.

#### (3) Multi-GPU Inference with FSDP + DeepSpeed Ulysses

```bash
torchrun --nproc_per_node=8 generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ./Wan2.2-TI2V-5B --dit_fsdp --t5_fsdp --ulysses_size 8 --image examples/i2v_input.JPG --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
```

## Computational Efficiency

<div align="center">
    <img src="assets/comp_effic.png" alt="" style="width: 80%;" />
</div>

>   Parameters:
>   (1) Multi-GPU: 14B: `--ulysses_size 4/8 --dit_fsdp --t5_fsdp`, 5B: `--ulysses_size 4/8 --offload_model True --convert_model_dtype --t5_cpu`; Single-GPU: 14B: `--offload_model True --convert_model_dtype`, 5B: `--offload_model True --convert_model_dtype --t5_cpu`
>   (--convert_model_dtype converts model parameter types to config.param_dtype);
>   (2) Distributed testing utilizes FSDP and Ulysses implementations.
>   (3) Tests without `--use_prompt_extend`.
>   (4) Results are averaged over multiple samples after warm-up.

## Deep Dive into Wan2.2

Wan2.2 builds upon Wan2.1 with advancements in video quality and model capabilities, driven by key innovations:

### 1. Mixture-of-Experts (MoE) Architecture

*   Implemented in A14B model series for more efficient scaling.
*   Employs high-noise expert for early denoising stages and a low-noise expert for detail refinement.
*   Each expert model (~14B parameters) results in a 27B parameter model with only 14B active per step, maintaining inference cost.

<div align="center">
    <img src="assets/moe_arch.png" alt="" style="width: 90%;" />
</div>

The transition point between the two experts is determined by the signal-to-noise ratio (SNR), a metric that decreases monotonically as the denoising step $t$ increases. At the beginning of the denoising process, $t$ is large and the noise level is high, so the SNR is at its minimum, denoted as ${SNR}_{min}$. In this stage, the high-noise expert is activated. We define a threshold step ${t}_{moe}$ corresponding to half of the ${SNR}_{min}$, and switch to the low-noise expert when $t<{t}_{moe}$.

<div align="center">
    <img src="assets/moe_2.png" alt="" style="width: 90%;" />
</div>

### 2. Efficient High-Definition Hybrid TI2V

*   TI2V-5B model for efficient deployment, supported by Wan2.2-VAE.
*   Achieves a 16x16x4 compression ratio.
*   Supports 720P/24fps generation on consumer GPUs.
*   Native support for text-to-video and image-to-video within a unified framework.

<div align="center">
    <img src="assets/vae.png" alt="" style="width: 80%;" />
</div>

### 3. Performance vs. Leading Models

<div align="center">
    <img src="assets/performance.png" alt="" style="width: 90%;" />
</div>

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

Licensed under the Apache 2.0 License. See [LICENSE.txt](LICENSE.txt) for full details.

## Acknowledgements

Thanks to the contributors of the SD3, Qwen, umt5-xxl, diffusers and HuggingFace repositories.

## Contact Us

Join our [Discord](https://discord.gg/AKNgpMK4Yj) or [WeChat groups](https://gw.alicdn.com/imgextra/i2/O1CN01tqjWFi1ByuyehkTSB_!!6000000000015-0-tps-611-1279.jpg)!