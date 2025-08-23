# Wan2.2: Revolutionizing Video Generation with Open-Source Innovation

**Wan2.2** is a cutting-edge open-source video generation model that empowers users to create high-quality videos with unparalleled control and efficiency, all available at [Wan2.2 GitHub](https://github.com/Wan-Video/Wan2.2).

<p align="center">
    <img src="assets/logo.png" width="400"/>
</p>

**Quick Links:** 
[Wan Website](https://wan.video) | [GitHub](https://github.com/Wan-Video/Wan2.2) | [Hugging Face](https://huggingface.co/Wan-AI/) | [ModelScope](https://modelscope.cn/organization/Wan-AI) | [Paper](https://arxiv.org/abs/2503.20314) | [Blog](https://wan.video/welcome?spm=a2ty_o02.30011076.0.0.6c9ee41eCcluqg) | [Discord](https://discord.gg/AKNgpMK4Yj)

---

**Key Features:**

*   👍 **Mixture-of-Experts (MoE) Architecture:**  Leverages MoE for efficient scaling, enhancing model capacity without drastically increasing computational costs.
*   👍 **Cinematic-Level Aesthetics:** Trained with curated aesthetic data and detailed labels, enabling precise control over video style and aesthetics.
*   👍 **Enhanced Motion Generation:** Trained on significantly larger datasets, resulting in superior generalization across motion, semantics, and aesthetics.
*   👍 **Efficient High-Definition Hybrid TI2V:** Introduces a 5B model with a compression ratio of **16×16×4**, supporting 720P generation at 24fps and running on consumer-grade GPUs.

## Video Demos

<div align="center">
  <video src="https://github.com/user-attachments/assets/b63bfa58-d5d7-4de6-a1a2-98970b06d9a7" width="70%" poster=""> </video>
</div>

## 🔥 Latest News

*   **July 28, 2025:**  TI2V-5B model available on [Hugging Face Spaces](https://huggingface.co/spaces/Wan-AI/Wan-2.2-5B).
*   **July 28, 2025:**  Wan2.2 integrated into ComfyUI ([CN](https://docs.comfy.org/zh-CN/tutorials/video/wan/wan2_2) | [EN](https://docs.comfy.org/tutorials/video/wan/wan2_2)).
*   **July 28, 2025:** Wan2.2's T2V, I2V, and TI2V integrated into Diffusers ([T2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) | [I2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers) | [TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers)).
*   **July 28, 2025:** Inference code and model weights of Wan2.2 released.

## Community Contributions

Explore how others are leveraging Wan2.2:

*   [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio): Comprehensive support for Wan 2.2, including low-GPU-memory layer-by-layer offload, FP8 quantization, sequence parallelism, LoRA training, full training.
*   [Kijai's ComfyUI WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper): Alternative implementation of Wan models for ComfyUI.

If you are using Wan2.1 or Wan2.2, share your work with the community!

## 📑 Todo List
- Wan2.2 Text-to-Video
    - [x] Multi-GPU Inference code of the A14B and 14B models
    - [x] Checkpoints of the A14B and 14B models
    - [x] ComfyUI integration
    - [x] Diffusers integration
- Wan2.2 Image-to-Video
    - [x] Multi-GPU Inference code of the A14B model
    - [x] Checkpoints of the A14B model
    - [x] ComfyUI integration
    - [x] Diffusers integration
- Wan2.2 Text-Image-to-Video
    - [x] Multi-GPU Inference code of the 5B model
    - [x] Checkpoints of the 5B model
    - [x] ComfyUI integration
    - [x] Diffusers integration

## Getting Started: Run Wan2.2

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

Choose your desired model and download it from Hugging Face or ModelScope:

| Models              | Download Links                                                                                                                              | Description |
|--------------------|---------------------------------------------------------------------------------------------------------------------------------------------|-------------|
| T2V-A14B    | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B)    🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-T2V-A14B)    | Text-to-Video MoE model, supports 480P & 720P |
| I2V-A14B    | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B)    🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-I2V-A14B)    | Image-to-Video MoE model, supports 480P & 720P |
| TI2V-5B     | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B)     🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-TI2V-5B)     | High-compression VAE, T2V+I2V, supports 720P |

> 💡Note: The TI2V-5B model supports 720P video generation at **24 FPS**.

**Using `huggingface-cli`:**

```bash
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.2-T2V-A14B --local-dir ./Wan2.2-T2V-A14B
```

**Using `modelscope-cli`:**

```bash
pip install modelscope
modelscope download Wan-AI/Wan2.2-T2V-A14B --local_dir ./Wan2.2-T2V-A14B
```

### Generation Examples

**1. Text-to-Video Generation**

**(1) Without Prompt Extension**

*   **Single-GPU:**

```bash
python generate.py  --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --offload_model True --convert_model_dtype --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
```

> 💡 This command can run on a GPU with at least 80GB VRAM.
> 💡If you encounter OOM (Out-of-Memory) issues, you can use the `--offload_model True`, `--convert_model_dtype` and `--t5_cpu` options to reduce GPU memory usage.

*   **Multi-GPU (FSDP + DeepSpeed Ulysses):**

```bash
torchrun --nproc_per_node=8 generate.py --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
```

**(2) Using Prompt Extension**

*   **Using Dashscope API:**
    -   Apply for a `dashscope.api_key` in advance ([EN](https://www.alibabacloud.com/help/en/model-studio/getting-started/first-api-call-to-qwen) | [CN](https://help.aliyun.com/zh/model-studio/getting-started/first-api-call-to-qwen)).
    -   Configure the environment variable `DASH_API_KEY` to specify the Dashscope API key. For users of Alibaba Cloud's international site, you also need to set the environment variable `DASH_API_URL` to 'https://dashscope-intl.aliyuncs.com/api/v1'. For more detailed instructions, please refer to the [dashscope document](https://www.alibabacloud.com/help/en/model-studio/developer-reference/use-qwen-by-calling-api?spm=a2c63.p38356.0.i1).
    -   Use the `qwen-plus` model for text-to-video tasks and `qwen-vl-max` for image-to-video tasks.
    -   You can modify the model used for extension with the parameter `--prompt_extend_model`. For example:
```bash
DASH_API_KEY=your_key torchrun --nproc_per_node=8 generate.py  --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage" --use_prompt_extend --prompt_extend_method 'dashscope' --prompt_extend_target_lang 'zh'
```

*   **Using a local model for extension:**
    -   By default, the Qwen model on HuggingFace is used for this extension. Users can choose Qwen models or other models based on the available GPU memory size.
    -   For text-to-video tasks, you can use models like `Qwen/Qwen2.5-14B-Instruct`, `Qwen/Qwen2.5-7B-Instruct` and `Qwen/Qwen2.5-3B-Instruct`.
    -   For image-to-video tasks, you can use models like `Qwen/Qwen2.5-VL-7B-Instruct` and `Qwen/Qwen2.5-VL-3B-Instruct`.
    -   Larger models generally provide better extension results but require more GPU memory.
    -   You can modify the model used for extension with the parameter `--prompt_extend_model` , allowing you to specify either a local model path or a Hugging Face model. For example:

```bash
torchrun --nproc_per_node=8 generate.py  --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage" --use_prompt_extend --prompt_extend_method 'local_qwen' --prompt_extend_target_lang 'zh'
```

**2. Image-to-Video Generation**

*   **Single-GPU:**

```bash
python generate.py --task i2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-I2V-A14B --offload_model True --convert_model_dtype --image examples/i2v_input.JPG --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
```

> This command can run on a GPU with at least 80GB VRAM.
> 💡For the Image-to-Video task, the `size` parameter represents the area of the generated video, with the aspect ratio following that of the original input image.

*   **Multi-GPU (FSDP + DeepSpeed Ulysses):**

```bash
torchrun --nproc_per_node=8 generate.py --task i2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-I2V-A14B --image examples/i2v_input.JPG --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
```

*   **Image-to-Video Generation without prompt**

```bash
DASH_API_KEY=your_key torchrun --nproc_per_node=8 generate.py --task i2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-I2V-A14B --prompt '' --image examples/i2v_input.JPG --dit_fsdp --t5_fsdp --ulysses_size 8 --use_prompt_extend --prompt_extend_method 'dashscope'
```

> 💡The model can generate videos solely from the input image. You can use prompt extension to generate prompt from the image.
> The process of prompt extension can be referenced [here](#2-using-prompt-extention).

**3. Text-Image-to-Video Generation**

*   **Single-GPU Text-to-Video:**

```bash
python generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ./Wan2.2-TI2V-5B --offload_model True --convert_model_dtype --t5_cpu --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage"
```

> 💡Unlike other tasks, the 720P resolution of the Text-Image-to-Video task is `1280*704` or `704*1280`.
> This command can run on a GPU with at least 24GB VRAM (e.g, RTX 4090 GPU).
> 💡If you are running on a GPU with at least 80GB VRAM, you can remove the `--offload_model True`, `--convert_model_dtype` and `--t5_cpu` options to speed up execution.

*   **Single-GPU Image-to-Video:**

```bash
python generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ./Wan2.2-TI2V-5B --offload_model True --convert_model_dtype --t5_cpu --image examples/i2v_input.JPG --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
```

> 💡If the image parameter is configured, it is an Image-to-Video generation; otherwise, it defaults to a Text-to-Video generation.
> 💡Similar to Image-to-Video, the `size` parameter represents the area of the generated video, with the aspect ratio following that of the original input image.

*   **Multi-GPU (FSDP + DeepSpeed Ulysses):**

```bash
torchrun --nproc_per_node=8 generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ./Wan2.2-TI2V-5B --dit_fsdp --t5_fsdp --ulysses_size 8 --image examples/i2v_input.JPG --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
```

> The process of prompt extension can be referenced [here](#2-using-prompt-extention).

## Computational Efficiency

The following table summarizes the computational efficiency of different **Wan2.2** models on various GPUs.  Results are in the format: **Total time (s) / peak GPU memory (GB)**.

<div align="center">
    <img src="assets/comp_effic.png" alt="" style="width: 80%;" />
</div>

> Notes:
> 1. Multi-GPU: 14B: `--ulysses_size 4/8 --dit_fsdp --t5_fsdp`, 5B: `--ulysses_size 4/8 --offload_model True --convert_model_dtype --t5_cpu`; Single-GPU: 14B: `--offload_model True --convert_model_dtype`, 5B: `--offload_model True --convert_model_dtype --t5_cpu`(--convert_model_dtype converts model parameter types to config.param_dtype);
> 2. Distributed testing uses built-in FSDP and Ulysses implementations; FlashAttention3 deployed on Hopper architecture GPUs;
> 3. Tests were run without the `--use_prompt_extend` flag;
> 4. Reported results are the average of multiple samples taken after the warm-up phase.

---

## Deep Dive: Understanding Wan2.2

**Wan2.2** offers significant advancements over Wan2.1, driven by architectural innovations and improved training.

##### (1) Mixture-of-Experts (MoE) Architecture

Wan2.2's A14B models feature a MoE architecture. The key idea is to split the denoising process across specialized expert models. In Wan2.2, this is tailored to the diffusion model's denoising process with a high-noise expert and a low-noise expert.

<div align="center">
    <img src="assets/moe_arch.png" alt="" style="width: 90%;" />
</div>

*   **High-Noise Expert:** Focuses on overall layout during the early denoising stages.
*   **Low-Noise Expert:** Refines video details in the later stages.

Each expert has approximately 14B parameters, leading to a total of 27B parameters, but only 14B active parameters per step, maintaining inference efficiency.

<div align="center">
    <img src="assets/moe_2.png" alt="" style="width: 90%;" />
</div>

To validate the effectiveness of the MoE architecture, four settings are compared based on their validation loss curves. The baseline **Wan2.1** model does not employ the MoE architecture. Among the MoE-based variants, the **Wan2.1 & High-Noise Expert** reuses the Wan2.1 model as the low-noise expert while uses the  Wan2.2's high-noise expert, while the **Wan2.1 & Low-Noise Expert** uses Wan2.1 as the high-noise expert and employ the Wan2.2's low-noise expert. The **Wan2.2 (MoE)** (our final version) achieves the lowest validation loss, indicating that its generated video distribution is closest to ground-truth and exhibits superior convergence.

##### (2) Efficient High-Definition Hybrid TI2V

For efficient deployment, the 5B dense model (TI2V-5B) is available. It utilizes a high-compression Wan2.2-VAE, achieving a $T\times H\times W$ compression ratio of $4\times16\times16$. This supports 720P video generation at 24fps with fast inference on consumer-grade GPUs.

<div align="center">
    <img src="assets/vae.png" alt="" style="width: 80%;" />
</div>

##### Comparisons to SOTAs

Wan2.2 surpasses leading closed-source models across various dimensions, according to our Wan-Bench 2.0 evaluation.

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

Licensed under the Apache 2.0 License. Please review the [LICENSE.txt](LICENSE.txt) for details and restrictions.

## Acknowledgements

Special thanks to the contributors of [SD3](https://huggingface.co/stabilityai/stable-diffusion-3-medium), [Qwen](https://huggingface.co/Qwen), [umt5-xxl](https://huggingface.co/google/umt5-xxl), [diffusers](https://github.com/huggingface/diffusers) and [HuggingFace](https://huggingface.co).

## Contact

Join our community: [Discord](https://discord.gg/AKNgpMK4Yj) or [WeChat groups](https://gw.alicdn.com/imgextra/i2/O1CN01tqjWFi1ByuyehkTSB_!!6000000000015-0-tps-611-1279.jpg).