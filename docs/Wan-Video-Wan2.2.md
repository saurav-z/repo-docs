# Wan2.2: Unleash Cinematic-Quality Video Generation with Advanced AI

**Create stunning videos with unparalleled quality and control using Wan2.2, a cutting-edge video generative model. ([Original Repo](https://github.com/Wan-Video/Wan2.2))**

<p align="center">
    <img src="assets/logo.png" width="400"/>
<p>

<p align="center">
    💜 <a href="https://wan.video"><b>Wan</b></a> &nbsp&nbsp ｜ &nbsp&nbsp 🖥️ <a href="https://github.com/Wan-Video/Wan2.2">GitHub</a> &nbsp&nbsp  | &nbsp&nbsp🤗 <a href="https://huggingface.co/Wan-AI/">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/organization/Wan-AI">ModelScope</a>&nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://arxiv.org/abs/2503.20314">Paper</a> &nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://wan.video/welcome?spm=a2ty_o02.30011076.0.0.6c9ee41eCcluqg">Blog</a> &nbsp&nbsp |  &nbsp&nbsp 💬  <a href="https://discord.gg/AKNgpMK4Yj">Discord</a>&nbsp&nbsp
    <br>
    📕 <a href="https://alidocs.dingtalk.com/i/nodes/jb9Y4gmKWrx9eo4dCql9LlbYJGXn6lpz">使用指南(中文)</a>&nbsp&nbsp | &nbsp&nbsp 📘 <a href="https://alidocs.dingtalk.com/i/nodes/EpGBa2Lm8aZxe5myC99MelA2WgN7R35y">User Guide(English)</a>&nbsp&nbsp | &nbsp&nbsp💬 <a href="https://gw.alicdn.com/imgextra/i2/O1CN01tqjWFi1ByuyehkTSB_!!6000000000015-0-tps-611-1279.jpg">WeChat(微信)</a>&nbsp&nbsp
<br>

-----

Wan2.2 is a major leap forward in video generation, offering powerful new features and enhanced performance.  Leveraging advanced AI, Wan2.2 delivers cinematic-quality video creation with improved control and efficiency.

**Key Features:**

*   **Mixture-of-Experts (MoE) Architecture:**  Enhances model capacity while maintaining low computational cost, with specialized experts for different stages of the denoising process.
*   **Cinematic Aesthetics:** Achieve precise control over video style with meticulously curated aesthetic data, including lighting, composition, and color grading.
*   **Enhanced Motion Generation:** Trained on significantly expanded datasets, Wan2.2 excels in generating complex and realistic motions, surpassing existing open-source and closed-source models.
*   **Efficient High-Definition (HD) Hybrid TI2V:**  Introducing a 5B model (TI2V-5B) with a 16x16x4 compression ratio for both text-to-video and image-to-video generation at 720P resolution and 24fps, even on consumer-grade GPUs.

## Video Demos

<div align="center">
  <video src="https://github.com/user-attachments/assets/b63bfa58-d5d7-4de6-a1a2-98970b06d9a7" width="70%" poster=""> </video>
</div>

## 🔥 Latest Updates

*   **July 28, 2025:**  HF Space:  [HF space](https://huggingface.co/spaces/Wan-AI/Wan-2.2-5B) using the TI2V-5B model.
*   **July 28, 2025:** ComfyUI Integration:  Integrated into ComfyUI ([CN](https://docs.comfy.org/zh-CN/tutorials/video/wan/wan2_2) | [EN](https://docs.comfy.org/tutorials/video/wan/wan2_2)).
*   **July 28, 2025:** Diffusers Integration: T2V, I2V and TI2V integrated into Diffusers ([T2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) | [I2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers) | [TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers)).
*   **July 28, 2025:** Inference Code & Weights: Released the inference code and model weights for Wan2.2.

## Community Works

Explore community contributions built on Wan2.1 and Wan2.2.  Share your projects with us!

*   [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio): Comprehensive support for Wan 2.2, including features like low-GPU-memory layer-by-layer offload, FP8 quantization, sequence parallelism, LoRA training, and full training.
*   [Kijai's ComfyUI WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper): Alternative implementation of Wan models optimized for ComfyUI.

## 📑 Todo List

*   Wan2.2 Text-to-Video
    -   [x] Multi-GPU Inference code of the A14B and 14B models
    -   [x] Checkpoints of the A14B and 14B models
    -   [x] ComfyUI integration
    -   [x] Diffusers integration
*   Wan2.2 Image-to-Video
    -   [x] Multi-GPU Inference code of the A14B model
    -   [x] Checkpoints of the A14B model
    -   [x] ComfyUI integration
    -   [x] Diffusers integration
*   Wan2.2 Text-Image-to-Video
    -   [x] Multi-GPU Inference code of the 5B model
    -   [x] Checkpoints of the 5B model
    -   [x] ComfyUI integration
    -   [x] Diffusers integration

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
    ```

### Model Download

| Model         | Download Links                                                                                                                               | Description                     |
| ------------- | -------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------- |
| T2V-A14B      | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B)    🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-T2V-A14B)        | Text-to-Video MoE model, 480P & 720P |
| I2V-A14B      | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B)    🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-I2V-A14B)        | Image-to-Video MoE model, 480P & 720P |
| TI2V-5B       | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B)     🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-TI2V-5B)         | High-compression VAE, T2V+I2V, 720P |

> **Note:** The TI2V-5B model supports 720P video generation at **24 FPS**.

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

### Run Text-to-Video Generation

**(1) Without Prompt Extension (Basic)**

*   **Single-GPU Inference:**

```bash
python generate.py  --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --offload_model True --convert_model_dtype --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
```

> **Note:** This command requires a GPU with at least 80GB VRAM.  Use `--offload_model True`, `--convert_model_dtype` and `--t5_cpu` to reduce memory usage.

*   **Multi-GPU Inference (FSDP + DeepSpeed Ulysses):**

```bash
torchrun --nproc_per_node=8 generate.py --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
```

**(2) Using Prompt Extension (Recommended)**

Extend prompts for enhanced video detail and quality.  Choose from two methods:

*   **Dashscope API Extension:**
    *   Obtain a `dashscope.api_key` ([EN](https://www.alibabacloud.com/help/en/model-studio/getting-started/first-api-call-to-qwen) | [CN](https://help.aliyun.com/zh/model-studio/getting-started/first-api-call-to-qwen)).
    *   Set the `DASH_API_KEY` environment variable.  For Alibaba Cloud international users, set `DASH_API_URL` to `https://dashscope-intl.aliyuncs.com/api/v1`.
    *   Use `qwen-plus` for text-to-video and `qwen-vl-max` for image-to-video tasks.
    *   Modify the extension model with the `--prompt_extend_model` parameter (e.g.,  `--prompt_extend_model qwen-plus`).

```bash
DASH_API_KEY=your_key torchrun --nproc_per_node=8 generate.py  --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage" --use_prompt_extend --prompt_extend_method 'dashscope' --prompt_extend_target_lang 'zh'
```

*   **Local Model Extension:**
    *   Use models like `Qwen/Qwen2.5-14B-Instruct`, `Qwen/Qwen2.5-7B-Instruct`,  or `Qwen/Qwen2.5-3B-Instruct` for text-to-video, or `Qwen/Qwen2.5-VL-7B-Instruct` and `Qwen/Qwen2.5-VL-3B-Instruct` for image-to-video (larger models generally provide better results).
    *   Use `--prompt_extend_model` to specify a local model path or Hugging Face model.

```bash
torchrun --nproc_per_node=8 generate.py  --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage" --use_prompt_extend --prompt_extend_method 'local_qwen' --prompt_extend_target_lang 'zh'
```

### Run Image-to-Video Generation

*   **Single-GPU Inference:**

```bash
python generate.py --task i2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-I2V-A14B --offload_model True --convert_model_dtype --image examples/i2v_input.JPG --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
```

> **Note:**  The `size` parameter adjusts video area with the input image's aspect ratio.

*   **Multi-GPU Inference (FSDP + DeepSpeed Ulysses):**

```bash
torchrun --nproc_per_node=8 generate.py --task i2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-I2V-A14B --image examples/i2v_input.JPG --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
```

*   **Image-to-Video Generation without prompt:**

```bash
DASH_API_KEY=your_key torchrun --nproc_per_node=8 generate.py --task i2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-I2V-A14B --prompt '' --image examples/i2v_input.JPG --dit_fsdp --t5_fsdp --ulysses_size 8 --use_prompt_extend --prompt_extend_method 'dashscope'
```

> **Note:** The model can generate videos from just the input image.

### Run Text-Image-to-Video Generation

*   **Single-GPU Text-to-Video Inference:**

```bash
python generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ./Wan2.2-TI2V-5B --offload_model True --convert_model_dtype --t5_cpu --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage"
```

> **Note:**  For Text-Image-to-Video, the 720P resolution is `1280*704` or `704*1280`.  This can run on a GPU with as little as 24GB VRAM (e.g., RTX 4090). Omit `--offload_model True`, `--convert_model_dtype` and `--t5_cpu` for GPUs with more VRAM for faster execution.

*   **Single-GPU Image-to-Video Inference:**

```bash
python generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ./Wan2.2-TI2V-5B --offload_model True --convert_model_dtype --t5_cpu --image examples/i2v_input.JPG --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
```

>  Include the `--image` parameter for Image-to-Video generation; otherwise, it defaults to Text-to-Video. The `size` parameter reflects the area of the generated video, keeping the aspect ratio of the original image.

*   **Multi-GPU Inference (FSDP + DeepSpeed Ulysses):**

```bash
torchrun --nproc_per_node=8 generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ./Wan2.2-TI2V-5B --dit_fsdp --t5_fsdp --ulysses_size 8 --image examples/i2v_input.JPG --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
```

## Computational Efficiency

This table shows the performance of different models on various GPUs.

<div align="center">
    <img src="assets/comp_effic.png" alt="" style="width: 80%;" />
</div>

> Test configuration details:
> (1) Multi-GPU: 14B: `--ulysses_size 4/8 --dit_fsdp --t5_fsdp`, 5B: `--ulysses_size 4/8 --offload_model True --convert_model_dtype --t5_cpu`; Single-GPU: 14B: `--offload_model True --convert_model_dtype`, 5B: `--offload_model True --convert_model_dtype --t5_cpu`
> (2) The distributed testing utilizes the built-in FSDP and Ulysses implementations, with FlashAttention3 deployed on Hopper architecture GPUs;
> (3) Tests were run without the `--use_prompt_extend` flag;
> (4) Reported results are the average of multiple samples taken after the warm-up phase.

## Technical Details

**Wan2.2** builds upon Wan2.1 with significant improvements in generation quality and model capabilities. Key innovations include:

**(1) Mixture-of-Experts (MoE) Architecture:**  A two-expert design (high-noise and low-noise) in the A14B models that improves video quality while maintaining efficient inference.

<div align="center">
    <img src="assets/moe_arch.png" alt="" style="width: 90%;" />
</div>

*   The transition point between experts is determined by the Signal-to-Noise Ratio (SNR), switching from the high-noise expert to the low-noise expert as the denoising step increases.

<div align="center">
    <img src="assets/moe_2.png" alt="" style="width: 90%;" />
</div>

**(2) Efficient High-Definition Hybrid TI2V:** The TI2V-5B model utilizes a high-compression Wan2.2-VAE, achieving a 4x16x16 compression ratio (total compression ratio of 64).  This enables fast 720P@24fps video generation on consumer-grade GPUs.

<div align="center">
    <img src="assets/vae.png" alt="" style="width: 80%;" />
</div>

**(3) Performance Comparison:**  Wan2.2 demonstrates superior performance compared to leading commercial video generation models on Wan-Bench 2.0.  See the performance comparison chart.

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

The models are licensed under the Apache 2.0 License.  Your generated content is yours, provided you comply with the license terms.  See `LICENSE.txt`.

## Acknowledgements

Thank you to the contributors of [SD3](https://huggingface.co/stabilityai/stable-diffusion-3-medium), [Qwen](https://huggingface.co/Qwen), [umt5-xxl](https://huggingface.co/google/umt5-xxl), [diffusers](https://github.com/huggingface/diffusers), and [HuggingFace](https://huggingface.co) for their open-source research.

## Contact Us

Join our [Discord](https://discord.gg/AKNgpMK4Yj) or [WeChat groups](https://gw.alicdn.com/imgextra/i2/O1CN01tqjWFi1ByuyehkTSB_!!6000000000015-0-tps-611-1279.jpg) for support and discussions!