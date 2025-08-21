# Wan2.2: Revolutionizing Video Generation with Open and Advanced AI

**Unleash cinematic-quality video creation with Wan2.2, the next generation of open-source video generative models. [Explore the original repository](https://github.com/Wan-Video/Wan2.2)**

<p align="center">
    <img src="assets/logo.png" width="400"/>
<p>

<p align="center">
    💜 <a href="https://wan.video"><b>Wan</b></a> &nbsp&nbsp ｜ &nbsp&nbsp 🖥️ <a href="https://github.com/Wan-Video/Wan2.2">GitHub</a> &nbsp&nbsp  | &nbsp&nbsp🤗 <a href="https://huggingface.co/Wan-AI/">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/organization/Wan-AI">ModelScope</a>&nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://arxiv.org/abs/2503.20314">Paper</a> &nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://wan.video/welcome?spm=a2ty_o02.30011076.0.0.6c9ee41eCcluqg">Blog</a> &nbsp&nbsp |  &nbsp&nbsp 💬  <a href="https://discord.gg/AKNgpMK4Yj">Discord</a>&nbsp&nbsp
    <br>
    📕 <a href="https://alidocs.dingtalk.com/i/nodes/jb9Y4gmKWrx9eo4dCql9LlbYJGXn6lpz">使用指南(中文)</a>&nbsp&nbsp | &nbsp&nbsp 📘 <a href="https://alidocs.dingtalk.com/i/nodes/EpGBa2Lm8aZxe5myC99MelA2WgN7R35y">User Guide(English)</a>&nbsp&nbsp | &nbsp&nbsp💬 <a href="https://gw.alicdn.com/imgextra/i2/O1CN01tqjWFi1ByuyehkTSB_!!6000000000015-0-tps-611-1279.jpg">WeChat(微信)</a>&nbsp&nbsp
<br>

---

## Key Features of Wan2.2:

*   ✅ **Mixture-of-Experts (MoE) Architecture:** Enhances model capacity and generation quality while maintaining efficient inference.
*   ✅ **Cinematic-Level Aesthetics:** Leverage carefully curated aesthetic data for precise control over lighting, composition, and style.
*   ✅ **Enhanced Complex Motion Generation:** Trained on significantly larger datasets for improved realism and broader applicability.
*   ✅ **Efficient High-Definition TI2V:**  Generate 720P videos at 24fps on consumer-grade GPUs with our optimized TI2V-5B model.

## Video Demos

<div align="center">
  <video src="https://github.com/user-attachments/assets/b63bfa58-d5d7-4de6-a1a2-98970b06d9a7" width="70%" poster=""> </video>
</div>

## 🔥 Latest News

*   **July 28, 2025:** HF Space launched for TI2V-5B model: [HF space](https://huggingface.co/spaces/Wan-AI/Wan-2.2-5B)
*   **July 28, 2025:** Wan2.2 integrated into ComfyUI: [CN](https://docs.comfy.org/zh-CN/tutorials/video/wan/wan2_2) | [EN](https://docs.comfy.org/tutorials/video/wan/wan2_2)
*   **July 28, 2025:** Wan2.2 T2V, I2V and TI2V integrated into Diffusers: ([T2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) | [I2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers) | [TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers))
*   **July 28, 2025:** Inference code and model weights for Wan2.2 released.

## Community Works

Explore community contributions and integrations with Wan2.2:

*   [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio): Comprehensive support, including optimizations for low-GPU-memory environments, FP8 quantization, and more.
*   [Kijai's ComfyUI WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper): Alternative implementation focusing on Wan models, enabling cutting-edge optimizations.

## 📑 Todo List

*   **Wan2.2 Text-to-Video**
    *   [x] Multi-GPU Inference code of the A14B and 14B models
    *   [x] Checkpoints of the A14B and 14B models
    *   [x] ComfyUI integration
    *   [x] Diffusers integration
*   **Wan2.2 Image-to-Video**
    *   [x] Multi-GPU Inference code of the A14B model
    *   [x] Checkpoints of the A14B model
    *   [x] ComfyUI integration
    *   [x] Diffusers integration
*   **Wan2.2 Text-Image-to-Video**
    *   [x] Multi-GPU Inference code of the 5B model
    *   [x] Checkpoints of the 5B model
    *   [x] ComfyUI integration
    *   [x] Diffusers integration

## Run Wan2.2

### Installation

Clone the repository:

```bash
git clone https://github.com/Wan-Video/Wan2.2.git
cd Wan2.2
```

Install dependencies:

```bash
# Ensure torch >= 2.4.0
# If the installation of `flash_attn` fails, try installing the other packages first and install `flash_attn` last
pip install -r requirements.txt
```

### Model Download

| Model                | Download Links                                                                                                                              | Description                                   |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------- |
| T2V-A14B             | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B)    🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-T2V-A14B)    | Text-to-Video MoE model, supports 480P & 720P |
| I2V-A14B             | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B)    🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-I2V-A14B)    | Image-to-Video MoE model, supports 480P & 720P |
| TI2V-5B              | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B)     🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-TI2V-5B)     | High-compression VAE, T2V+I2V, supports 720P |

> 💡**Note:** The TI2V-5B model supports 720P video generation at **24 FPS**.

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

### Run Text-to-Video Generation

This repository supports the `Wan2.2-T2V-A14B` Text-to-Video model and can simultaneously support video generation at 480P and 720P resolutions.

#### (1) Without Prompt Extension

-   Single-GPU inference

```bash
python generate.py  --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --offload_model True --convert_model_dtype --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
```

> 💡 This command can run on a GPU with at least 80GB VRAM.

> 💡If you encounter OOM (Out-of-Memory) issues, you can use the `--offload_model True`, `--convert_model_dtype` and `--t5_cpu` options to reduce GPU memory usage.

-   Multi-GPU inference using FSDP + DeepSpeed Ulysses

    We use [PyTorch FSDP](https://docs.pytorch.org/docs/stable/fsdp.html) and [DeepSpeed Ulysses](https://arxiv.org/abs/2309.14509) to accelerate inference.

```bash
torchrun --nproc_per_node=8 generate.py --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
```

#### (2) Using Prompt Extension

Extending the prompts can effectively enrich the details in the generated videos, further enhancing the video quality. Therefore, we recommend enabling prompt extension. We provide the following two methods for prompt extension:

-   Use the Dashscope API for extension.

    *   Apply for a `dashscope.api_key` in advance ([EN](https://www.alibabacloud.com/help/en/model-studio/getting-started/first-api-call-to-qwen) | [CN](https://help.aliyun.com/zh/model-studio/getting-started/first-api-call-to-qwen)).
    *   Configure the environment variable `DASH_API_KEY` to specify the Dashscope API key. For users of Alibaba Cloud's international site, you also need to set the environment variable `DASH_API_URL` to 'https://dashscope-intl.aliyuncs.com/api/v1'. For more detailed instructions, please refer to the [dashscope document](https://www.alibabacloud.com/help/en/model-studio/developer-reference/use-qwen-by-calling-api?spm=a2c63.p38356.0.i1).
    *   Use the `qwen-plus` model for text-to-video tasks and `qwen-vl-max` for image-to-video tasks.
    *   You can modify the model used for extension with the parameter `--prompt_extend_model`. For example:

```bash
DASH_API_KEY=your_key torchrun --nproc_per_node=8 generate.py  --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage" --use_prompt_extend --prompt_extend_method 'dashscope' --prompt_extend_target_lang 'zh'
```

-   Using a local model for extension.

    *   By default, the Qwen model on HuggingFace is used for this extension. Users can choose Qwen models or other models based on the available GPU memory size.
    *   For text-to-video tasks, you can use models like `Qwen/Qwen2.5-14B-Instruct`, `Qwen/Qwen2.5-7B-Instruct` and `Qwen/Qwen2.5-3B-Instruct`.
    *   For image-to-video tasks, you can use models like `Qwen/Qwen2.5-VL-7B-Instruct` and `Qwen/Qwen2.5-VL-3B-Instruct`.
    *   Larger models generally provide better extension results but require more GPU memory.
    *   You can modify the model used for extension with the parameter `--prompt_extend_model` , allowing you to specify either a local model path or a Hugging Face model. For example:

```bash
torchrun --nproc_per_node=8 generate.py  --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage" --use_prompt_extend --prompt_extend_method 'local_qwen' --prompt_extend_target_lang 'zh'
```

### Run Image-to-Video Generation

This repository supports the `Wan2.2-I2V-A14B` Image-to-Video model and can simultaneously support video generation at 480P and 720P resolutions.

-   Single-GPU inference

```bash
python generate.py --task i2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-I2V-A14B --offload_model True --convert_model_dtype --image examples/i2v_input.JPG --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
```

> This command can run on a GPU with at least 80GB VRAM.

> 💡For the Image-to-Video task, the `size` parameter represents the area of the generated video, with the aspect ratio following that of the original input image.

-   Multi-GPU inference using FSDP + DeepSpeed Ulysses

```bash
torchrun --nproc_per_node=8 generate.py --task i2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-I2V-A14B --image examples/i2v_input.JPG --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
```

-   Image-to-Video Generation without prompt

```bash
DASH_API_KEY=your_key torchrun --nproc_per_node=8 generate.py --task i2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-I2V-A14B --prompt '' --image examples/i2v_input.JPG --dit_fsdp --t5_fsdp --ulysses_size 8 --use_prompt_extend --prompt_extend_method 'dashscope'
```

> 💡The model can generate videos solely from the input image. You can use prompt extension to generate prompt from the image.

> The process of prompt extension can be referenced [here](#2-using-prompt-extention).

### Run Text-Image-to-Video Generation

This repository supports the `Wan2.2-TI2V-5B` Text-Image-to-Video model and can support video generation at 720P resolutions.

-   Single-GPU Text-to-Video inference

```bash
python generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ./Wan2.2-TI2V-5B --offload_model True --convert_model_dtype --t5_cpu --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage"
```

> 💡Unlike other tasks, the 720P resolution of the Text-Image-to-Video task is `1280*704` or `704*1280`.

> This command can run on a GPU with at least 24GB VRAM (e.g, RTX 4090 GPU).

> 💡If you are running on a GPU with at least 80GB VRAM, you can remove the `--offload_model True`, `--convert_model_dtype` and `--t5_cpu` options to speed up execution.

-   Single-GPU Image-to-Video inference

```bash
python generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ./Wan2.2-TI2V-5B --offload_model True --convert_model_dtype --t5_cpu --image examples/i2v_input.JPG --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
```

> 💡If the image parameter is configured, it is an Image-to-Video generation; otherwise, it defaults to a Text-to-Video generation.

> 💡Similar to Image-to-Video, the `size` parameter represents the area of the generated video, with the aspect ratio following that of the original input image.

-   Multi-GPU inference using FSDP + DeepSpeed Ulysses

```bash
torchrun --nproc_per_node=8 generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ./Wan2.2-TI2V-5B --dit_fsdp --t5_fsdp --ulysses_size 8 --image examples/i2v_input.JPG --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
```

> The process of prompt extension can be referenced [here](#2-using-prompt-extention).

## Computational Efficiency on Different GPUs

Computational efficiency tests are shown in the table below. The results are presented in the format: **Total time (s) / peak GPU memory (GB)**.

<div align="center">
    <img src="assets/comp_effic.png" alt="" style="width: 80%;" />
</div>

> The parameter settings for the tests presented in this table are as follows:
> (1) Multi-GPU: 14B: `--ulysses_size 4/8 --dit_fsdp --t5_fsdp`, 5B: `--ulysses_size 4/8 --offload_model True --convert_model_dtype --t5_cpu`; Single-GPU: 14B: `--offload_model True --convert_model_dtype`, 5B: `--offload_model True --convert_model_dtype --t5_cpu`
(--convert_model_dtype converts model parameter types to config.param_dtype);
> (2) The distributed testing utilizes the built-in FSDP and Ulysses implementations, with FlashAttention3 deployed on Hopper architecture GPUs;
> (3) Tests were run without the `--use_prompt_extend` flag;
> (4) Reported results are the average of multiple samples taken after the warm-up phase.

---

## Technical Deep Dive

Wan2.2 builds upon Wan2.1 with several innovative improvements.

### (1) Mixture-of-Experts (MoE) Architecture

Wan2.2 leverages a Mixture-of-Experts (MoE) architecture to increase model capacity without significantly increasing inference costs. The A14B models employ a two-expert design for denoising: a high-noise expert for initial layout and a low-noise expert for refining details.

<div align="center">
    <img src="assets/moe_arch.png" alt="" style="width: 90%;" />
</div>

The transition between experts is determined by the signal-to-noise ratio (SNR).

<div align="center">
    <img src="assets/moe_2.png" alt="" style="width: 90%;" />
</div>

### (2) Efficient High-Definition Hybrid TI2V

For efficient deployment, Wan2.2 also includes a 5B dense model (TI2V-5B) with a high-compression Wan2.2-VAE, achieving a compression ratio of $4\times16\times16$. This model supports text-to-video and image-to-video tasks and can generate 720P videos at 24fps on consumer-grade GPUs.

<div align="center">
    <img src="assets/vae.png" alt="" style="width: 80%;" />
</div>

### Comparisons to SOTAs

Wan2.2 was compared to leading commercial models on Wan-Bench 2.0, and achieved superior performance across multiple dimensions.

<div align="center">
    <img src="assets/performance.png" alt="" style="width: 90%;" />
</div>

## Citation

If you use this work, please cite us:

```
@article{wan2025,
      title={Wan: Open and Advanced Large-Scale Video Generative Models}, 
      author={Team Wan and Ang Wang and Baole Ai and Bin Wen and Chaojie Mao and Chen-Wei Xie and Di Chen and Feiwu Yu and Haiming Zhao and Jianxiao Yang and Jianyuan Zeng and Jiayu Wang and Jingfeng Zhang and Jingren Zhou and Jinkai Wang and Jixuan Chen and Kai Zhu and Kang Zhao and Keyu Yan and Lianghua Huang and Mengyang Feng and Ningyi Zhang and Pandeng Li and Pingyu Wu and Ruihang Chu and Ruili Feng and Shiwei Zhang and Siyang Sun and Tao Fang and Tianxing Wang and Tianyi Gui and Tingyu Weng and Tong Shen and Wei Lin and Wei Wang and Wei Wang and Wenmeng Zhou and Wente Wang and Wenting Shen and Wenyuan Yu and Xianzhong Shi and Xiaoming Huang and Xin Xu and Yan Kou and Yangyu Lv and Yifei Li and Yijing Liu and Yiming Wang and Yingya Zhang and Yitong Huang and Yong Li and You Wu and Yu Liu and Yulin Pan and Yun Zheng and Yuntao Hong and Yupeng Shi and Yutong Feng and Zeyinzi Jiang and Zhen Han and Zhi-Fan Wu and Ziyu Liu},
      journal = {arXiv preprint arXiv:2503.20314},
      year={2025}
}
```

## License

This project is licensed under the Apache 2.0 License. See [LICENSE.txt](LICENSE.txt) for details.

## Acknowledgements

We thank the contributors to [SD3](https://huggingface.co/stabilityai/stable-diffusion-3-medium), [Qwen](https://huggingface.co/Qwen), [umt5-xxl](https://huggingface.co/google/umt5-xxl), [diffusers](https://github.com/huggingface/diffusers) and [HuggingFace](https://huggingface.co) repositories.

## Contact Us

Join our [Discord](https://discord.gg/AKNgpMK4Yj) or [WeChat groups](https://gw.alicdn.com/imgextra/i2/O1CN01tqjWFi1ByuyehkTSB_!!6000000000015-0-tps-611-1279.jpg) to connect with our research and product teams.