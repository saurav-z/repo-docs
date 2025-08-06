<!-- Improved & SEO-Optimized README for Wan2.2 -->

# Wan2.2: Unleash Cinematic-Quality Video Generation

**Wan2.2 is a cutting-edge video generation model offering unparalleled cinematic quality and efficiency, revolutionizing the way we create moving visuals.** ([Original Repo](https://github.com/Wan-Video/Wan2.2))

<p align="center">
    <img src="assets/logo.png" width="400"/>
</p>

<p align="center">
    💜 <a href="https://wan.video"><b>Wan</b></a> &nbsp&nbsp ｜ &nbsp&nbsp 🖥️ <a href="https://github.com/Wan-Video/Wan2.2">GitHub</a> &nbsp&nbsp  | &nbsp&nbsp🤗 <a href="https://huggingface.co/Wan-AI/">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/organization/Wan-AI">ModelScope</a>&nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://arxiv.org/abs/2503.20314">Paper</a> &nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://wan.video/welcome?spm=a2ty_o02.30011076.0.0.6c9ee41eCcluqg">Blog</a> &nbsp&nbsp |  &nbsp&nbsp 💬  <a href="https://discord.gg/AKNgpMK4Yj">Discord</a>&nbsp&nbsp
    <br>
    📕 <a href="https://alidocs.dingtalk.com/i/nodes/jb9Y4gmKWrx9eo4dCql9LlbYJGXn6lpz">使用指南(中文)</a>&nbsp&nbsp | &nbsp&nbsp 📘 <a href="https://alidocs.dingtalk.com/i/nodes/EpGBa2Lm8aZxe5myC99MelA2WgN7R35y">User Guide(English)</a>&nbsp&nbsp | &nbsp&nbsp💬 <a href="https://gw.alicdn.com/imgextra/i2/O1CN01tqjWFi1ByuyehkTSB_!!6000000000015-0-tps-611-1279.jpg">WeChat(微信)</a>&nbsp&nbsp
<br>

---

## Key Features:

*   👍 **Mixture-of-Experts (MoE) Architecture:**  Leverages a MoE architecture to significantly enhance model capacity while maintaining efficient computational performance.
*   👍 **Cinematic Aesthetics:** Incorporates meticulously curated aesthetic data with detailed labels, providing precise control over lighting, composition, and color for stunning visual styles.
*   👍 **Advanced Motion Generation:** Trained on a massive dataset (+65.6% more images and +83.2% more videos than Wan2.1) for improved generalization across motion, semantics, and aesthetics.
*   👍 **Efficient High-Definition TI2V:** Includes a 5B model with a 16×16×4 compression ratio, supporting text-to-video and image-to-video generation at 720P/24fps on consumer-grade GPUs.

## Video Demos

<div align="center">
  <video src="https://github.com/user-attachments/assets/b63bfa58-d5d7-4de6-a1a2-98970b06d9a7" width="70%" poster=""> </video>
</div>

## Recent Updates:

*   **[HF Space]:**  HF space with the TI2V-5B model is available.
*   **ComfyUI Integration:**  Wan2.2 is now integrated into ComfyUI ([CN](https://docs.comfy.org/zh-CN/tutorials/video/wan/wan2_2) | [EN](https://docs.comfy.org/tutorials/video/wan/wan2_2)).
*   **Diffusers Integration:**  T2V, I2V, and TI2V models integrated into Diffusers ([T2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) | [I2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers) | [TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers)).
*   **Inference Code and Weights Released:**  Access the inference code and model weights for Wan2.2.

## Community Contributions

*   [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) provides comprehensive support for Wan 2.2.
*   [Kijai's ComfyUI WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper) is an alternative implementation of Wan models for ComfyUI.

##  Todo List

*   Wan2.2 Text-to-Video:
    *   [x] Multi-GPU Inference code of the A14B and 14B models
    *   [x] Checkpoints of the A14B and 14B models
    *   [x] ComfyUI integration
    *   [x] Diffusers integration
*   Wan2.2 Image-to-Video:
    *   [x] Multi-GPU Inference code of the A14B model
    *   [x] Checkpoints of the A14B model
    *   [x] ComfyUI integration
    *   [x] Diffusers integration
*   Wan2.2 Text-Image-to-Video:
    *   [x] Multi-GPU Inference code of the 5B model
    *   [x] Checkpoints of the 5B model
    *   [x] ComfyUI integration
    *   [x] Diffusers integration

## Getting Started:

### Installation
```bash
git clone https://github.com/Wan-Video/Wan2.2.git
cd Wan2.2
```

### Install Dependencies
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

> 💡Note: The TI2V-5B model supports 720P video generation at **24 FPS**.

#### Download Models:

Using huggingface-cli:
```bash
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.2-T2V-A14B --local-dir ./Wan2.2-T2V-A14B
```

Using modelscope-cli:
```bash
pip install modelscope
modelscope download Wan-AI/Wan2.2-T2V-A14B --local_dir ./Wan2.2-T2V-A14B
```

## Run Examples

### Text-to-Video Generation

**(1) Without Prompt Extension**

*   Single-GPU inference:
```bash
python generate.py  --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --offload_model True --convert_model_dtype --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
```

> 💡 Runs on a GPU with at least 80GB VRAM.  Use `--offload_model True`, `--convert_model_dtype`, and `--t5_cpu` to reduce memory usage.

*   Multi-GPU inference using FSDP + DeepSpeed Ulysses:
```bash
torchrun --nproc_per_node=8 generate.py --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
```

**(2) Using Prompt Extension**

*   **Using Dashscope API:**
    *   Get `dashscope.api_key`.
    *   Set `DASH_API_KEY` and (for Alibaba Cloud intl) `DASH_API_URL`.
    *   Use `qwen-plus` (T2V) or `qwen-vl-max` (I2V).
    *   Adjust `--prompt_extend_model`.
```bash
DASH_API_KEY=your_key torchrun --nproc_per_node=8 generate.py  --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage" --use_prompt_extend --prompt_extend_method 'dashscope' --prompt_extend_target_lang 'zh'
```

*   **Using Local Model for Extension:**
    *   Use models like `Qwen/Qwen2.5-14B-Instruct` (T2V) or `Qwen/Qwen2.5-VL-7B-Instruct` (I2V).
    *   Adjust `--prompt_extend_model` to a local path or Hugging Face model.
```bash
torchrun --nproc_per_node=8 generate.py  --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage" --use_prompt_extend --prompt_extend_method 'local_qwen' --prompt_extend_target_lang 'zh'
```

### Image-to-Video Generation

*   Single-GPU inference:
```bash
python generate.py --task i2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-I2V-A14B --offload_model True --convert_model_dtype --image examples/i2v_input.JPG --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
```

> 💡 Use `--size` to specify the video area, matching the input image's aspect ratio.

*   Multi-GPU inference using FSDP + DeepSpeed Ulysses:
```bash
torchrun --nproc_per_node=8 generate.py --task i2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-I2V-A14B --image examples/i2v_input.JPG --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
```

*   Image-to-Video Generation without prompt:
```bash
DASH_API_KEY=your_key torchrun --nproc_per_node=8 generate.py --task i2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-I2V-A14B --prompt '' --image examples/i2v_input.JPG --dit_fsdp --t5_fsdp --ulysses_size 8 --use_prompt_extend --prompt_extend_method 'dashscope'
```
> 💡The model can generate videos solely from the input image. You can use prompt extension to generate prompt from the image.

### Text-Image-to-Video Generation

*   Single-GPU Text-to-Video inference:
```bash
python generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ./Wan2.2-TI2V-5B --offload_model True --convert_model_dtype --t5_cpu --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage"
```

> 💡 Use `--size` as 1280*704 or 704*1280 for 720P. Use `--offload_model True`, `--convert_model_dtype` and `--t5_cpu` options to reduce GPU memory usage.

*   Single-GPU Image-to-Video inference:
```bash
python generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ./Wan2.2-TI2V-5B --offload_model True --convert_model_dtype --t5_cpu --image examples/i2v_input.JPG --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
```
> 💡 If the image parameter is configured, it is an Image-to-Video generation; otherwise, it defaults to a Text-to-Video generation.

*   Multi-GPU inference using FSDP + DeepSpeed Ulysses:
```bash
torchrun --nproc_per_node=8 generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ./Wan2.2-TI2V-5B --dit_fsdp --t5_fsdp --ulysses_size 8 --image examples/i2v_input.JPG --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
```

## Computational Efficiency

<div align="center">
    <img src="assets/comp_effic.png" alt="Computational Efficiency" style="width: 80%;" />
</div>

## Introduction of Wan2.2

**Wan2.2** represents a significant leap forward in video generation, built upon the innovations of Wan2.1.  Key advancements include:

**(1) Mixture-of-Experts (MoE) Architecture:**

*   MoE architecture is implemented in the A14B model.
*   Each expert model has 14B parameters, resulting in 27B parameters in total.
*   The transition point between the two experts is determined by the signal-to-noise ratio (SNR).
   *   The high-noise expert activated in the beginning of the denoising process.
   *   The low-noise expert activated when $t<{t}_{moe}$.

**(2) Efficient High-Definition Hybrid TI2V:**

*   The 5B model (TI2V-5B) uses a high-compression Wan2.2-VAE.
*   The compression ratio is $4\times16\times16$, reaching a 64 compression ratio.
*   Without specific optimization, TI2V-5B can generate a 5-second 720P video in under 9 minutes on a single consumer-grade GPU.

**(3) Comparisons to SOTAs**

*   Wan2.2 achieves superior performance compared to these leading models.

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

The models in this repository are licensed under the [Apache 2.0 License](LICENSE.txt).

## Acknowledgements

Thanks to the contributors of [SD3](https://huggingface.co/stabilityai/stable-diffusion-3-medium), [Qwen](https://huggingface.co/Qwen), [umt5-xxl](https://huggingface.co/google/umt5-xxl), [diffusers](https://github.com/huggingface/diffusers) and [HuggingFace](https://huggingface.co).

## Contact

Join our [Discord](https://discord.gg/AKNgpMK4Yj) or [WeChat groups](https://gw.alicdn.com/imgextra/i2/O1CN01tqjWFi1ByuyehkTSB_!!6000000000015-0-tps-611-1279.jpg)!