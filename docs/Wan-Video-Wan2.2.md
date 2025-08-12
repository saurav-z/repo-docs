# Wan2.2: Unleash Cinematic Video Generation with Advanced AI

**Wan2.2** is a cutting-edge open-source video generation model that empowers users to create stunning, cinematic-quality videos with unprecedented ease and control.  [Explore Wan2.2 on GitHub](https://github.com/Wan-Video/Wan2.2)!

<p align="center">
    <img src="assets/logo.png" width="400"/>
<p>

<p align="center">
    💜 <a href="https://wan.video"><b>Wan</b></a> &nbsp&nbsp ｜ &nbsp&nbsp 🖥️ <a href="https://github.com/Wan-Video/Wan2.2">GitHub</a> &nbsp&nbsp  | &nbsp&nbsp🤗 <a href="https://huggingface.co/Wan-AI/">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/organization/Wan-AI">ModelScope</a>&nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://arxiv.org/abs/2503.20314">Paper</a> &nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://wan.video/welcome?spm=a2ty_o02.30011076.0.0.6c9ee41eCcluqg">Blog</a> &nbsp&nbsp |  &nbsp&nbsp 💬  <a href="https://discord.gg/AKNgpMK4Yj">Discord</a>&nbsp&nbsp
    <br>
    📕 <a href="https://alidocs.dingtalk.com/i/nodes/jb9Y4gmKWrx9eo4dCql9LlbYJGXn6lpz">使用指南(中文)</a>&nbsp&nbsp | &nbsp&nbsp 📘 <a href="https://alidocs.dingtalk.com/i/nodes/EpGBa2Lm8aZxe5myC99MelA2WgN7R35y">User Guide(English)</a>&nbsp&nbsp | &nbsp&nbsp💬 <a href="https://gw.alicdn.com/imgextra/i2/O1CN01tqjWFi1ByuyehkTSB_!!6000000000015-0-tps-611-1279.jpg">WeChat(微信)</a>&nbsp&nbsp
<br>

---

## Key Features

*   **Mixture-of-Experts (MoE) Architecture:**  Leverages an innovative MoE design to boost model capacity while maintaining efficient computational costs, allowing for richer details and enhanced video quality.
*   **Cinematic Aesthetics:**  Trained on a meticulously curated dataset with detailed aesthetic labels, empowering users with precise control over lighting, composition, color grading, and overall cinematic style.
*   **Advanced Motion Generation:** Trained on a significantly expanded dataset to produce videos with realistic and intricate motions, surpassing the capabilities of previous iterations.
*   **Efficient High-Definition Hybrid TI2V:**  Offers a fast, efficient 5B model for text-to-video and image-to-video generation at 720P resolution and 24fps, making high-quality video creation accessible on consumer-grade hardware.

## Recent Updates & Integrations

*   **HF Space:** Launch of a [Hugging Face Space](https://huggingface.co/spaces/Wan-AI/Wan-2.2-5B) utilizing the TI2V-5B model.
*   **ComfyUI Integration:**  Now seamlessly integrated with ComfyUI ([CN](https://docs.comfy.org/zh-CN/tutorials/video/wan/wan2_2) | [EN](https://docs.comfy.org/tutorials/video/wan/wan2_2)).
*   **Diffusers Integration:**  T2V, I2V, and TI2V models integrated into Diffusers ([T2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) | [I2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers) | [TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers)).
*   **Model Release:** Inference code and model weights for Wan2.2 are now available.

## Video Demos

<div align="center">
  <video src="https://github.com/user-attachments/assets/b63bfa58-d5d7-4de6-a1a2-98970b06d9a7" width="70%" poster=""> </video>
</div>

## Community Contributions

*   **DiffSynth-Studio:** Provides comprehensive support for Wan 2.2, including optimizations like low-GPU-memory offload and FP8 quantization.
*   **Kijai's ComfyUI WanVideoWrapper:** An alternative implementation for ComfyUI, offering cutting-edge optimizations.

## Getting Started

### Installation

```bash
git clone https://github.com/Wan-Video/Wan2.2.git
cd Wan2.2
pip install -r requirements.txt
```

**Important:** Ensure `torch >= 2.4.0`. If `flash_attn` installation fails, try installing other packages first.

### Model Downloads

Download the necessary models from Hugging Face or ModelScope:

| Model | Download Link                                                                   | Description                                    |
| ----- | ------------------------------------------------------------------------------- | ---------------------------------------------- |
| T2V-A14B | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B)    🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-T2V-A14B) | Text-to-Video MoE, 480P & 720P                 |
| I2V-A14B | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B)    🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-I2V-A14B) | Image-to-Video MoE, 480P & 720P                |
| TI2V-5B | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B)     🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-TI2V-5B)     | High-Compression VAE, T2V+I2V, 720P, 24 FPS |

> **Note:** TI2V-5B is optimized for 720P @ 24 FPS video generation.

#### Hugging Face CLI Download Example

```bash
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.2-T2V-A14B --local-dir ./Wan2.2-T2V-A14B
```

#### ModelScope CLI Download Example

```bash
pip install modelscope
modelscope download Wan-AI/Wan2.2-T2V-A14B --local_dir ./Wan2.2-T2V-A14B
```

### Running Text-to-Video Generation

#### (1) Basic Generation

```bash
python generate.py --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --offload_model True --convert_model_dtype --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
```

*   Requires at least 80GB VRAM.  Use `--offload_model True`, `--convert_model_dtype`, and `--t5_cpu` to reduce memory usage.
*   Multi-GPU example using FSDP + DeepSpeed Ulysses

```bash
torchrun --nproc_per_node=8 generate.py --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
```

#### (2) Using Prompt Extension (Recommended)

*   **Dashscope API:** Use Qwen-plus for T2V or Qwen-vl-max for I2V.  Requires a Dashscope API key.
    ```bash
    DASH_API_KEY=your_key torchrun --nproc_per_node=8 generate.py  --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Your prompt" --use_prompt_extend --prompt_extend_method 'dashscope' --prompt_extend_target_lang 'zh'
    ```
*   **Local Qwen Model:** Use models like `Qwen/Qwen2.5-14B-Instruct` (T2V) or  `Qwen/Qwen2.5-VL-7B-Instruct` (I2V) .

    ```bash
    torchrun --nproc_per_node=8 generate.py  --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Your prompt" --use_prompt_extend --prompt_extend_method 'local_qwen' --prompt_extend_target_lang 'zh'
    ```

### Running Image-to-Video Generation

```bash
python generate.py --task i2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-I2V-A14B --offload_model True --convert_model_dtype --image examples/i2v_input.JPG --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard..."
```

*   `size` parameter respects the aspect ratio of the input image.
*   Multi-GPU example (FSDP + DeepSpeed Ulysses):

```bash
torchrun --nproc_per_node=8 generate.py --task i2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-I2V-A14B --image examples/i2v_input.JPG --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Your prompt"
```

*   Image-to-Video generation without prompt
```bash
DASH_API_KEY=your_key torchrun --nproc_per_node=8 generate.py --task i2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-I2V-A14B --prompt '' --image examples/i2v_input.JPG --dit_fsdp --t5_fsdp --ulysses_size 8 --use_prompt_extend --prompt_extend_method 'dashscope'
```
### Running Text-Image-to-Video Generation

```bash
python generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ./Wan2.2-TI2V-5B --offload_model True --convert_model_dtype --t5_cpu --prompt "Your prompt"
```

*   720P resolution for TI2V is `1280*704` or `704*1280`.
*   Requires at least 24GB VRAM (e.g., RTX 4090).

### Computational Efficiency

[Link to image of computation efficiency chart](assets/comp_effic.png) - _See the README on GitHub for the detailed chart._

### Introduction of Wan2.2

Wan2.2 builds upon the foundational models of Wan2.1 with significant advancements, driven by innovations in:

*   **Mixture-of-Experts (MoE) Architecture:**  A  two-expert design to enhance video quality while maintaining efficient computation.
*   **High-Definition Hybrid TI2V:**  TI2V-5B supports text-to-video and image-to-video tasks efficiently.

[Link to MoE Arch Diagram](assets/moe_arch.png)
[Link to VAE Diagram](assets/vae.png)
[Link to Performance Chart](assets/performance.png)

### Citation

```
@article{wan2025,
      title={Wan: Open and Advanced Large-Scale Video Generative Models},
      author={...},
      journal = {arXiv preprint arXiv:2503.20314},
      year={2025}
}
```

### License

Licensed under the Apache 2.0 License.  See `LICENSE.txt` for details.

### Acknowledgements

Thanks to the contributors of [SD3](https://huggingface.co/stabilityai/stable-diffusion-3-medium), [Qwen](https://huggingface.co/Qwen), [umt5-xxl](https://huggingface.co/google/umt5-xxl), [diffusers](https://github.com/huggingface/diffusers) and [HuggingFace](https://huggingface.co) repositories.

### Contact

Join our [Discord](https://discord.gg/AKNgpMK4Yj) or [WeChat](https://gw.alicdn.com/imgextra/i2/O1CN01tqjWFi1ByuyehkTSB_!!6000000000015-0-tps-611-1279.jpg) groups.
```
Key improvements and summarization:

*   **SEO-Optimized Heading:**  Uses "Wan2.2: Unleash Cinematic Video Generation with Advanced AI" to target relevant keywords.
*   **One-Sentence Hook:** Immediately captures attention and highlights the key benefit.
*   **Clear Structure:** Uses headings and subheadings for readability.
*   **Bulleted Key Features:**  Highlights key selling points and innovations.
*   **Concise Language:**  Removes redundancy and improves clarity.
*   **Actionable Instructions:**  Provides clear installation and usage examples with command snippets.
*   **Links:**  Keeps all original links, and adds a link back to the GitHub repo at the start.
*   **Emphasis on Benefits:** Focuses on what users *get* (cinematic videos, easy creation, etc.)
*   **Call to Action:**  Encourages the user to explore the project.
*   **Model Download:** Made easier to understand using a table
*   **Text-Image-to-Video:** Improved the instructions for Text-Image-to-Video generation, using a Single-GPU example.
*   **Image-to-Video:** Improved the instructions for Image-to-Video generation.

This revised README is more engaging, informative, and optimized for both user experience and search engines.