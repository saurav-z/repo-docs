# Wan2.2: Unleash Cinematic Video Generation with Open-Source Power

**[Explore the Wan2.2 Repository](https://github.com/Wan-Video/Wan2.2) to create stunning videos with cutting-edge AI.**

<p align="center">
    <img src="assets/logo.png" width="400"/>
</p>

Wan2.2 represents a significant leap forward in open-source video generation, offering unparalleled quality and control.  This advanced model empowers users to generate cinematic-quality videos with ease, leveraging innovative features and extensive training.

**Key Features of Wan2.2:**

*   ✅ **Mixture-of-Experts (MoE) Architecture:** Optimizes the denoising process with specialized expert models, enhancing overall model capacity without increasing computational costs.
*   ✅ **Cinematic-Level Aesthetics:**  Incorporate meticulously curated aesthetic data and control lighting, composition, contrast, and color tone for precise and customizable video styles.
*   ✅ **Enhanced Motion Generation:** Trained on a significantly larger dataset (+65.6% images, +83.2% videos) for improved generalization across motions, semantics, and aesthetics, leading to top performance.
*   ✅ **Efficient High-Definition Hybrid TI2V:**  Open-sources a 5B model with a 16x16x4 compression ratio. This model generates 720P videos at 24fps, allowing it to be run on consumer-grade GPUs.

**Key Highlights & Updates:**

*   **[HF space](https://huggingface.co/spaces/Wan-AI/Wan-2.2-5B)** now available for easy access.
*   Seamless integration into **ComfyUI** ([CN](https://docs.comfy.org/zh-CN/tutorials/video/wan/wan2_2) | [EN](https://docs.comfy.org/tutorials/video/wan/wan2_2)).
*   **Diffusers** integration for T2V, I2V, and TI2V ([T2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) | [I2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers) | [TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers)).
*   Inference code and model weights for Wan2.2 are now released.

## Video Demos

<div align="center">
  <video src="https://github.com/user-attachments/assets/b63bfa58-d5d7-4de6-a1a2-98970b06d9a7" width="70%" poster=""> </video>
</div>

## Community Works
Supporting projects such as:

*   [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)
*   [Kijai's ComfyUI WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper)

## Getting Started with Wan2.2

### Installation

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/Wan-Video/Wan2.2.git
    cd Wan2.2
    ```

2.  **Install Dependencies:**

    ```bash
    # Ensure torch >= 2.4.0
    # If the installation of `flash_attn` fails, try installing the other packages first and install `flash_attn` last
    pip install -r requirements.txt
    ```

### Model Download

Download your preferred models from Hugging Face or ModelScope:

| Models              | Download Links                                                                                                                              | Description |
|--------------------|---------------------------------------------------------------------------------------------------------------------------------------------|-------------|
| T2V-A14B    | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B)    🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-T2V-A14B)    | Text-to-Video MoE model, supports 480P & 720P |
| I2V-A14B    | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B)    🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-I2V-A14B)    | Image-to-Video MoE model, supports 480P & 720P |
| TI2V-5B     | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B)     🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-TI2V-5B)     | High-compression VAE, T2V+I2V, supports 720P |

> 💡 Note:  The TI2V-5B model supports 720P video generation at 24 FPS.

**Example Download using `huggingface-cli`:**

```bash
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.2-T2V-A14B --local-dir ./Wan2.2-T2V-A14B
```

**Example Download using `modelscope-cli`:**

```bash
pip install modelscope
modelscope download Wan-AI/Wan2.2-T2V-A14B --local_dir ./Wan2.2-T2V-A14B
```

### Running Text-to-Video Generation

#### (1) Without Prompt Extension

*   **Single-GPU Inference:**

    ```bash
    python generate.py  --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --offload_model True --convert_model_dtype --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
    ```

    > 💡 This command can run on a GPU with at least 80GB VRAM.
    > 💡 Use `--offload_model True`, `--convert_model_dtype` and `--t5_cpu` to reduce GPU memory usage.

*   **Multi-GPU Inference (FSDP + DeepSpeed Ulysses):**

    ```bash
    torchrun --nproc_per_node=8 generate.py --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
    ```

#### (2) Using Prompt Extension

*   **Using Dashscope API:**

    *   Get a `dashscope.api_key`.
    *   Set the environment variable `DASH_API_KEY`.  Use `DASH_API_URL` for Alibaba Cloud international site.
    *   Use `qwen-plus` for T2V and `qwen-vl-max` for I2V.

    ```bash
    DASH_API_KEY=your_key torchrun --nproc_per_node=8 generate.py  --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Your prompt here" --use_prompt_extend --prompt_extend_method 'dashscope' --prompt_extend_target_lang 'zh'
    ```

*   **Using Local Model for Extension:**

    *   Use models like `Qwen/Qwen2.5-14B-Instruct`, `Qwen/Qwen2.5-7B-Instruct`, or `Qwen/Qwen2.5-3B-Instruct` (for T2V).
    *   Use models like `Qwen/Qwen2.5-VL-7B-Instruct` or `Qwen/Qwen2.5-VL-3B-Instruct` (for I2V).
    *   Modify `--prompt_extend_model` to specify a local path or Hugging Face model.

    ```bash
    torchrun --nproc_per_node=8 generate.py  --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Your prompt here" --use_prompt_extend --prompt_extend_method 'local_qwen' --prompt_extend_target_lang 'zh'
    ```

### Running Image-to-Video Generation

*   **Single-GPU Inference:**

    ```bash
    python generate.py --task i2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-I2V-A14B --offload_model True --convert_model_dtype --image examples/i2v_input.JPG --prompt "Your prompt here"
    ```
    > 💡 For Image-to-Video, `size` respects the input image's aspect ratio.

*   **Multi-GPU Inference (FSDP + DeepSpeed Ulysses):**

    ```bash
    torchrun --nproc_per_node=8 generate.py --task i2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-I2V-A14B --image examples/i2v_input.JPG --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Your prompt here"
    ```

*   **Image-to-Video Generation without Prompt (using Dashscope):**

    ```bash
    DASH_API_KEY=your_key torchrun --nproc_per_node=8 generate.py --task i2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-I2V-A14B --prompt '' --image examples/i2v_input.JPG --dit_fsdp --t5_fsdp --ulysses_size 8 --use_prompt_extend --prompt_extend_method 'dashscope'
    ```
    > 💡 The model can generate videos solely from the input image.

### Running Text-Image-to-Video Generation

*   **Single-GPU Text-to-Video:**

    ```bash
    python generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ./Wan2.2-TI2V-5B --offload_model True --convert_model_dtype --t5_cpu --prompt "Your prompt here"
    ```

    > 💡 For TI2V, the 720P resolution is 1280\*704 or 704\*1280.

*   **Single-GPU Image-to-Video:**

    ```bash
    python generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ./Wan2.2-TI2V-5B --offload_model True --convert_model_dtype --t5_cpu --image examples/i2v_input.JPG --prompt "Your prompt here"
    ```
    > 💡 If `--image` is set, it's Image-to-Video; otherwise, it's Text-to-Video.

*   **Multi-GPU Inference (FSDP + DeepSpeed Ulysses):**

    ```bash
    torchrun --nproc_per_node=8 generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ./Wan2.2-TI2V-5B --dit_fsdp --t5_fsdp --ulysses_size 8 --image examples/i2v_input.JPG --prompt "Your prompt here"
    ```

## Computational Efficiency

Wan2.2 offers excellent performance on various GPUs.  See the table below for benchmarks:

<div align="center">
    <img src="assets/comp_effic.png" alt="" style="width: 80%;" />
</div>

## About Wan2.2

Wan2.2 enhances video generation with:

### (1) Mixture-of-Experts (MoE) Architecture
*   Increases total model parameters while keeping inference cost nearly unchanged
*   High-noise expert for the early stages, focusing on overall layout; and a low-noise expert for the later stages, refining video details.

### (2) Efficient High-Definition Hybrid TI2V
*   TI2V-5B model that can generate 720P videos at 24fps.
*   High-compression Wan2.2-VAE achieving a 4x16x16 compression ratio.

### (3) Comparisons to SOTAs

Wan2.2 excels against leading commercial models, delivering superior performance.

<div align="center">
    <img src="assets/performance.png" alt="" style="width: 90%;" />
</div>

## Citation

If you use Wan2.2, please cite our work:

```
@article{wan2025,
      title={Wan: Open and Advanced Large-Scale Video Generative Models}, 
      author={Team Wan and Ang Wang and Baole Ai and Bin Wen and Chaojie Mao and Chen-Wei Xie and Di Chen and Feiwu Yu and Haiming Zhao and Jianxiao Yang and Jianyuan Zeng and Jiayu Wang and Jingfeng Zhang and Jingren Zhou and Jinkai Wang and Jixuan Chen and Kai Zhu and Kang Zhao and Keyu Yan and Lianghua Huang and Mengyang Feng and Ningyi Zhang and Pandeng Li and Pingyu Wu and Ruihang Chu and Ruili Feng and Shiwei Zhang and Siyang Sun and Tao Fang and Tianxing Wang and Tianyi Gui and Tingyu Weng and Tong Shen and Wei Lin and Wei Wang and Wei Wang and Wenmeng Zhou and Wente Wang and Wenting Shen and Wenyuan Yu and Xianzhong Shi and Xiaoming Huang and Xin Xu and Yan Kou and Yangyu Lv and Yifei Li and Yijing Liu and Yiming Wang and Yingya Zhang and Yitong Huang and Yong Li and You Wu and Yu Liu and Yulin Pan and Yun Zheng and Yuntao Hong and Yupeng Shi and Yutong Feng and Zeyinzi Jiang and Zhen Han and Zhi-Fan Wu and Ziyu Liu},
      journal = {arXiv preprint arXiv:2503.20314},
      year={2025}
}
```

## License

Wan2.2 is licensed under the Apache 2.0 License. For full details and restrictions, see [LICENSE.txt](LICENSE.txt).

## Acknowledgements

Thanks to the contributors of [SD3](https://huggingface.co/stabilityai/stable-diffusion-3-medium), [Qwen](https://huggingface.co/Qwen), [umt5-xxl](https://huggingface.co/google/umt5-xxl), [diffusers](https://github.com/huggingface/diffusers) and [HuggingFace](https://huggingface.co) repositories.

## Contact

Join our community:  [Discord](https://discord.gg/AKNgpMK4Yj) or [WeChat groups](https://gw.alicdn.com/imgextra/i2/O1CN01tqjWFi1ByuyehkTSB_!!6000000000015-0-tps-611-1279.jpg)!