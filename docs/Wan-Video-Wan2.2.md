# Wan2.2: Generate Stunning Videos with Advanced AI Models

**Unleash your creativity with Wan2.2, the cutting-edge video generation models that bring your ideas to life with unparalleled quality.**  ([Original Repo](https://github.com/Wan-Video/Wan2.2))

<p align="center">
    <img src="assets/logo.png" width="400"/>
</p>

**Key Features:**

*   **Mixture-of-Experts (MoE) Architecture:** Leverages MoE for enhanced model capacity and efficient performance.
*   **Cinematic Aesthetics:** Generates videos with customizable styles and precise control over lighting, composition, and more.
*   **Enhanced Motion & Data:** Trained on significantly larger datasets for improved generalization across motions, semantics, and aesthetics.
*   **Efficient High-Definition Hybrid TI2V:** Includes a 5B model (TI2V-5B) that supports both text-to-video and image-to-video at 720P and 24fps on consumer-grade GPUs.
*   **Versatile Generation:** Supports Text-to-Video (T2V), Image-to-Video (I2V), and Text-Image-to-Video (TI2V) generation.
*   **Integration Friendly:** Seamlessly integrates into popular platforms like ComfyUI and Diffusers.

**Quick Links:**

*   **Paper:** [https://arxiv.org/abs/2503.20314](https://arxiv.org/abs/2503.20314)
*   **Hugging Face:** [https://huggingface.co/Wan-AI/](https://huggingface.co/Wan-AI/)
*   **ModelScope:** [https://modelscope.cn/organization/Wan-AI](https://modelscope.cn/organization/Wan-AI)
*   **Discord:** [https://discord.gg/AKNgpMK4Yj](https://discord.gg/AKNgpMK4Yj)

---

## What's New in Wan2.2?

Wan2.2 builds upon Wan2.1, introducing significant advancements in video generation:

*   **MoE Architecture:**  Employing Mixture-of-Experts architecture to increase model size while maintaining computational efficiency.
*   **Enhanced Data & Aesthetics:** Leverages meticulously curated aesthetic data for precise style generation and a substantially larger training dataset for improved video quality.
*   **Efficient High-Definition TI2V:** The TI2V-5B model achieves impressive speed and quality, making high-definition video generation accessible.

## Video Demos

<div align="center">
  <video src="https://github.com/user-attachments/assets/b63bfa58-d5d7-4de6-a1a2-98970b06d9a7" width="70%" poster=""> </video>
</div>

## Recent Updates

*   **July 28, 2025:** [HF space](https://huggingface.co/spaces/Wan-AI/Wan-2.2-5B) launched using the TI2V-5B model.
*   **July 28, 2025:** Wan2.2 integrated into ComfyUI ([CN](https://docs.comfy.org/zh-CN/tutorials/video/wan/wan2_2) | [EN](https://docs.comfy.org/tutorials/video/wan/wan2_2)).
*   **July 28, 2025:** T2V, I2V, and TI2V models integrated into Diffusers ([T2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) | [I2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers) | [TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers)).
*   **July 28, 2025:** Inference code and model weights of Wan2.2 released.

## Community Contributions

We appreciate community contributions and are excited to showcase projects built upon Wan2.1 and Wan2.2.

*   [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio): Offers comprehensive support for Wan 2.2.
*   [Kijai's ComfyUI WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper): Provides an alternative Wan model implementation for ComfyUI.

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

| Model        | Download Link                                                                                                                              | Description                        |
| :----------- | :----------------------------------------------------------------------------------------------------------------------------------------- | :--------------------------------- |
| T2V-A14B     | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B)  🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-T2V-A14B)      | Text-to-Video (480P & 720P)       |
| I2V-A14B     | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B)  🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-I2V-A14B)      | Image-to-Video (480P & 720P)      |
| TI2V-5B      | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B)   🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-TI2V-5B)       | Text-Image-to-Video (720P, 24 FPS) |

**Download using huggingface-cli:**

```bash
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.2-T2V-A14B --local-dir ./Wan2.2-T2V-A14B
```

**Download using modelscope-cli:**

```bash
pip install modelscope
modelscope download Wan-AI/Wan2.2-T2V-A14B --local_dir ./Wan2.2-T2V-A14B
```

### Running Wan2.2

#### Text-to-Video Generation

**(1) Basic Inference**

```bash
python generate.py  --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --offload_model True --convert_model_dtype --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
```

*   Requires at least 80GB VRAM.  Use `--offload_model True`, `--convert_model_dtype`, and `--t5_cpu` for lower memory usage.

**(2) Multi-GPU Inference**

```bash
torchrun --nproc_per_node=8 generate.py --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
```

**(3) Using Prompt Extension (Recommended)**

*   **Using Dashscope API:**

    *   Requires `DASH_API_KEY` environment variable and API key.
    *   Use `qwen-plus` for text-to-video and `qwen-vl-max` for image-to-video.
    *   You can change model for extension using the parameter `--prompt_extend_model`.

    ```bash
    DASH_API_KEY=your_key torchrun --nproc_per_node=8 generate.py  --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage" --use_prompt_extend --prompt_extend_method 'dashscope' --prompt_extend_target_lang 'zh'
    ```

*   **Using a Local Model:**

    *   Use models like `Qwen/Qwen2.5-14B-Instruct`, `Qwen/Qwen2.5-7B-Instruct` (for T2V) and `Qwen/Qwen2.5-VL-7B-Instruct` (for I2V).
    *   Specify the model path with `--prompt_extend_model`.

    ```bash
    torchrun --nproc_per_node=8 generate.py  --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage" --use_prompt_extend --prompt_extend_method 'local_qwen' --prompt_extend_target_lang 'zh'
    ```

#### Image-to-Video Generation

**(1) Single-GPU Inference**

```bash
python generate.py --task i2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-I2V-A14B --offload_model True --convert_model_dtype --image examples/i2v_input.JPG --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
```

*   `size` respects the aspect ratio of the input image.

**(2) Multi-GPU Inference**

```bash
torchrun --nproc_per_node=8 generate.py --task i2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-I2V-A14B --image examples/i2v_input.JPG --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
```

**(3) Image-to-Video Generation without prompt**

```bash
DASH_API_KEY=your_key torchrun --nproc_per_node=8 generate.py --task i2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-I2V-A14B --prompt '' --image examples/i2v_input.JPG --dit_fsdp --t5_fsdp --ulysses_size 8 --use_prompt_extend --prompt_extend_method 'dashscope'
```

#### Text-Image-to-Video Generation

**(1) Single-GPU Inference**

*   **Text-to-Video:**

```bash
python generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ./Wan2.2-TI2V-5B --offload_model True --convert_model_dtype --t5_cpu --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage"
```

*   **Image-to-Video:**

```bash
python generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ./Wan2.2-TI2V-5B --offload_model True --convert_model_dtype --t5_cpu --image examples/i2v_input.JPG --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
```

*   Text-to-Video (720P) resolution is `1280*704` or `704*1280`.
*   Requires a GPU with at least 24GB VRAM (e.g., RTX 4090).

**(2) Multi-GPU Inference**

```bash
torchrun --nproc_per_node=8 generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ./Wan2.2-TI2V-5B --dit_fsdp --t5_fsdp --ulysses_size 8 --image examples/i2v_input.JPG --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
```

## Computational Efficiency

The following table summarizes computational efficiency on different GPUs.

<div align="center">
    <img src="assets/comp_effic.png" alt="" style="width: 80%;" />
</div>

> Test settings are provided in the original README.

## Technical Details

### (1) Mixture-of-Experts (MoE) Architecture

Wan2.2 utilizes an MoE architecture to improve performance. The A14B model series employs a two-expert design, with experts specializing in different noise levels.

### (2) Efficient High-Definition Hybrid TI2V

The TI2V-5B model uses a high-compression Wan2.2-VAE, achieving a high compression ratio while maintaining video quality.

### (3) Performance

Wan2.2 demonstrates superior performance compared to leading commercial models, as shown in the comparisons in the original README.

## Citation

If you use Wan2.2, please cite the following paper:

```
@article{wan2025,
      title={Wan: Open and Advanced Large-Scale Video Generative Models},
      author={Team Wan and Ang Wang and Baole Ai and Bin Wen and Chaojie Mao and Chen-Wei Xie and Di Chen and Feiwu Yu and Haiming Zhao and Jianxiao Yang and Jianyuan Zeng and Jiayu Wang and Jingfeng Zhang and Jingren Zhou and Jinkai Wang and Jixuan Chen and Kai Zhu and Kang Zhao and Keyu Yan and Lianghua Huang and Mengyang Feng and Ningyi Zhang and Pandeng Li and Pingyu Wu and Ruihang Chu and Ruili Feng and Shiwei Zhang and Siyang Sun and Tao Fang and Tianxing Wang and Tianyi Gui and Tingyu Weng and Tong Shen and Wei Lin and Wei Wang and Wei Wang and Wenmeng Zhou and Wente Wang and Wenting Shen and Wenyuan Yu and Xianzhong Shi and Xiaoming Huang and Xin Xu and Yan Kou and Yangyu Lv and Yifei Li and Yijing Liu and Yiming Wang and Yingya Zhang and Yitong Huang and Yong Li and You Wu and Yu Liu and Yulin Pan and Yun Zheng and Yuntao Hong and Yupeng Shi and Yutong Feng and Zeyinzi Jiang and Zhen Han and Zhi-Fan Wu and Ziyu Liu},
      journal = {arXiv preprint arXiv:2503.20314},
      year={2025}
}
```

## License

Wan2.2 is licensed under the Apache 2.0 License.  See [LICENSE.txt](LICENSE.txt) for details.

## Acknowledgements

Thanks to the contributors of [SD3](https://huggingface.co/stabilityai/stable-diffusion-3-medium), [Qwen](https://huggingface.co/Qwen), [umt5-xxl](https://huggingface.co/google/umt5-xxl), [diffusers](https://github.com/huggingface/diffusers) and [HuggingFace](https://huggingface.co).

## Contact

*   Join our [Discord](https://discord.gg/AKNgpMK4Yj) or [WeChat groups](https://gw.alicdn.com/imgextra/i2/O1CN01tqjWFi1ByuyehkTSB_!!6000000000015-0-tps-611-1279.jpg) for support and updates.