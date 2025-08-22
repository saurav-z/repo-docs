# Wan2.2: Revolutionizing Video Generation with Advanced AI

**Wan2.2** is an open-source, state-of-the-art video generation model, pushing the boundaries of AI-driven video creation.  Visit the [original repo](https://github.com/Wan-Video/Wan2.2) for the code and more information.

## Key Features & Benefits:

*   **Mixture-of-Experts (MoE) Architecture**: Enhanced efficiency and quality by leveraging specialized expert models for different stages of the denoising process.
*   **Cinematic Aesthetics**:  Create videos with precise control over lighting, composition, and color tones, resulting in professional-grade visuals.
*   **Advanced Motion Generation**: Generates more complex and realistic video content with superior generalization capabilities.
*   **Efficient High-Definition Hybrid TI2V**:  Generate high-quality videos at 720P resolution and 24fps, even on consumer-grade GPUs (e.g., 4090), using our innovative Wan2.2-VAE.
*   **Supports Multiple Generation Modes**:  Text-to-Video (T2V), Image-to-Video (I2V), and Text-Image-to-Video (TI2V).

## What's New in Wan2.2?

Wan2.2 builds on the success of Wan2.1 with significant improvements in video quality and model capabilities. Here's a deeper look:

### Mixture-of-Experts (MoE) Architecture

*   **Increased Efficiency**:  The MoE architecture significantly boosts model capacity while maintaining manageable computational costs.
*   **Specialized Experts**:  High-noise and low-noise experts are tailored for distinct denoising stages.  The model dynamically switches between these experts based on the signal-to-noise ratio (SNR).
*   **Improved Validation Loss**:  The MoE architecture showcases superior convergence.

### Efficient High-Definition Hybrid TI2V

*   **High-Compression Design**:  The TI2V-5B model, built on a 5B dense model and high-compression Wan2.2-VAE, offers a $4\times16\times16$ compression ratio.  This results in a significant increase in overall compression.
*   **High-Quality Reconstruction**:  TI2V-5B enables high-quality video reconstruction while maintaining high-quality.
*   **Text-to-Video and Image-to-Video**: The TI2V-5B model natively supports both text-to-video and image-to-video generation within a unified framework.
*   **Fast 720P Generation**:  Generate a 5-second 720P video in under 9 minutes on a single consumer-grade GPU.

### Performance Comparisons

*   Wan2.2 outperforms leading commercial closed-source models on our new Wan-Bench 2.0.

## Getting Started: Run Wan2.2

### Installation

```bash
git clone https://github.com/Wan-Video/Wan2.2.git
cd Wan2.2
# Ensure torch >= 2.4.0
# If the installation of `flash_attn` fails, try installing the other packages first and install `flash_attn` last
pip install -r requirements.txt
```

### Model Download

Available models:

| Model               | Download Links                                                                                                                              | Description                      |
| :------------------ | :------------------------------------------------------------------------------------------------------------------------------------------ | :------------------------------- |
| T2V-A14B            | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B)    🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-T2V-A14B)    | Text-to-Video MoE, 480P & 720P  |
| I2V-A14B            | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B)    🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-I2V-A14B)    | Image-to-Video MoE, 480P & 720P |
| TI2V-5B             | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B)     🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-TI2V-5B)     | High-Compression, T2V+I2V, 720P |

*Note: TI2V-5B supports 720P video generation at 24 FPS.*

Download models using Hugging Face CLI:

```bash
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.2-T2V-A14B --local-dir ./Wan2.2-T2V-A14B
```

Download models using ModelScope CLI:

```bash
pip install modelscope
modelscope download Wan-AI/Wan2.2-T2V-A14B --local_dir ./Wan2.2-T2V-A14B
```

### Run Text-to-Video Generation

*(1) Without Prompt Extension*

*   Single-GPU inference:

```bash
python generate.py  --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --offload_model True --convert_model_dtype --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
```

> 💡 Use `--offload_model True`, `--convert_model_dtype` and `--t5_cpu` to reduce GPU memory usage.

*   Multi-GPU inference using FSDP + DeepSpeed Ulysses:

```bash
torchrun --nproc_per_node=8 generate.py --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
```

*(2) Using Prompt Extension*

*   Use the Dashscope API for extension.
    *   Apply for a `dashscope.api_key` in advance ([EN](https://www.alibabacloud.com/help/en/model-studio/getting-started/first-api-call-to-qwen) | [CN](https://help.aliyun.com/zh/model-studio/getting-started/first-api-call-to-qwen)).
    *   Configure the environment variable `DASH_API_KEY` to specify the Dashscope API key. For users of Alibaba Cloud's international site, you also need to set the environment variable `DASH_API_URL` to 'https://dashscope-intl.aliyuncs.com/api/v1'. For more detailed instructions, please refer to the [dashscope document](https://www.alibabacloud.com/help/en/model-studio/developer-reference/use-qwen-by-calling-api?spm=a2c63.p38356.0.i1).
    *   Use the `qwen-plus` model for text-to-video tasks and `qwen-vl-max` for image-to-video tasks.
    *   You can modify the model used for extension with the parameter `--prompt_extend_model`. For example:

```bash
DASH_API_KEY=your_key torchrun --nproc_per_node=8 generate.py  --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage" --use_prompt_extend --prompt_extend_method 'dashscope' --prompt_extend_target_lang 'zh'
```
*   Using a local model for extension.
  * By default, the Qwen model on HuggingFace is used for this extension. Users can choose Qwen models or other models based on the available GPU memory size.
  * For text-to-video tasks, you can use models like `Qwen/Qwen2.5-14B-Instruct`, `Qwen/Qwen2.5-7B-Instruct` and `Qwen/Qwen2.5-3B-Instruct`.
  * For image-to-video tasks, you can use models like `Qwen/Qwen2.5-VL-7B-Instruct` and `Qwen/Qwen2.5-VL-3B-Instruct`.
  * Larger models generally provide better extension results but require more GPU memory.
  * You can modify the model used for extension with the parameter `--prompt_extend_model` , allowing you to specify either a local model path or a Hugging Face model. For example:

``` bash
torchrun --nproc_per_node=8 generate.py  --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage" --use_prompt_extend --prompt_extend_method 'local_qwen' --prompt_extend_target_lang 'zh'
```

### Run Image-to-Video Generation

*   Single-GPU inference:

```bash
python generate.py --task i2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-I2V-A14B --offload_model True --convert_model_dtype --image examples/i2v_input.JPG --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
```
> 💡 The `size` parameter respects the aspect ratio of the input image.

*   Multi-GPU inference using FSDP + DeepSpeed Ulysses:

```bash
torchrun --nproc_per_node=8 generate.py --task i2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-I2V-A14B --image examples/i2v_input.JPG --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
```

*   Image-to-Video Generation without prompt

```bash
DASH_API_KEY=your_key torchrun --nproc_per_node=8 generate.py --task i2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-I2V-A14B --prompt '' --image examples/i2v_input.JPG --dit_fsdp --t5_fsdp --ulysses_size 8 --use_prompt_extend --prompt_extend_method 'dashscope'
```

> 💡 Model can generate videos solely from the input image. Use prompt extension to generate prompt from the image.

### Run Text-Image-to-Video Generation

*   Single-GPU Text-to-Video inference:

```bash
python generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ./Wan2.2-TI2V-5B --offload_model True --convert_model_dtype --t5_cpu --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage"
```

> 💡 For the Text-Image-to-Video task, the 720P resolution is `1280*704` or `704*1280`.

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

The computational efficiency of different Wan2.2 models on different GPUs is as follows:

<div align="center">
    <img src="assets/comp_effic.png" alt="Computational Efficiency" style="width: 80%;" />
</div>

## Community Works

*   [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) provides comprehensive support for Wan 2.2, including low-GPU-memory layer-by-layer offload, FP8 quantization, sequence parallelism, LoRA training, full training.
*   [Kijai's ComfyUI WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper) is an alternative implementation of Wan models for ComfyUI. Thanks to its Wan-only focus, it's on the frontline of getting cutting edge optimizations and hot research features, which are often hard to integrate into ComfyUI quickly due to its more rigid structure.

## Useful Links

*   [Paper](https://arxiv.org/abs/2503.20314)
*   [Blog](https://wan.video/welcome?spm=a2ty_o02.30011076.0.0.6c9ee41eCcluqg)
*   [Discord](https://discord.gg/AKNgpMK4Yj)
*   [Hugging Face](https://huggingface.co/Wan-AI/)
*   [ModelScope](https://modelscope.cn/organization/Wan-AI)

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

Licensed under the [Apache 2.0 License](LICENSE.txt).

## Acknowledgements

Special thanks to the contributors of SD3, Qwen, umt5-xxl, diffusers and HuggingFace.