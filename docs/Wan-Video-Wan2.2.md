# Wan2.2: Unleashing Advanced Video Generation - [Explore the Original Repo](https://github.com/Wan-Video/Wan2.2)

**Create stunning, high-quality videos with Wan2.2, the open-source video generation model that delivers cinematic aesthetics and advanced features.**

[![Wan2.2 Logo](assets/logo.png)](https://github.com/Wan-Video/Wan2.2)

**Key Features:**

*   ✅ **Mixture-of-Experts (MoE) Architecture:** Enhanced model capacity with optimized inference costs.
*   ✅ **Cinematic-Level Aesthetics:** Generate videos with precise control over lighting, composition, and color.
*   ✅ **Complex Motion Generation:** Produces videos with enhanced realism and detailed movement.
*   ✅ **Efficient High-Definition Hybrid TI2V:** Supports text-to-video and image-to-video at 720P/24fps on consumer-grade GPUs.
*   ✅ **Speech-to-Video Generation:**  Transform audio prompts and reference images into dynamic visuals.
*   ✅ **Character Animation & Replacement:** Animate and replace characters with realistic movement.

**Links:**

*   🌐 [Wan Website](https://wan.video)
*   🖥️ [GitHub](https://github.com/Wan-Video/Wan2.2)
*   🤗 [Hugging Face](https://huggingface.co/Wan-AI/)
*   🤖 [ModelScope](https://modelscope.cn/organization/Wan-AI)
*   📑 [Research Paper](https://arxiv.org/abs/2503.20314)
*   📑 [Blog](https://wan.video/welcome?spm=a2ty_o02.30011076.0.0.6c9ee41eCcluqg)
*   💬 [Discord](https://discord.gg/AKNgpMK4Yj)
*   📖 [User Guide (English)](https://alidocs.dingtalk.com/i/nodes/EpGBa2Lm8aZxe5myC99MelA2WgN7R35y)
*   📖 [User Guide (中文)](https://alidocs.dingtalk.com/i/nodes/jb9Y4gmKWrx9eo4dCql9LlbYJGXn6lpz)

---

## Introduction

Wan2.2 represents a significant advancement in open-source video generation, building upon the success of Wan2.1. This release focuses on enhancing the quality and capabilities of video generation through several key innovations:

*   **MoE Architecture:** This design allows for increased model capacity, while maintaining cost-efficiency during inference.
*   **Advanced Training Data:**  The model leverages meticulously curated aesthetic data, contributing to more precise and controllable video styles.
*   **High-Compression TI2V:** This feature enables the creation of 720P videos on consumer-grade hardware.

## Key Innovations in Detail

### Mixture-of-Experts (MoE) Architecture

The MoE architecture has been integrated into the video generation diffusion model to provide an efficient approach to scale model size while keeping inference costs nearly unchanged. This A14B model series adopts a two-expert design tailored to the denoising process of diffusion models: a high-noise expert for the early stages, focusing on overall layout; and a low-noise expert for the later stages, refining video details. Each expert model has about 14B parameters, resulting in a total of 27B parameters but only 14B active parameters per step, keeping inference computation and GPU memory nearly unchanged.

### Efficient High-Definition Hybrid TI2V

The TI2V-5B model offers efficient deployment options.  Utilizing a high-compression Wan2.2-VAE achieves a compression ratio of 64 while maintaining high-quality video reconstruction.  TI2V-5B can generate 720P videos in under 9 minutes on a single consumer-grade GPU, and supports both text-to-video and image-to-video generation.

### Performance & Benchmarks

Wan2.2 has been benchmarked against leading commercial models, demonstrating superior performance in key areas. (See [Performance.png](assets/performance.png) for detailed comparison)

## Latest Updates

*   **Sep 19, 2025:**  🚀 Introduced [Wan2.2-Animate-14B](https://humanaigc.github.io/wan-animate), a model for character animation and replacement.  Available on [wan.video](https://wan.video/), [ModelScope Studio](https://www.modelscope.cn/studios/Wan-AI/Wan2.2-Animate), and [HuggingFace Space](https://huggingface.co/spaces/Wan-AI/Wan2.2-Animate).
*   **Aug 26, 2025:**  🎵 Released [Wan2.2-S2V-14B](https://humanaigc.github.io/wan-s2v-webpage), an audio-driven cinematic video generation model, with inference code and model weights. Try it on [wan.video](https://wan.video/), [ModelScope Gradio](https://www.modelscope.cn/studios/Wan-AI/Wan2.2-S2V), or [HuggingFace Gradio](https://huggingface.co/spaces/Wan-AI/Wan2.2-S2V).
*   **Jul 28, 2025:**  👋 HF space for TI2V-5B is open:  [HF space](https://huggingface.co/spaces/Wan-AI/Wan-2.2-5B) 
*   **Jul 28, 2025:** 👋 Wan2.2 has been integrated into ComfyUI ([CN](https://docs.comfy.org/zh-CN/tutorials/video/wan/wan2_2) | [EN](https://docs.comfy.org/tutorials/video/wan/wan2_2)). Enjoy!
*   **Jul 28, 2025:** 👋 Wan2.2's T2V, I2V and TI2V have been integrated into Diffusers ([T2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) | [I2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers) | [TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers)). Feel free to give it a try!
*   **Jul 28, 2025:** 👋 We've released the inference code and model weights of **Wan2.2**.
*   **Sep 5, 2025:** 👋 We add text-to-speech synthesis support with [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) for Speech-to-Video generation task.

## Community Contributions

*   [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)
*   [Kijai's ComfyUI WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper)
*   [Cache-dit](https://github.com/vipshop/cache-dit)
*   [FastVideo](https://github.com/hao-ai-lab/FastVideo)

## Model Download & Usage

### Installation

```bash
git clone https://github.com/Wan-Video/Wan2.2.git
cd Wan2.2
pip install -r requirements.txt
pip install -r requirements_s2v.txt  # for Speech-to-Video
```

### Available Models

| Model                  | Download Link                                                                                                | Description                                                     |
| :--------------------- | :----------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------- |
| T2V-A14B               | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B) 🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-T2V-A14B)  | Text-to-Video (480P/720P)                                       |
| I2V-A14B               | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B) 🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-I2V-A14B)  | Image-to-Video (480P/720P)                                      |
| TI2V-5B                | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) 🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-TI2V-5B)  | Text-Image-to-Video (720P)                                    |
| S2V-14B                | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-S2V-14B) 🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-S2V-14B)  | Speech-to-Video (480P/720P)                                   |
| Animate-14B          | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-Animate-14B) 🤖 [ModelScope](https://www.modelscope.cn/models/Wan-AI/Wan2.2-Animate-14B) | Character Animation & Replacement                             |

> **Note:** TI2V-5B supports 720P video generation at 24 FPS.

### Model Download

```bash
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.2-T2V-A14B --local-dir ./Wan2.2-T2V-A14B
```

or

```bash
pip install modelscope
modelscope download Wan-AI/Wan2.2-T2V-A14B --local_dir ./Wan2.2-T2V-A14B
```

### Run Text-to-Video Generation

*Single GPU*:

```bash
python generate.py  --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --offload_model True --convert_model_dtype --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
```

*Multi-GPU (FSDP + DeepSpeed Ulysses)*:

```bash
torchrun --nproc_per_node=8 generate.py --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
```

### Run Image-to-Video Generation

*Single GPU*:

```bash
python generate.py --task i2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-I2V-A14B --offload_model True --convert_model_dtype --image examples/i2v_input.JPG --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
```

*Multi-GPU (FSDP + DeepSpeed Ulysses)*:

```bash
torchrun --nproc_per_node=8 generate.py --task i2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-I2V-A14B --image examples/i2v_input.JPG --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
```

### Run Text-Image-to-Video Generation

*Single GPU*:

```bash
python generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ./Wan2.2-TI2V-5B --offload_model True --convert_model_dtype --t5_cpu --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage"
```

*Multi-GPU (FSDP + DeepSpeed Ulysses)*:

```bash
torchrun --nproc_per_node=8 generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ./Wan2.2-TI2V-5B --dit_fsdp --t5_fsdp --ulysses_size 8 --image examples/i2v_input.JPG --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
```

### Run Speech-to-Video Generation

*Single GPU*:

```bash
python generate.py  --task s2v-14B --size 1024*704 --ckpt_dir ./Wan2.2-S2V-14B/ --offload_model True --convert_model_dtype --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard."  --image "examples/i2v_input.JPG" --audio "examples/talk.wav"
```

*Multi-GPU (FSDP + DeepSpeed Ulysses)*:

```bash
torchrun --nproc_per_node=8 generate.py --task s2v-14B --size 1024*704 --ckpt_dir ./Wan2.2-S2V-14B/ --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard." --image "examples/i2v_input.JPG" --audio "examples/talk.wav"
```

### Run Wan-Animate

#### Preprocessing

**Animation:**
```bash
python ./wan/modules/animate/preprocess/preprocess_data.py \
    --ckpt_path ./Wan2.2-Animate-14B/process_checkpoint \
    --video_path ./examples/wan_animate/animate/video.mp4 \
    --refer_path ./examples/wan_animate/animate/image.jpeg \
    --save_path ./examples/wan_animate/animate/process_results \
    --resolution_area 1280 720 \
    --retarget_flag \
    --use_flux
```

**Replacement:**
```bash
python ./wan/modules/animate/preprocess/preprocess_data.py \
    --ckpt_path ./Wan2.2-Animate-14B/process_checkpoint \
    --video_path ./examples/wan_animate/replace/video.mp4 \
    --refer_path ./examples/wan_animate/replace/image.jpeg \
    --save_path ./examples/wan_animate/replace/process_results \
    --resolution_area 1280 720 \
    --iterations 3 \
    --k 7 \
    --w_len 1 \
    --h_len 1 \
    --replace_flag
```

#### Single GPU Inference Animation Mode:
```bash
python generate.py --task animate-14B --ckpt_dir ./Wan2.2-Animate-14B/ --src_root_path ./examples/wan_animate/animate/process_results/ --refert_num 1
```

#### Multi-GPU Inference Animation Mode:

```bash
python -m torch.distributed.run --nnodes 1 --nproc_per_node 8 generate.py --task animate-14B --ckpt_dir ./Wan2.2-Animate-14B/ --src_root_path ./examples/wan_animate/animate/process_results/ --refert_num 1 --dit_fsdp --t5_fsdp --ulysses_size 8
```

#### Single GPU Inference Replacement Mode:

```bash
python generate.py --task animate-14B --ckpt_dir ./Wan2.2-Animate-14B/ --src_root_path ./examples/wan_animate/replace/process_results/ --refert_num 1 --replace_flag --use_relighting_lora 
```

#### Multi-GPU Inference Replacement Mode:

```bash
python -m torch.distributed.run --nnodes 1 --nproc_per_node 8 generate.py --task animate-14B --ckpt_dir ./Wan2.2-Animate-14B/ --src_root_path ./examples/wan_animate/replace/process_results/src_pose.mp4  --refert_num 1 --replace_flag --use_relighting_lora --dit_fsdp --t5_fsdp --ulysses_size 8
```

### Computational Efficiency

See [comp_effic.png](assets/comp_effic.png) for GPU performance.

---

## Technical Details

### Mixture-of-Experts (MoE) Architecture

The A14B model series uses MoE architecture, which utilizes separate experts for the early (high-noise) and late (low-noise) stages of the denoising process.

### Efficient High-Definition Hybrid TI2V

TI2V-5B is a high-compression model using a Wan2.2-VAE with a 4x16x16 compression ratio.

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

Wan2.2 is licensed under the Apache 2.0 License.

## Acknowledgements

Thank you to the contributors of [SD3](https://huggingface.co/stabilityai/stable-diffusion-3-medium), [Qwen](https://huggingface.co/Qwen), [umt5-xxl](https://huggingface.co/google/umt5-xxl), [diffusers](https://github.com/huggingface/diffusers) and [HuggingFace](https://huggingface.co) for their open research.

## Contact

Join our [Discord](https://discord.gg/AKNgpMK4Yj) or [WeChat groups](https://gw.alicdn.com/imgextra/i2/O1CN01tqjWFi1ByuyehkTSB_!!6000000000015-0-tps-611-1279.jpg) for support and updates.