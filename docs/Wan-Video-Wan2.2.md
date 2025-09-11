# Wan2.2: Unleash Cinematic Video Generation with Advanced AI

**Create stunning, high-quality videos with Wan2.2, a cutting-edge open-source video generative model.**  ([Original Repo](https://github.com/Wan-Video/Wan2.2))

## Key Features

*   ✨ **Mixture-of-Experts (MoE) Architecture:**  Leverages a MoE architecture for enhanced model capacity, while maintaining computational efficiency.
*   🎬 **Cinematic Aesthetics:** Trained on meticulously curated aesthetic data, enabling precise control over lighting, composition, and color grading for stunning visuals.
*   🎥 **Complex Motion Generation:** Trained on a significantly larger dataset compared to Wan2.1, resulting in improved generalization across motions, semantics, and aesthetics, achieving top performance.
*   🚀 **Efficient High-Definition Hybrid TI2V:** Offers a fast and efficient 5B model (TI2V-5B) supporting 720P generation at 24fps on consumer-grade GPUs, ideal for both research and industry applications.
*   🔊 **Speech-to-Video Capabilities:** Generate videos directly from audio with the S2V-14B model.

## What's New

*   **August 26, 2025:** Launched the **Wan2.2-S2V-14B**, an audio-driven cinematic video generation model with inference code, model weights, and a technical report. You can try it out on [wan.video](https://wan.video/), [ModelScope Gradio](https://www.modelscope.cn/studios/Wan-AI/Wan2.2-S2V), or [HuggingFace Gradio](https://huggingface.co/spaces/Wan-AI/Wan2.2-S2V)!
*   **July 28, 2025:** Added integration in ComfyUI ([CN](https://docs.comfy.org/zh-CN/tutorials/video/wan/wan2_2) | [EN](https://docs.comfy.org/tutorials/video/wan/wan2_2)).
*   **July 28, 2025:** Released the [Hugging Face Space](https://huggingface.co/spaces/Wan-AI/Wan-2.2-5B) using the TI2V-5B model.
*   **July 28, 2025:** Incorporated Wan2.2's T2V, I2V, and TI2V models into Diffusers ([T2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) | [I2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers) | [TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers)).
*   **July 28, 2025:** Released inference code and model weights for Wan2.2.
*   **September 5, 2025:** Added text-to-speech support with [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) for Speech-to-Video generation.

## Video Demos

<div align="center">
  <video src="https://github.com/user-attachments/assets/b63bfa58-d5d7-4de6-a1a2-98970b06d9a7" width="70%" poster=""> </video>
</div>

## Community Contributions

*   [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) - Supports Wan 2.2 with low-GPU-memory features, FP8 quantization, sequence parallelism, LoRA training, and full training.
*   [Kijai's ComfyUI WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper) - An alternative implementation of Wan models for ComfyUI.
*   [Cache-dit](https://github.com/vipshop/cache-dit) -  Fully Cache Acceleration support for Wan2.2 MoE with DBCache, TaylorSeer and Cache CFG.
*   [FastVideo](https://github.com/hao-ai-lab/FastVideo) - Includes distilled Wan models with sparse attention that significantly speed up inference time.

## Quick Start

### Installation

1.  Clone the repository:

    ```bash
    git clone https://github.com/Wan-Video/Wan2.2.git
    cd Wan2.2
    ```

2.  Install dependencies:

    ```bash
    # Ensure torch >= 2.4.0
    # If the installation of `flash_attn` fails, try installing the other packages first and install `flash_attn` last
    pip install -r requirements.txt
    # If you want to use CosyVoice to synthesize speech for Speech-to-Video Generation, please install requirements_s2v.txt additionally
    pip install -r requirements_s2v.txt
    ```

### Model Download

| Model               | Download Links                                                                                                                              | Description                                   |
|---------------------|---------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------|
| T2V-A14B           | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B)    🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-T2V-A14B)    | Text-to-Video MoE model (480P & 720P)          |
| I2V-A14B           | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B)    🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-I2V-A14B)    | Image-to-Video MoE model (480P & 720P)         |
| TI2V-5B            | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B)     🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-TI2V-5B)     | High-compression VAE, T2V+I2V (720P)          |
| S2V-14B            | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-S2V-14B)     🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-S2V-14B)     | Speech-to-Video model (480P & 720P)           |

> 💡 The TI2V-5B model supports 720P video generation at **24 FPS**.

#### Download using huggingface-cli:

```bash
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.2-T2V-A14B --local-dir ./Wan2.2-T2V-A14B
```

#### Download using modelscope-cli:

```bash
pip install modelscope
modelscope download Wan-AI/Wan2.2-T2V-A14B --local_dir ./Wan2.2-T2V-A14B
```

### Running Inference

The repository offers multiple inference options, including text-to-video, image-to-video, and speech-to-video.

#### Text-to-Video Generation

1.  **Without Prompt Extension (Single-GPU):**

    ```bash
    python generate.py  --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --offload_model True --convert_model_dtype --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
    ```

    > 💡 For the Image-to-Video task, the `size` parameter represents the area of the generated video, with the aspect ratio following that of the original input image.
    > 💡This command can run on a GPU with at least 80GB VRAM.

    > 💡If you encounter OOM (Out-of-Memory) issues, you can use the `--offload_model True`, `--convert_model_dtype` and `--t5_cpu` options to reduce GPU memory usage.

2.  **Multi-GPU Inference (FSDP + DeepSpeed Ulysses):**

    ```bash
    torchrun --nproc_per_node=8 generate.py --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
    ```

3.  **Using Prompt Extension:** Prompt extension enhances video quality by enriching details using methods like the Dashscope API or a local Qwen model. Use the following commands:

    *   **Dashscope API:**
        ```bash
        DASH_API_KEY=your_key torchrun --nproc_per_node=8 generate.py  --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage" --use_prompt_extend --prompt_extend_method 'dashscope' --prompt_extend_target_lang 'zh'
        ```
    *   **Local Qwen Model:**
        ```bash
        torchrun --nproc_per_node=8 generate.py  --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage" --use_prompt_extend --prompt_extend_method 'local_qwen' --prompt_extend_target_lang 'zh'
        ```

#### Image-to-Video Generation

1.  **Single-GPU Inference:**

    ```bash
    python generate.py --task i2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-I2V-A14B --offload_model True --convert_model_dtype --image examples/i2v_input.JPG --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
    ```
    > This command can run on a GPU with at least 80GB VRAM.

2.  **Multi-GPU Inference (FSDP + DeepSpeed Ulysses):**

    ```bash
    torchrun --nproc_per_node=8 generate.py --task i2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-I2V-A14B --image examples/i2v_input.JPG --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
    ```

3.  **Image-to-Video Generation without prompt:**

    ```bash
    DASH_API_KEY=your_key torchrun --nproc_per_node=8 generate.py --task i2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-I2V-A14B --prompt '' --image examples/i2v_input.JPG --dit_fsdp --t5_fsdp --ulysses_size 8 --use_prompt_extend --prompt_extend_method 'dashscope'
    ```

#### Text-Image-to-Video Generation

1.  **Single-GPU Text-to-Video Inference:**

    ```bash
    python generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ./Wan2.2-TI2V-5B --offload_model True --convert_model_dtype --t5_cpu --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage"
    ```
    > The 720P resolution of the Text-Image-to-Video task is `1280*704` or `704*1280`.
    > This command can run on a GPU with at least 24GB VRAM (e.g, RTX 4090 GPU).

2.  **Single-GPU Image-to-Video Inference:**

    ```bash
    python generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ./Wan2.2-TI2V-5B --offload_model True --convert_model_dtype --t5_cpu --image examples/i2v_input.JPG --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
    ```

3.  **Multi-GPU Inference (FSDP + DeepSpeed Ulysses):**

    ```bash
    torchrun --nproc_per_node=8 generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ./Wan2.2-TI2V-5B --dit_fsdp --t5_fsdp --ulysses_size 8 --image examples/i2v_input.JPG --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
    ```

#### Speech-to-Video Generation

1.  **Single-GPU Speech-to-Video Inference:**

    ```bash
    python generate.py  --task s2v-14B --size 1024*704 --ckpt_dir ./Wan2.2-S2V-14B/ --offload_model True --convert_model_dtype --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard."  --image "examples/i2v_input.JPG" --audio "examples/talk.wav"
    # Without setting --num_clip, the generated video length will automatically adjust based on the input audio length

    # You can use CosyVoice to generate audio with --enable_tts
    python generate.py  --task s2v-14B --size 1024*704 --ckpt_dir ./Wan2.2-S2V-14B/ --offload_model True --convert_model_dtype --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard."  --image "examples/i2v_input.JPG" --enable_tts --tts_prompt_audio "examples/zero_shot_prompt.wav" --tts_prompt_text "希望你以后能够做的比我还好呦。" --tts_text "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。"
    ```
    > This command can run on a GPU with at least 80GB VRAM.

2.  **Multi-GPU Inference (FSDP + DeepSpeed Ulysses):**

    ```bash
    torchrun --nproc_per_node=8 generate.py --task s2v-14B --size 1024*704 --ckpt_dir ./Wan2.2-S2V-14B/ --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard." --image "examples/i2v_input.JPG" --audio "examples/talk.wav"
    ```

3.  **Pose + Audio Driven Generation:**

    ```bash
    torchrun --nproc_per_node=8 generate.py --task s2v-14B --size 1024*704 --ckpt_dir ./Wan2.2-S2V-14B/ --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "a person is singing" --image "examples/pose.png" --audio "examples/sing.MP3" --pose_video "./examples/pose.mp4"
    ```

## Computational Efficiency

*   **Wan2.2** models demonstrate efficiency across different GPUs.  See the table in the original README for details.

## Technical Details

Wan2.2 is built upon these core innovations:

*   **Mixture-of-Experts (MoE) Architecture**: Improves overall model capacity, while maintaining the same computational cost.
*   **Efficient High-Definition Hybrid TI2V**: Utilizes a high-compression Wan2.2-VAE for enhanced performance and 720P generation at 24fps.
*   **Advanced Training Data**: Enhanced training data has improved motion, semantics, and aesthetics, leading to top-tier performance.
*   **Comparisons to SOTAs**: Performance results in the original README.

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

Wan2.2 is licensed under the Apache 2.0 License. Please review the [LICENSE.txt](LICENSE.txt) for full details.

## Acknowledgements

The project would like to thank the contributors of SD3, Qwen, umt5-xxl, diffusers, and HuggingFace.

## Contact

*   Join our [Discord](https://discord.gg/AKNgpMK4Yj) or [WeChat groups](https://gw.alicdn.com/imgextra/i2/O1CN01tqjWFi1ByuyehkTSB_!!6000000000015-0-tps-611-1279.jpg) for support and updates.