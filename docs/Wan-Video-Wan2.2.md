# Wan2.2: Unleashing the Power of Advanced Video Generation

**Wan2.2 offers cutting-edge, open-source video generation models, enabling users to create high-quality videos from text, images, and audio.  [Visit the original repo](https://github.com/Wan-Video/Wan2.2) to explore this groundbreaking technology.**

---

## Key Features of Wan2.2:

*   ✅ **Mixture-of-Experts (MoE) Architecture:** Leverages MoE to enhance model capacity while maintaining computational efficiency, delivering superior video quality.
*   ✅ **Cinematic-Level Aesthetics:** Incorporates meticulously curated aesthetic data with detailed labels for lighting, composition, contrast, and color tone, enabling more precise and controllable cinematic style generation.
*   ✅ **Enhanced Complex Motion Generation:** Trained on a significantly larger dataset (+65.6% images, +83.2% videos) compared to Wan2.1, resulting in improved generalization across various aspects of video generation.
*   ✅ **Efficient High-Definition Hybrid TI2V:** Introduces a 5B model with a 16x16x4 compression ratio, supporting 720P resolution at 24fps, and is optimized for consumer-grade GPUs like the 4090.

## Video Demos

<!-- Replace with a link to the most compelling demo video, or a collection of them.  This is critical. -->
[Demo Video Placeholder - Add a link to a great demo of the model's capabilities]

---

## Latest News & Updates

*   **[Wan2.2-Animate-14B](https://humanaigc.github.io/wan-animate):** A unified model for character animation and replacement.  Available on [wan.video](https://wan.video/), [ModelScope Studio](https://www.modelscope.cn/studios/Wan-AI/Wan2.2-Animate), and [HuggingFace Space](https://huggingface.co/spaces/Wan-AI/Wan2.2-Animate)!
*   **[Wan2.2-S2V-14B](https://humanaigc.github.io/wan-s2v-webpage):** Audio-driven cinematic video generation. Try it on [wan.video](https://wan.video/), [ModelScope Gradio](https://www.modelscope.cn/studios/Wan-AI/Wan2.2-S2V), or [HuggingFace Gradio](https://huggingface.co/spaces/Wan-AI/Wan2.2-S2V)!
*   **[HF space](https://huggingface.co/spaces/Wan-AI/Wan-2.2-5B) using the TI2V-5B model.**
*   **Wan2.2 integration into ComfyUI** ([CN](https://docs.comfy.org/zh-CN/tutorials/video/wan/wan2_2) | [EN](https://docs.comfy.org/tutorials/video/wan/wan2_2)).
*   **Wan2.2 T2V, I2V, and TI2V integration into Diffusers** ([T2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) | [I2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers) | [TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers)).
*   **Inference code and model weights of Wan2.2 have been released.**
*   **Text-to-speech synthesis support with [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) for Speech-to-Video generation task.**

---

## Community Contributions

*   **[DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)**
*   **[Kijai's ComfyUI WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper)**
*   **[Cache-dit](https://github.com/vipshop/cache-dit)**
*   **[FastVideo](https://github.com/hao-ai-lab/FastVideo)**

---

## Run Wan2.2: Getting Started

### Installation

```bash
git clone https://github.com/Wan-Video/Wan2.2.git
cd Wan2.2
# Ensure torch >= 2.4.0
# If the installation of `flash_attn` fails, try installing the other packages first and install `flash_attn` last
pip install -r requirements.txt
# If you want to use CosyVoice to synthesize speech for Speech-to-Video Generation, please install requirements_s2v.txt additionally
pip install -r requirements_s2v.txt
```

### Model Download

| Model              | Download Link                                                                                                                              | Description                                       |
|--------------------|---------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------|
| T2V-A14B           | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B)    🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-T2V-A14B)    | Text-to-Video MoE model, supports 480P & 720P     |
| I2V-A14B           | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B)    🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-I2V-A14B)    | Image-to-Video MoE model, supports 480P & 720P     |
| TI2V-5B            | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B)     🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-TI2V-5B)     | High-compression VAE, T2V+I2V, supports 720P      |
| S2V-14B            | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-S2V-14B)     🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-S2V-14B)     | Speech-to-Video model, supports 480P & 720P        |
| Animate-14B | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-Animate-14B) 🤖 [ModelScope](https://www.modelscope.cn/models/Wan-AI/Wan2.2-Animate-14B)  | Character animation and replacement               |

>  **Note:** The TI2V-5B model generates 720P videos at **24 FPS**.

Download models using huggingface-cli:
``` sh
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.2-T2V-A14B --local-dir ./Wan2.2-T2V-A14B
```

Download models using modelscope-cli:
``` sh
pip install modelscope
modelscope download Wan-AI/Wan2.2-T2V-A14B --local_dir ./Wan2.2-T2V-A14B
```

### Run Text-to-Video Generation

#### (1) Without Prompt Extension

-   Single-GPU inference
    ```bash
    python generate.py  --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --offload_model True --convert_model_dtype --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
    ```
    >  **Note:** This command needs at least 80GB of VRAM.  Use `--offload_model True`, `--convert_model_dtype`, and `--t5_cpu` for reduced GPU memory usage.

-   Multi-GPU inference (FSDP + DeepSpeed Ulysses)
    ```bash
    torchrun --nproc_per_node=8 generate.py --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
    ```

#### (2) Using Prompt Extension

-   **Using Dashscope API:**  Set up your `DASH_API_KEY` and optionally, `DASH_API_URL`.  Use `--prompt_extend_model`, `--prompt_extend_method`, and `--prompt_extend_target_lang` to specify the extension model and settings.
    ```bash
    DASH_API_KEY=your_key torchrun --nproc_per_node=8 generate.py  --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage" --use_prompt_extend --prompt_extend_method 'dashscope' --prompt_extend_target_lang 'zh'
    ```

-   **Using a Local Model for Extension:** Use `--prompt_extend_model` to specify the local model path or Hugging Face model.
    ```bash
    torchrun --nproc_per_node=8 generate.py  --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage" --use_prompt_extend --prompt_extend_method 'local_qwen' --prompt_extend_target_lang 'zh'
    ```

### Run Image-to-Video Generation

-   Single-GPU inference
    ```bash
    python generate.py --task i2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-I2V-A14B --offload_model True --convert_model_dtype --image examples/i2v_input.JPG --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
    ```
    >  **Note:** The `size` parameter represents the video area and follows the aspect ratio of the input image.

-   Multi-GPU inference (FSDP + DeepSpeed Ulysses)
    ```bash
    torchrun --nproc_per_node=8 generate.py --task i2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-I2V-A14B --image examples/i2v_input.JPG --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
    ```

-   Image-to-Video Generation without prompt
    ```bash
    DASH_API_KEY=your_key torchrun --nproc_per_node=8 generate.py --task i2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-I2V-A14B --prompt '' --image examples/i2v_input.JPG --dit_fsdp --t5_fsdp --ulysses_size 8 --use_prompt_extend --prompt_extend_method 'dashscope'
    ```

### Run Text-Image-to-Video Generation

-   Single-GPU Text-to-Video inference
    ```bash
    python generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ./Wan2.2-TI2V-5B --offload_model True --convert_model_dtype --t5_cpu --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage"
    ```
    >  **Note:** TI2V 720P resolution uses `1280*704` or `704*1280`.  The `--offload_model True`, `--convert_model_dtype`, and `--t5_cpu` options can be omitted if you have more VRAM.

-   Single-GPU Image-to-Video inference
    ```bash
    python generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ./Wan2.2-TI2V-5B --offload_model True --convert_model_dtype --t5_cpu --image examples/i2v_input.JPG --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
    ```

-   Multi-GPU inference (FSDP + DeepSpeed Ulysses)
    ```bash
    torchrun --nproc_per_node=8 generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ./Wan2.2-TI2V-5B --dit_fsdp --t5_fsdp --ulysses_size 8 --image examples/i2v_input.JPG --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
    ```

### Run Speech-to-Video Generation

-   Single-GPU Speech-to-Video inference
    ```bash
    python generate.py  --task s2v-14B --size 1024*704 --ckpt_dir ./Wan2.2-S2V-14B/ --offload_model True --convert_model_dtype --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard."  --image "examples/i2v_input.JPG" --audio "examples/talk.wav"
    # Without setting --num_clip, the generated video length will automatically adjust based on the input audio length

    # You can use CosyVoice to generate audio with --enable_tts
    python generate.py  --task s2v-14B --size 1024*704 --ckpt_dir ./Wan2.2-S2V-14B/ --offload_model True --convert_model_dtype --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard."  --image "examples/i2v_input.JPG" --enable_tts --tts_prompt_audio "examples/zero_shot_prompt.wav" --tts_prompt_text "希望你以后能够做的比我还好呦。" --tts_text "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。"
    ```
-   Multi-GPU inference (FSDP + DeepSpeed Ulysses)
    ```bash
    torchrun --nproc_per_node=8 generate.py --task s2v-14B --size 1024*704 --ckpt_dir ./Wan2.2-S2V-14B/ --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard." --image "examples/i2v_input.JPG" --audio "examples/talk.wav"
    ```
-   Pose + Audio driven generation
    ```bash
    torchrun --nproc_per_node=8 generate.py --task s2v-14B --size 1024*704 --ckpt_dir ./Wan2.2-S2V-14B/ --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "a person is singing" --image "examples/pose.png" --audio "examples/sing.MP3" --pose_video "./examples/pose.mp4"
    ```
    >  **Note:** The `--pose_video` parameter enables pose-driven generation.

### Run Wan-Animate

1.  **Preprocessing:** Follow the steps in the [UserGuider](https://github.com/Wan-Video/Wan2.2/blob/main/wan/modules/animate/preprocess/UserGuider.md) for data preparation, using either animation or replacement modes.  Examples are below:
    *   Animation Mode
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
    *   Replacement Mode
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

2.  **Animation Mode Inference:**
    *   Single-GPU
        ```bash
        python generate.py --task animate-14B --ckpt_dir ./Wan2.2-Animate-14B/ --src_root_path ./examples/wan_animate/animate/process_results/ --refert_num 1
        ```
    *   Multi-GPU
        ```bash
        python -m torch.distributed.run --nnodes 1 --nproc_per_node 8 generate.py --task animate-14B --ckpt_dir ./Wan2.2-Animate-14B/ --src_root_path ./examples/wan_animate/animate/process_results/ --refert_num 1 --dit_fsdp --t5_fsdp --ulysses_size 8
        ```

3.  **Replacement Mode Inference:**
    *   Single-GPU
        ```bash
        python generate.py --task animate-14B --ckpt_dir ./Wan2.2-Animate-14B/ --src_root_path ./examples/wan_animate/replace/process_results/ --refert_num 1 --replace_flag --use_relighting_lora
        ```
    *   Multi-GPU
        ```bash
        python -m torch.distributed.run --nnodes 1 --nproc_per_node 8 generate.py --task animate-14B --ckpt_dir ./Wan2.2-Animate-14B/ --src_root_path ./examples/wan_animate/replace/process_results/src_pose.mp4  --refert_num 1 --replace_flag --use_relighting_lora --dit_fsdp --t5_fsdp --ulysses_size 8
        ```

    >  **Important:** Do not use LoRA models trained on `Wan2.2` with **Wan-Animate**.

---

## Computational Efficiency

| Model            | GPU             | Total Time (s) / Peak GPU Memory (GB) |
|-----------------|-----------------|---------------------------------------|
| T2V-A14B (14B) | NVIDIA A100 80G |  [Results Table Placeholder -  Insert the actual table here]     |
| ...             | ...             | ...                                   |

>  **Note:**  Table based on the tests and parameter settings (as per original readme).

---

## Introduction of Wan2.2

**Wan2.2** offers improvements over Wan2.1, notably in generation quality and model capabilities.  Key innovations include:

*   **Mixture-of-Experts (MoE) Architecture:**  A two-expert design for denoising, increasing model capacity while maintaining efficiency.
*   **Efficient High-Definition Hybrid TI2V:**  A 5B dense model with high compression for fast 720P video generation.

## Comparisons to SOTAs

[Performance Image Placeholder - Insert the actual performance image here.]
>  **Wan2.2** achieves superior performance compared to leading closed-source commercial models (as per original readme).

---

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

The models in this repository are licensed under the Apache 2.0 License.

---

## Acknowledgements

Special thanks to the contributors of [SD3](https://huggingface.co/stabilityai/stable-diffusion-3-medium), [Qwen](https://huggingface.co/Qwen), [umt5-xxl](https://huggingface.co/google/umt5-xxl), [diffusers](https://github.com/huggingface/diffusers), and [HuggingFace](https://huggingface.co).

---

## Contact Us

Join the [Discord](https://discord.gg/AKNgpMK4Yj) or [WeChat groups](https://gw.alicdn.com/imgextra/i2/O1CN01tqjWFi1ByuyehkTSB_!!6000000000015-0-tps-611-1279.jpg)!