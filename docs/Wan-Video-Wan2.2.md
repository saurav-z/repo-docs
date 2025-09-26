# Wan2.2: Unleash Cinematic Video Generation with Advanced AI

**Wan2.2** is a cutting-edge open-source video generation model that empowers users to create stunning cinematic videos from text, images, and audio. Visit the original repository [here](https://github.com/Wan-Video/Wan2.2).

<p align="center">
    <img src="assets/logo.png" width="400"/>
<p>

**Key Features:**

*   👍 **Mixture-of-Experts (MoE) Architecture:** Leverages MoE for increased model capacity without added computational cost, featuring specialized experts for improved denoising.
*   👍 **Cinematic Aesthetics:** Incorporates meticulously curated aesthetic data for precise control over lighting, composition, and color, enabling customizable cinematic styles.
*   👍 **Enhanced Motion Generation:** Trained on a significantly larger dataset than previous versions, resulting in superior generalization across various dimensions including motion, semantics, and aesthetics.
*   👍 **Efficient High-Definition TI2V:** Offers a 5B model with advanced compression, supporting 720P video generation at 24fps on consumer-grade GPUs.

**Explore the Power of Wan2.2:**

*   [**Wan: Open and Advanced Large-Scale Video Generative Models**](https://arxiv.org/abs/2503.20314)
*   [**Wan**](https://wan.video)
*   [**Hugging Face**](https://huggingface.co/Wan-AI/)
*   [**ModelScope**](https://modelscope.cn/organization/Wan-AI)

---

## 🚀 What's New?

Stay up-to-date with the latest developments in Wan2.2:

*   **Wan2.2-Animate-14B:**  A unified model for character animation and replacement ([Model Weights]([https://humanaigc.github.io/wan-animate]), [Inference Code]([https://humanaigc.github.io/wan-animate])).  Try it out on [wan.video](https://wan.video/), [ModelScope Studio](https://www.modelscope.cn/studios/Wan-AI/Wan2.2-Animate) or [HuggingFace Space](https://huggingface.co/spaces/Wan-AI/Wan2.2-Animate)!
*   **Wan2.2-S2V-14B:** Audio-driven cinematic video generation ([Inference Code](#run-speech-to-video-generation), [Model Weights](#model-download), [Technical Report](https://humanaigc.github.io/wan-s2v-webpage/content/wan-s2v.pdf)). Explore it on [wan.video](https://wan.video/), [ModelScope Gradio](https://www.modelscope.cn/studios/Wan-AI/Wan2.2-S2V) or [HuggingFace Gradio](https://huggingface.co/spaces/Wan-AI/Wan2.2-S2V)!
*   [HF space](https://huggingface.co/spaces/Wan-AI/Wan-2.2-5B)
*   ComfyUI integration ([CN](https://docs.comfy.org/zh-CN/tutorials/video/wan/wan2_2) | [EN](https://docs.comfy.org/tutorials/video/wan/wan2_2))
*   Diffusers integration ([T2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) | [I2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers) | [TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers))
*   Wan2.2 released ([Inference Code](#run-text-to-video-generation) & [Model Weights](#model-download))
*   Text-to-speech synthesis support with [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) for Speech-to-Video generation

## 🎞️ Video Demos

<!-- Replace with a short, compelling video showcasing the capabilities -->
<div align="center">
  <video src="https://github.com/user-attachments/assets/b63bfa58-d5d7-4de6-a1a2-98970b06d9a7" width="70%" poster=""> </video>
</div>

## 🤝 Community Contributions

We encourage and welcome community contributions. Here are some projects built on top of Wan2.2 or its predecessor:

*   [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)
*   [Kijai's ComfyUI WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper)
*   [Cache-dit](https://github.com/vipshop/cache-dit)
*   [FastVideo](https://github.com/hao-ai-lab/FastVideo)

## 📝 Todo List

*   Wan2.2 Text-to-Video
    *   \[x] Multi-GPU Inference code of the A14B and 14B models
    *   \[x] Checkpoints of the A14B and 14B models
    *   \[x] ComfyUI integration
    *   \[x] Diffusers integration
*   Wan2.2 Image-to-Video
    *   \[x] Multi-GPU Inference code of the A14B model
    *   \[x] Checkpoints of the A14B model
    *   \[x] ComfyUI integration
    *   \[x] Diffusers integration
*   Wan2.2 Text-Image-to-Video
    *   \[x] Multi-GPU Inference code of the 5B model
    *   \[x] Checkpoints of the 5B model
    *   \[x] ComfyUI integration
    *   \[x] Diffusers integration
*   Wan2.2-S2V Speech-to-Video
    *   \[x] Inference code of Wan2.2-S2V
    *   \[x] Checkpoints of Wan2.2-S2V-14B
    *   \[x] ComfyUI integration
    *   \[x] Diffusers integration
*   Wan2.2-Animate Character Animation and Replacement
    *   \[x] Inference code of Wan2.2-Animate
    *   \[x] Checkpoints of Wan2.2-Animate
    *   \[x] ComfyUI integration
    *   \[ ] Diffusers integration

## 💻 Run Wan2.2

### 1. Installation

```bash
git clone https://github.com/Wan-Video/Wan2.2.git
cd Wan2.2
```

```bash
# Ensure torch >= 2.4.0
# If the installation of `flash_attn` fails, try installing the other packages first and install `flash_attn` last
pip install -r requirements.txt
# If you want to use CosyVoice to synthesize speech for Speech-to-Video Generation, please install requirements_s2v.txt additionally
pip install -r requirements_s2v.txt
```

### 2. Model Download

| Models          | Download Links                                                                                                                              | Description |
|----------------|---------------------------------------------------------------------------------------------------------------------------------------------|-------------|
| T2V-A14B      | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B)    🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-T2V-A14B)    | Text-to-Video MoE model, supports 480P & 720P |
| I2V-A14B      | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B)    🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-I2V-A14B)    | Image-to-Video MoE model, supports 480P & 720P |
| TI2V-5B       | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B)     🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-TI2V-5B)     | High-compression VAE, T2V+I2V, supports 720P |
| S2V-14B       | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-S2V-14B)     🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-S2V-14B)     | Speech-to-Video model, supports 480P & 720P |
| Animate-14B   | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-Animate-14B) 🤖 [ModelScope](https://www.modelscope.cn/models/Wan-AI/Wan2.2-Animate-14B)  | Character animation and replacement | |

> **Note:** The TI2V-5B model generates 720P videos at **24 FPS**.

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

### 3. Run Text-to-Video Generation

This repository supports the `Wan2.2-T2V-A14B` Text-to-Video model and can simultaneously support video generation at 480P and 720P resolutions.

#### (1) Without Prompt Extension

-   Single-GPU inference

```bash
python generate.py  --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --offload_model True --convert_model_dtype --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
```

>   **Note:** This command needs a GPU with at least 80GB VRAM.
>
>   **Tip:** Use `--offload_model True`, `--convert_model_dtype` and `--t5_cpu` to reduce GPU memory if you encounter OOM (Out-of-Memory) issues.

-   Multi-GPU inference using FSDP + DeepSpeed Ulysses

```bash
torchrun --nproc_per_node=8 generate.py --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
```

#### (2) Using Prompt Extension

Extending prompts to enrich details:

-   Using Dashscope API for extension:

    *   Get a `dashscope.api_key`  ([EN](https://www.alibabacloud.com/help/en/model-studio/getting-started/first-api-call-to-qwen) | [CN](https://help.aliyun.com/zh/model-studio/getting-started/first-api-call-to-qwen))
    *   Set environment variable `DASH_API_KEY` and  `DASH_API_URL` to 'https://dashscope-intl.aliyuncs.com/api/v1'.
    *   Use `qwen-plus` for text-to-video or `qwen-vl-max` for image-to-video tasks.
    *   Modify the extension model with the `--prompt_extend_model` flag:

```bash
DASH_API_KEY=your_key torchrun --nproc_per_node=8 generate.py  --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage" --use_prompt_extend --prompt_extend_method 'dashscope' --prompt_extend_target_lang 'zh'
```

-   Using a local model for extension:

    *   Use Qwen models from Hugging Face (e.g., `Qwen/Qwen2.5-14B-Instruct`, `Qwen/Qwen2.5-7B-Instruct`, or `Qwen/Qwen2.5-3B-Instruct` for text-to-video, and  `Qwen/Qwen2.5-VL-7B-Instruct` or  `Qwen/Qwen2.5-VL-3B-Instruct` for image-to-video).
    *   Larger models yield better results but need more GPU memory.
    *   Specify local or Hugging Face model with `--prompt_extend_model`:

```bash
torchrun --nproc_per_node=8 generate.py  --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage" --use_prompt_extend --prompt_extend_method 'local_qwen' --prompt_extend_target_lang 'zh'
```

### 4. Run Image-to-Video Generation

This repository supports the `Wan2.2-I2V-A14B` Image-to-Video model, supporting 480P and 720P.

-   Single-GPU inference

```bash
python generate.py --task i2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-I2V-A14B --offload_model True --convert_model_dtype --image examples/i2v_input.JPG --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
```

>   **Note:** Requires a GPU with at least 80GB VRAM.
>
>   **Tip:** Image aspect ratio dictates video size.

-   Multi-GPU inference using FSDP + DeepSpeed Ulysses

```bash
torchrun --nproc_per_node=8 generate.py --task i2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-I2V-A14B --image examples/i2v_input.JPG --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
```

-   Image-to-Video Generation without prompt

```bash
DASH_API_KEY=your_key torchrun --nproc_per_node=8 generate.py --task i2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-I2V-A14B --prompt '' --image examples/i2v_input.JPG --dit_fsdp --t5_fsdp --ulysses_size 8 --use_prompt_extend --prompt_extend_method 'dashscope'
```

> **Note:**  Videos can be generated solely from the input image. You can use prompt extension to generate prompts from the image.

### 5. Run Text-Image-to-Video Generation

This repository supports the `Wan2.2-TI2V-5B` Text-Image-to-Video model and can support video generation at 720P resolutions.

-   Single-GPU Text-to-Video inference

```bash
python generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ./Wan2.2-TI2V-5B --offload_model True --convert_model_dtype --t5_cpu --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage"
```

>   **Note:** The 720P resolution is `1280*704` or `704*1280`.
>
>   Requires a GPU with at least 24GB VRAM (e.g., RTX 4090).
>
>   Omit `--offload_model True`, `--convert_model_dtype` and `--t5_cpu` for GPUs with more VRAM.

-   Single-GPU Image-to-Video inference

```bash
python generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ./Wan2.2-TI2V-5B --offload_model True --convert_model_dtype --t5_cpu --image examples/i2v_input.JPG --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
```

>   **Note:**  Image-to-Video if the `--image` parameter is used.
>
>   The video size follows the input image aspect ratio.

-   Multi-GPU inference using FSDP + DeepSpeed Ulysses

```bash
torchrun --nproc_per_node=8 generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ./Wan2.2-TI2V-5B --dit_fsdp --t5_fsdp --ulysses_size 8 --image examples/i2v_input.JPG --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
```

### 6. Run Speech-to-Video Generation

This repository supports the `Wan2.2-S2V-14B` Speech-to-Video model, offering 480P and 720P resolutions.

-   Single-GPU Speech-to-Video inference

```bash
python generate.py  --task s2v-14B --size 1024*704 --ckpt_dir ./Wan2.2-S2V-14B/ --offload_model True --convert_model_dtype --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard."  --image "examples/i2v_input.JPG" --audio "examples/talk.wav"
# Adjust video length based on audio input.

# Use CosyVoice for speech generation:
python generate.py  --task s2v-14B --size 1024*704 --ckpt_dir ./Wan2.2-S2V-14B/ --offload_model True --convert_model_dtype --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard."  --image "examples/i2v_input.JPG" --enable_tts --tts_prompt_audio "examples/zero_shot_prompt.wav" --tts_prompt_text "希望你以后能够做的比我还好呦。" --tts_text "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。"
```

>   **Note:** Requires a GPU with at least 80GB VRAM.

-   Multi-GPU inference using FSDP + DeepSpeed Ulysses

```bash
torchrun --nproc_per_node=8 generate.py --task s2v-14B --size 1024*704 --ckpt_dir ./Wan2.2-S2V-14B/ --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard." --image "examples/i2v_input.JPG" --audio "examples/talk.wav"
```

-   Pose + Audio Driven Generation

```bash
torchrun --nproc_per_node=8 generate.py --task s2v-14B --size 1024*704 --ckpt_dir ./Wan2.2-S2V-14B/ --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "a person is singing" --image "examples/pose.png" --audio "examples/sing.MP3" --pose_video "./examples/pose.mp4"
```

>   **Note:**  Video size follows the input image aspect ratio for the S2V task.
>
>   Uses audio, an image, and an optional text prompt.
>
>   The `--pose_video` parameter allows pose-driven generation.
>
>   The `--num_clip` parameter helps for quicker preview.
>
>   Visit the project page for examples.

### 7. Run Wan-Animate

Wan-Animate generates videos based on a character image and a video. You can choose either "animation" or "replacement" mode.

1.  **Animation Mode:** The character image mimics the human motion in the input video.
2.  **Replacement Mode:** The model replaces the character image with the input video.

Visit the [project page](https://humanaigc.github.io/wan-animate) for more information.

#### (1) Preprocessing

Prepare the video input: ([UserGuider](https://github.com/Wan-Video/Wan2.2/blob/main/wan/modules/animate/preprocess/UserGuider.md))

*   **Animation:**

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

*   **Replacement:**

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

#### (2) Run in animation mode

*   Single-GPU inference

```bash
python generate.py --task animate-14B --ckpt_dir ./Wan2.2-Animate-14B/ --src_root_path ./examples/wan_animate/animate/process_results/ --refert_num 1
```

*   Multi-GPU inference using FSDP + DeepSpeed Ulysses

```bash
python -m torch.distributed.run --nnodes 1 --nproc_per_node 8 generate.py --task animate-14B --ckpt_dir ./Wan2.2-Animate-14B/ --src_root_path ./examples/wan_animate/animate/process_results/ --refert_num 1 --dit_fsdp --t5_fsdp --ulysses_size 8
```

#### (3) Run in replacement mode

*   Single-GPU inference

```bash
python generate.py --task animate-14B --ckpt_dir ./Wan2.2-Animate-14B/ --src_root_path ./examples/wan_animate/replace/process_results/ --refert_num 1 --replace_flag --use_relighting_lora
```

*   Multi-GPU inference using FSDP + DeepSpeed Ulysses

```bash
python -m torch.distributed.run --nnodes 1 --nproc_per_node 8 generate.py --task animate-14B --ckpt_dir ./Wan2.2-Animate-14B/ --src_root_path ./examples/wan_animate/replace/process_results/src_pose.mp4  --refert_num 1 --replace_flag --use_relighting_lora --dit_fsdp --t5_fsdp --ulysses_size 8
```

> **Important:** Avoid using LoRA models trained on `Wan2.2` with **Wan-Animate**.

## 📊 Computational Efficiency

Comparative performance across different GPUs:

<!-- Replace with an image (or table) of the comp_effic.png content -->

<div align="center">
    <img src="assets/comp_effic.png" alt="" style="width: 80%;" />
</div>

>   **Note:** See the table in the original README for the exact settings.

## 🧠 Introduction of Wan2.2

Wan2.2 offers improvements in quality and capability based on Wan2.1. Key innovations include:

#### (1) Mixture-of-Experts (MoE) Architecture

MoE increases model capacity efficiently. In Wan2.2, the A14B model uses a two-expert design tailored to the denoising process of diffusion models: a high-noise expert and a low-noise expert.

<div align="center">
    <img src="assets/moe_arch.png" alt="" style="width: 90%;" />
</div>

<div align="center">
    <img src="assets/moe_2.png" alt="" style="width: 90%;" />
</div>

#### (2) Efficient High-Definition Hybrid TI2V

The TI2V-5B model uses high-compression to improve deployment. The compression rate of Wan2.2-VAE reaches $4\times16\times16$, increasing the overall compression rate to 64.

<div align="center">
    <img src="assets/vae.png" alt="" style="width: 80%;" />
</div>

#### (3) Comparisons to SOTAs
We compared Wan2.2 with leading closed-source commercial models on our new Wan-Bench 2.0, evaluating performance across multiple crucial dimensions. The results demonstrate that Wan2.2 achieves superior performance compared to these leading models.

<div align="center">
    <img src="assets/performance.png" alt="" style="width: 90%;" />
</div>

## 📚 Citation

```
@article{wan2025,
      title={Wan: Open and Advanced Large-Scale Video Generative Models},
      author={Team Wan and Ang Wang and Baole Ai and Bin Wen and Chaojie Mao and Chen-Wei Xie and Di Chen and Feiwu Yu and Haiming Zhao and Jianxiao Yang and Jianyuan Zeng and Jiayu Wang and Jingfeng Zhang and Jingren Zhou and Jinkai Wang and Jixuan Chen and Kai Zhu and Kang Zhao and Keyu Yan and Lianghua Huang and Mengyang Feng and Ningyi Zhang and Pandeng Li and Pingyu Wu and Ruihang Chu and Ruili Feng and Shiwei Zhang and Siyang Sun and Tao Fang and Tianxing Wang and Tianyi Gui and Tingyu Weng and Tong Shen and Wei Lin and Wei Wang and Wei Wang and Wenmeng Zhou and Wente Wang and Wenting Shen and Wenyuan Yu and Xianzhong Shi and Xiaoming Huang and Xin Xu and Yan Kou and Yangyu Lv and Yifei Li and Yijing Liu and Yiming Wang and Yingya Zhang and Yitong Huang and Yong Li and You Wu and Yu Liu and Yulin Pan and Yun Zheng and Yuntao Hong and Yupeng Shi and Yutong Feng and Zeyinzi Jiang and Zhen Han and Zhi-Fan Wu and Ziyu Liu},
      journal = {arXiv preprint arXiv:2503.20314},
      year={2025}
}
```

## 📜 License

Licensed under the Apache 2.0 License. You are responsible for your usage and must adhere to the license terms, which are detailed in [LICENSE.txt](LICENSE.txt).

## 🙌 Acknowledgements

Thank you to the contributors of [SD3](https://huggingface.co/stabilityai/stable-diffusion-3-medium), [Qwen](https://huggingface.co/Qwen), [umt5-xxl](https://huggingface.co/google/umt5-xxl), [diffusers](https://github.com/huggingface/diffusers) and [HuggingFace](https://huggingface.co).

## 💬 Contact

Join our [Discord](https://discord.gg/AKNgpMK4Yj) or [WeChat groups](https://gw.alicdn.com/imgextra/i2/O1CN01tqjWFi1ByuyehkTSB_!!6000000000015-0-tps-611-1279.jpg) for support and discussions.