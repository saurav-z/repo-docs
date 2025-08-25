<div align="center">
  <img src="assets/logo2.jpg" alt="InfiniteTalk" width="440"/>
</div>

# InfiniteTalk: Generate Unlimited-Length Talking Videos from Audio

InfiniteTalk empowers you to create stunning, high-quality talking videos driven by audio, offering seamless lip synchronization and expressive animations.  Learn more and explore the possibilities at the [original repository](https://github.com/MeiGen-AI/InfiniteTalk).

<br>

[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://meigen-ai.github.io/InfiniteTalk/)
[![Technique Report](https://img.shields.io/badge/Technique-Report-red)](https://arxiv.org/abs/2508.14033)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/MeiGen-AI/InfiniteTalk)

<br>

> **InfiniteTalk** revolutionizes video generation by producing unlimited-length talking videos, accurately synchronizing lip movements, head gestures, body postures, and facial expressions with provided audio.

<p align="center">
  <img src="assets/pipeline.png">
</p>

## Key Features

*   💬 **Sparse-Frame Video Dubbing:** Precisely syncs lips, head, body, and facial expressions with audio.
*   ⏱️ **Infinite-Length Generation:** Create videos of any duration.
*   ✨ **Enhanced Stability:** Reduces distortions compared to other methods.
*   🚀 **Superior Lip Accuracy:** Delivers exceptional lip synchronization.
*   🖼️ **Image-to-Video Generation:** Transform images into talking videos with audio input.

## Latest Updates

*   **August 19, 2025:** Released the [Technique Report](https://arxiv.org/abs/2508.14033), along with model weights and code, including Gradio and ComfyUI (branch).
*   **August 19, 2025:** Launched the dedicated [project page](https://meigen-ai.github.io/InfiniteTalk/).

## Community Contributions

*   [Wan2GP](https://github.com/deepbeepmeep/Wan2GP/): Integration optimized for low VRAM and extensive video editing options (MMaudio, Qwen Image Edit, etc.) by [deepbeepmeep](https://github.com/deepbeepmeep).
*   [ComfyUI](https://github.com/kijai/ComfyUI-WanVideoWrapper): ComfyUI support provided by [kijai](https://github.com/kijai).

## Getting Started

### Prerequisites

1.  **Create a Conda Environment:**
    ```bash
    conda create -n multitalk python=3.10
    conda activate multitalk
    ```
2.  **Install PyTorch & xformers:**
    ```bash
    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
    pip install -U xformers==0.0.28 --index-url https://download.pytorch.org/whl/cu121
    ```
3.  **Install Flash-Attention:**
    ```bash
    pip install misaki[en]
    pip install ninja
    pip install psutil
    pip install packaging
    pip install wheel
    pip install flash_attn==2.7.4.post1
    ```
4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    conda install -c conda-forge librosa
    ```
5.  **Install FFmpeg:**
    ```bash
    conda install -c conda-forge ffmpeg
    ```
    or
    ```bash
    sudo yum install ffmpeg ffmpeg-devel
    ```

### Model Preparation

1.  **Download Models:**

    | Model                                  | Download Link                                                              | Notes                 |
    | -------------------------------------- | -------------------------------------------------------------------------- | --------------------- |
    | Wan2.1-I2V-14B-480P                    | [Huggingface](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P)            | Base Model            |
    | chinese-wav2vec2-base                  | [Huggingface](https://huggingface.co/TencentGameMate/chinese-wav2vec2-base) | Audio Encoder         |
    | MeiGen-InfiniteTalk                   | [Huggingface](https://huggingface.co/MeiGen-AI/InfiniteTalk)                | Audio Condition Weights |

    Download models using huggingface-cli:

    ```bash
    huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./weights/Wan2.1-I2V-14B-480P
    huggingface-cli download TencentGameMate/chinese-wav2vec2-base --local-dir ./weights/chinese-wav2vec2-base
    huggingface-cli download TencentGameMate/chinese-wav2vec2-base model.safetensors --revision refs/pr/1 --local-dir ./weights/chinese-wav2vec2-base
    huggingface-cli download MeiGen-AI/InfiniteTalk --local-dir ./weights/InfiniteTalk
    ```

### Quick Inference

The model supports both 480P and 720P resolutions.

> **Tips:**
>
> *   **Lip Synchronization:**  Adjust `Audio CFG` (3-5 optimal) for better sync.
> *   **FusionX:**  Faster inference, but can introduce color shifts and reduce identity preservation over longer durations.
> *   **V2V Generation:** Unlimited length, camera movement mimics the original video. SDEdit improves camera movement accuracy but can introduce color shifts.
> *   **I2V Generation:** Good results up to 1 minute from a single image.  Copying the image to a video by translating or zooming in the image (see the script for [convert image to video](https://github.com/MeiGen-AI/InfiniteTalk/blob/main/tools/convert_img_to_video.py)) can improve generation quality beyond 1 min.
> *   **Quantization:**  Use the quantization model for reduced memory usage.

#### 1. Inference

##### 1) Single GPU
```bash
python generate_infinitetalk.py \
    --ckpt_dir weights/Wan2.1-I2V-14B-480P \
    --wav2vec_dir 'weights/chinese-wav2vec2-base' \
    --infinitetalk_dir weights/InfiniteTalk/single/infinitetalk.safetensors \
    --input_json examples/single_example_image.json \
    --size infinitetalk-480 \
    --sample_steps 40 \
    --mode streaming \
    --motion_frame 9 \
    --save_file infinitetalk_res
```

##### 2) 720P
```bash
python generate_infinitetalk.py \
    --ckpt_dir weights/Wan2.1-I2V-14B-480P \
    --wav2vec_dir 'weights/chinese-wav2vec2-base' \
    --infinitetalk_dir weights/InfiniteTalk/single/infinitetalk.safetensors \
    --input_json examples/single_example_image.json \
    --size infinitetalk-720 \
    --sample_steps 40 \
    --mode streaming \
    --motion_frame 9 \
    --save_file infinitetalk_res_720p
```

##### 3) Low VRAM
```bash
python generate_infinitetalk.py \
    --ckpt_dir weights/Wan2.1-I2V-14B-480P \
    --wav2vec_dir 'weights/chinese-wav2vec2-base' \
    --infinitetalk_dir weights/InfiniteTalk/single/infinitetalk.safetensors \
    --input_json examples/single_example_image.json \
    --size infinitetalk-480 \
    --sample_steps 40 \
    --num_persistent_param_in_dit 0 \
    --mode streaming \
    --motion_frame 9 \
    --save_file infinitetalk_res_lowvram
```

##### 4) Multi-GPU
```bash
GPU_NUM=8
torchrun --nproc_per_node=$GPU_NUM --standalone generate_infinitetalk.py \
    --ckpt_dir weights/Wan2.1-I2V-14B-480P \
    --wav2vec_dir 'weights/chinese-wav2vec2-base' \
    --infinitetalk_dir weights/InfiniteTalk/single/infinitetalk.safetensors \
    --dit_fsdp --t5_fsdp \
    --ulysses_size=$GPU_NUM \
    --input_json examples/single_example_image.json \
    --size infinitetalk-480 \
    --sample_steps 40 \
    --mode streaming \
    --motion_frame 9 \
    --save_file infinitetalk_res_multigpu
```

##### 5) Multi-Person animation
```bash
python generate_infinitetalk.py \
    --ckpt_dir weights/Wan2.1-I2V-14B-480P \
    --wav2vec_dir 'weights/chinese-wav2vec2-base' \
    --infinitetalk_dir weights/InfiniteTalk/multi/infinitetalk.safetensors \
    --input_json examples/multi_example_image.json \
    --size infinitetalk-480 \
    --sample_steps 40 \
    --num_persistent_param_in_dit 0 \
    --mode streaming \
    --motion_frame 9 \
    --save_file infinitetalk_res_multiperson
```

#### 2. FusioniX or Lightx2v
[FusioniX](https://huggingface.co/vrgamedevgirl84/Wan14BT2VFusioniX/blob/main/FusionX_LoRa/Wan2.1_I2V_14B_FusionX_LoRA.safetensors) requires 8 steps and [lightx2v](https://huggingface.co/Kijai/WanVideo_comfy/blob/main/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors) requires only 4 steps.

```bash
python generate_infinitetalk.py \
    --ckpt_dir weights/Wan2.1-I2V-14B-480P \
    --wav2vec_dir 'weights/chinese-wav2vec2-base' \
    --infinitetalk_dir weights/InfiniteTalk/single/infinitetalk.safetensors \
    --lora_dir weights/Wan2.1_I2V_14B_FusionX_LoRA.safetensors \
    --input_json examples/single_example_image.json \
    --lora_scale 1.0 \
    --size infinitetalk-480 \
    --sample_text_guide_scale 1.0 \
    --sample_audio_guide_scale 2.0 \
    --sample_steps 8 \
    --mode streaming \
    --motion_frame 9 \
    --sample_shift 2 \
    --num_persistent_param_in_dit 0 \
    --save_file infinitetalk_res_lora
```

#### 3. Quantization Model (Single GPU)

```bash
python generate_infinitetalk.py \
    --ckpt_dir weights/Wan2.1-I2V-14B-480P \
    --wav2vec_dir 'weights/chinese-wav2vec2-base' \
    --infinitetalk_dir weights/InfiniteTalk/single/infinitetalk.safetensors \
    --input_json examples/single_example_image.json \
    --size infinitetalk-480 \
    --sample_steps 40 \
    --mode streaming \
    --quant fp8 \
    --quant_dir weights/InfiniteTalk/quant_models/infinitetalk_single_fp8.safetensors \
    --motion_frame 9 \
    --num_persistent_param_in_dit 0 \
    --save_file infinitetalk_res_quant
```

#### 4. Gradio Demo

```bash
python app.py \
    --ckpt_dir weights/Wan2.1-I2V-14B-480P \
    --wav2vec_dir 'weights/chinese-wav2vec2-base' \
    --infinitetalk_dir weights/InfiniteTalk/single/infinitetalk.safetensors \
    --num_persistent_param_in_dit 0 \
    --motion_frame 9
```

or

```bash
python app.py \
    --ckpt_dir weights/Wan2.1-I2V-14B-480P \
    --wav2vec_dir 'weights/chinese-wav2vec2-base' \
    --infinitetalk_dir weights/InfiniteTalk/multi/infinitetalk.safetensors \
    --num_persistent_param_in_dit 0 \
    --motion_frame 9
```

## Citation

```
@misc{yang2025infinitetalkaudiodrivenvideogeneration,
      title={InfiniteTalk: Audio-driven Video Generation for Sparse-Frame Video Dubbing},
      author={Shaoshu Yang and Zhe Kong and Feng Gao and Meng Cheng and Xiangyu Liu and Yong Zhang and Zhuoliang Kang and Wenhan Luo and Xunliang Cai and Ran He and Xiaoming Wei},
      year={2025},
      eprint={2508.14033},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2508.14033},
}
```

## License

The models in this repository are licensed under the Apache 2.0 License.  You are free to use and modify the generated content under the terms of this license, provided you adhere to its stipulations, which include responsible use and the avoidance of harmful or illegal content.