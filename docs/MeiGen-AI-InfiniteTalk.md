<div align="center">

<p align="center">
  <img src="assets/logo2.jpg" alt="InfinteTalk" width="440"/>
</p>

<h1>InfiniteTalk: Audio-driven Video Generation for Sparse-Frame Video Dubbing</h1>

</div>

> **Generate stunning, unlimited-length talking videos with InfiniteTalk, an innovative audio-driven video generation model!**

[Original Repository](https://github.com/MeiGen-AI/InfiniteTalk)

InfiniteTalk, developed by [Shaoshu Yang*](https://scholar.google.com/citations?user=JrdZbTsAAAAJ&hl=en) and collaborators, is a cutting-edge framework for generating talking videos from audio, enabling both video-to-video and image-to-video applications.

<p align="center">
  <img src="assets/pipeline.png">
</p>

## Key Features

*   💬 **Sparse-Frame Video Dubbing:**  Synchronizes not only lip movements, but also head movements, body posture, and facial expressions with the provided audio.
*   ⏱️ **Infinite-Length Generation:** Supports the creation of videos with unlimited duration.
*   ✨ **Improved Stability:** Reduces distortions in hands and body compared to MultiTalk, leading to more realistic results.
*   🚀 **Superior Lip Accuracy:**  Delivers significantly more accurate lip synchronization compared to previous methods.
*   🖼️ **Image-to-Video Capabilities:** Transform a single image and audio into a dynamic talking video.

## What's New

*   **August 19, 2025:** Release of the [Technique Report](https://arxiv.org/abs/2508.14033), along with model weights, and the source code. Includes Gradio and [ComfyUI](https://github.com/MeiGen-AI/InfiniteTalk/tree/comfyui) implementations.

## Community Works

*   **Wan2GP:** Integration of InfiniteTalk by [deepbeepmeep](https://github.com/deepbeepmeep)  optimized for low VRAM usage and a variety of video editing features.
*   **ComfyUI:**  [kijai](https://github.com/kijai) has provided support for InfiniteTalk within the ComfyUI framework.

## Getting Started

### 🛠️ Installation

**1. Set up a Conda Environment**
```bash
conda create -n multitalk python=3.10
conda activate multitalk
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install -U xformers==0.0.28 --index-url https://download.pytorch.org/whl/cu121
```

**2. Flash-Attention Installation**
```bash
pip install misaki[en]
pip install ninja 
pip install psutil 
pip install packaging
pip install wheel
pip install flash_attn==2.7.4.post1
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt
conda install -c conda-forge librosa
```

**4. FFmpeg Installation**
```bash
conda install -c conda-forge ffmpeg
```
or
```bash
sudo yum install ffmpeg ffmpeg-devel
```

### 🧱 Model Preparation

**1. Download Models**

*   **Wan2.1-I2V-14B-480P:** Base Model ([Hugging Face](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P))
*   **chinese-wav2vec2-base:** Audio Encoder ([Hugging Face](https://huggingface.co/TencentGameMate/chinese-wav2vec2-base))
*   **MeiGen-InfiniteTalk:** Audio Condition Weights ([Hugging Face](https://huggingface.co/MeiGen-AI/InfiniteTalk))

Download models using Hugging Face CLI:
```bash
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./weights/Wan2.1-I2V-14B-480P
huggingface-cli download TencentGameMate/chinese-wav2vec2-base --local-dir ./weights/chinese-wav2vec2-base
huggingface-cli download TencentGameMate/chinese-wav2vec2-base model.safetensors --revision refs/pr/1 --local-dir ./weights/chinese-wav2vec2-base
huggingface-cli download MeiGen-AI/InfiniteTalk --local-dir ./weights/InfiniteTalk
```

### 🔑 Quick Inference

**Usage Guidelines:**

*   Lip Sync Accuracy: Adjust `Audio CFG` (3-5 is optimal). Higher values improve synchronization.
*   FusionX: LoRA for faster inference, but can cause color shifts and reduce identity preservation over long videos.
*   V2V Generation: Enables unlimited length, mimicking original camera movement. Consider SDEdit for improved accuracy in camera movement (but can introduce color shifts, best for short clips).
*   I2V Generation: Generates good results from single images up to 1 minute. Beyond 1 minute, color shifts may appear. A script is available to convert the single image into a video format [here](https://github.com/MeiGen-AI/InfiniteTalk/blob/main/tools/convert_img_to_video.py).
*   Quantization Model: Use quantization to minimize memory usage if you encounter memory issues during inference.

**Inference Examples**

**1) Single GPU Inference**

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

**2) 720P Inference**

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

**3) Low VRAM Inference**

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

**4) Multi-GPU Inference**

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

**5) Multi-Person Animation**

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

**6) Inference with FusioniX or Lightx2v (Requires only 4-8 Steps)**

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

**7) Inference with Quantization Model (Single GPU Only)**

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

**8) Run with Gradio**

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

## 📚 Citation

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

## 📜 License

The models in this repository are licensed under the Apache 2.0 License.  You are responsible for your use of the models and must not share any content that violates applicable laws.