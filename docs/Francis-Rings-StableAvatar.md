# StableAvatar: Generate Infinite-Length Avatar Videos from Audio

StableAvatar empowers you to create mesmerizing, infinite-length avatar videos driven solely by audio. Check out the original repository for more details! ([Link to Original Repo](https://github.com/Francis-Rings/StableAvatar))

[![Project Page](https://img.shields.io/badge/Project-Page-Green)](https://francis-rings.github.io/StableAvatar)
[![Arxiv](https://img.shields.io/badge/Paper-Arxiv-red)](https://arxiv.org/abs/2508.08248)
[![HuggingFace Model](https://img.shields.io/badge/HuggingFace-Model-orange)](https://huggingface.co/FrancisRing/StableAvatar/tree/main)
[![Hugging Face Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue)](https://huggingface.co/spaces/YinmingHuang/StableAvatar)
[![YouTube](https://img.shields.io/badge/YouTube-Watch-red?style=flat-square&logo=youtube)](https://www.youtube.com/watch?v=6lhvmbzvv3Y)
[![Bilibili](https://img.shields.io/badge/Bilibili-Watch-blue?style=flat-square&logo=bilibili)](https://www.bilibili.com/video/BV1hUt9z4EoQ)

**Key Features:**

*   **Infinite-Length Video Generation:** Create avatar videos of any length.
*   **ID Preservation:** Maintain the identity of the avatar throughout the video.
*   **High-Fidelity:** Generate high-quality videos without post-processing.
*   **Audio-Driven:** Directly synthesize videos from audio input.

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/d7eca208-6a14-46af-b337-fb4d2b66ba8d" width="320" controls loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/b15784b1-c013-4126-a764-10c844341a4e" width="320" controls loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/87faa3c1-a118-4a03-a071-45f18e87e6a0" width="320" controls loop></video>
     </td>
  </tr>
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/531eb413-8993-4f8f-9804-e3c5ec5794d4" width="320" controls loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/cdc603e2-df46-4cf8-a14e-1575053f996f" width="320" controls loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/7022dc93-f705-46e5-b8fc-3a3fb755795c" width="320" controls loop></video>
     </td>
  </tr>
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/0ba059eb-ff6f-4d94-80e6-f758c613b737" width="320" controls loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/03e6c1df-85c6-448d-b40d-aacb8add4e45" width="320" controls loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/90b78154-dda0-4eaa-91fd-b5485b718a7f" width="320" controls loop></video>
     </td>
  </tr>
</table>

StableAvatar synthesizes infinite-length, ID-preserving videos directly from audio, without the need for face-related post-processing tools.

<p align="center">
  <video src="https://github.com/user-attachments/assets/90691318-311e-40b9-9bd9-62db83ab1492" width="768" autoplay loop muted playsinline></video>
  <br/>
  <span>Comparison highlights StableAvatar's superior performance in delivering infinite-length, high-fidelity, and identity-preserving avatar animations.</span>
</p>

## Overview

<p align="center">
  <img src="assets/figures/framework.jpg" alt="model architecture" width="1280"/>
  </br>
  <i>The overview of the framework of StableAvatar.</i>
</p>

StableAvatar overcomes limitations of existing audio-driven avatar video generation models by utilizing a novel Time-step-aware Audio Adapter and an Audio Native Guidance Mechanism, enabling it to generate long, high-quality videos with natural audio synchronization and identity consistency.

## What's New

*   `[2025-9-8]`:🔥  Check out the brand new demo on [YouTube](https://www.youtube.com/watch?v=GH4hrxIis3Q) and [Bilibili](https://www.bilibili.com/video/BV1jGYPzqEux)!
*   `[2025-8-29]`:🔥  Public demo is now live on [Hugging Face Spaces](https://huggingface.co/spaces/YinmingHuang/StableAvatar) (for Hugging Face Pro users).
*   `[2025-8-18]`:🔥  Runs on [ComfyUI](https://github.com/smthemex/ComfyUI_StableAvatar), 3x faster! Thanks @[smthemex](https://github.com/smthemex).
*   `[2025-8-16]`:🔥  Finetuning and LoRA training codes released!
*   `[2025-8-15]`:🔥  Runs on Gradio Interface. Thanks @[gluttony-10](https://space.bilibili.com/893892)!
*   `[2025-8-15]`:🔥  Runs on [ComfyUI](https://github.com/smthemex/ComfyUI_StableAvatar). Thanks @[smthemex](https://github.com/smthemex).
*   `[2025-8-13]`:🔥  Support for new Blackwell series Nvidia chips (e.g., RTX 6000 Pro).
*   `[2025-8-11]`:🔥  Project page, code, technical report, and basic model checkpoint released.

## 🛠️ To-Do List

*   \[x] StableAvatar-1.3B-basic
*   \[x] Inference Code
*   \[x] Data Pre-Processing Code (Audio Extraction)
*   \[x] Data Pre-Processing Code (Vocal Separation)
*   \[x] Training Code
*   \[x] Full Finetuning Code
*   \[x] Lora Training Code
*   \[x] Lora Finetuning Code
*   \[ ] Inference Code with Audio Native Guidance
*   \[ ] StableAvatar-pro

## 🔑 Quickstart

The basic version of the model (Wan2.1-1.3B-based) supports generating infinite-length videos at resolutions of 480x832, 832x480, or 512x512.

### 🧱 Environment Setup

**For Standard GPUs:**

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
# Optional to install flash_attn to accelerate attention computation
pip install flash_attn
```

**For Blackwell Series GPUs:**

```bash
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
# Optional to install flash_attn to accelerate attention computation
pip install flash_attn
```

### 🧱 Download Weights

If you encounter Hugging Face connection issues, use the mirror endpoint: `export HF_ENDPOINT=https://hf-mirror.com`.

```bash
pip install "huggingface_hub[cli]"
cd StableAvatar
mkdir checkpoints
huggingface-cli download FrancisRing/StableAvatar --local-dir ./checkpoints
```

File structure:

```
StableAvatar/
├── accelerate_config
├── deepspeed_config
├── examples
├── wan
├── checkpoints
│   ├── Kim_Vocal_2.onnx
│   ├── wav2vec2-base-960h
│   ├── Wan2.1-Fun-V1.1-1.3B-InP
│   └── StableAvatar-1.3B
├── inference.py
├── inference.sh
├── train_1B_square.py
├── train_1B_square.sh
├── train_1B_vec_rec.py
├── train_1B_vec_rec.sh
├── audio_extractor.py
├── vocal_seperator.py
├── requirement.txt
```

### 🧱 Audio Extraction

```bash
python audio_extractor.py --video_path="path/test/video.mp4" --saved_audio_path="path/test/audio.wav"
```

### 🧱 Vocal Separation

```bash
pip install audio-separator[gpu]
python vocal_seperator.py --audio_separator_model_file="path/StableAvatar/checkpoints/Kim_Vocal_2.onnx" --audio_file_path="path/test/audio.wav" --saved_vocal_path="path/test/vocal.wav"
```

### 🧱 Base Model Inference

Modify `inference.sh` parameters as needed:

```bash
bash inference.sh
```

Key parameters:

*   `--width` and `--height`: Output resolution (512x512, 480x832, or 832x480).
*   `--output_dir`: Save path for generated animation.
*   `--validation_reference_path`, `--validation_driven_audio_path`, `--validation_prompts`: Paths for reference image, audio, and prompts.
*   `--pretrained_model_name_or_path`, `--pretrained_wav2vec_path`, `--transformer_path`: Paths for pretrained model weights.
*   `--sample_steps`: Inference steps (recommended: 30-50).
*   `--overlap_window_length`: Overlapping context length (recommended: 5-15).
*   `--clip_sample_n_frames`: Frames per batch.
*   `--sample_text_guide_scale`, `--sample_audio_guide_scale`: CFG scales for prompts and audio (recommended: 3-6).

Run a Gradio interface:

```bash
python app.py
```

**Example videos are available in `path/StableAvatar/examples`.**

#### 💡 Tips

*   Use the `transformer3d-square.pt` or `transformer3d-rec-vec.pt` weights depending on your dataset's resolution during training.
*   `--GPU_memory_mode`:  `model_full_load`, `sequential_cpu_offload`, `model_cpu_offload_and_qfloat8`, `model_cpu_offload` (for memory optimization).
*   Multi-GPU inference:  Use `--ulysses_degree` and `--ring_degree` (e.g., for 8 GPUs, set `--ulysses_degree=4` and `--ring_degree=2`).  Also use `--fsdp_dit` for FSDP.
*   To create an MP4 with audio, use `ffmpeg`:
    ```bash
    ffmpeg -i video_without_audio.mp4 -i /path/audio.wav -c:v copy -c:a aac -shortest /path/output_with_audio.mp4
    ```

### 🧱 Model Training

**Training a conditioned Video Diffusion Transformer (DiT) model? This is a helpful guide!**

Dataset structure:

```
talking_face_data/
├── rec
│   │  ├──speech
│   │  │  ├──00001
│   │  │  │  ├──sub_clip.mp4
│   │  │  │  ├──audio.wav
│   │  │  │  ├──images
│   │  │  │  │  ├──frame_0.png
│   │  │  │  │  ├──frame_1.png
│   │  │  │  │  ├──frame_2.png
│   │  │  │  │  ├──...
│   │  │  │  ├──face_masks
│   │  │  │  │  ├──frame_0.png
│   │  │  │  │  ├──frame_1.png
│   │  │  │  │  ├──frame_2.png
│   │  │  │  │  ├──...
│   │  │  │  ├──lip_masks
│   │  │  │  │  ├──frame_0.png
│   │  │  │  │  ├──frame_1.png
│   │  │  │  │  ├──frame_2.png
│   │  │  │  │  ├──...
│   │  │  ├──00002
│   │  │  │  ├──sub_clip.mp4
│   │  │  │  ├──audio.wav
│   │  │  │  ├──images
│   │  │  │  ├──face_masks
│   │  │  │  ├──lip_masks
│   │  │  └──...
│   │  ├──singing
│   │  │  ├──00001
│   │  │  │  ├──sub_clip.mp4
│   │  │  │  ├──audio.wav
│   │  │  │  ├──images
│   │  │  │  ├──face_masks
│   │  │  │  ├──lip_masks
│   │  │  └──...
│   │  ├──dancing
│   │  │  ├──00001
│   │  │  │  ├──sub_clip.mp4
│   │  │  │  ├──audio.wav
│   │  │  │  ├──images
│   │  │  │  ├──face_masks
│   │  │  │  ├──lip_masks
│   │  │  └──...
├── vec
│   │  ├──speech
│   │  │  ├──00001
│   │  │  │  ├──sub_clip.mp4
│   │  │  │  ├──audio.wav
│   │  │  │  ├──images
│   │  │  │  ├──face_masks
│   │  │  │  ├──lip_masks
│   │  │  └──...
│   │  ├──singing
│   │  │  ├──00001
│   │  │  │  ├──sub_clip.mp4
│   │  │  │  ├──audio.wav
│   │  │  │  ├──images
│   │  │  │  ├──face_masks
│   │  │  │  ├──lip_masks
│   │  │  └──...
│   │  ├──dancing
│   │  │  ├──00001
│   │  │  │  ├──sub_clip.mp4
│   │  │  │  ├──audio.wav
│   │  │  │  ├──images
│   │  │  │  ├──face_masks
│   │  │  │  ├──lip_masks
│   │  │  └──...
├── square
│   │  ├──speech
│   │  │  ├──00001
│   │  │  │  ├──sub_clip.mp4
│   │  │  │  ├──audio.wav
│   │  │  │  ├──images
│   │  │  │  ├──face_masks
│   │  │  │  ├──lip_masks
│   │  │  └──...
│   │  ├──singing
│   │  │  ├──00001
│   │  │  │  ├──sub_clip.mp4
│   │  │  │  ├──audio.wav
│   │  │  │  ├──images
│   │  │  │  ├──face_masks
│   │  │  │  ├──lip_masks
│   │  │  └──...
│   │  ├──dancing
│   │  │  ├──00001
│   │  │  │  ├──sub_clip.mp4
│   │  │  │  ├──audio.wav
│   │  │  │  ├──images
│   │  │  │  ├──face_masks
│   │  │  │  ├──lip_masks
│   │  │  └──...
├── video_rec_path.txt
├── video_square_path.txt
└── video_vec_path.txt
```

*   `square`: 512x512 videos
*   `vec`: 480x832 videos
*   `rec`: 832x480 videos
*   Each folder contains: `speech`, `singing`, and `dancing` subfolders.
*   `.png` image naming: `frame_i.png`
*   `sub_clip.mp4` and `audio.wav` correspond to the video frames and audio.
*   `video_square_path.txt`, `video_rec_path.txt`, and `video_vec_path.txt`:  File paths for training.

**Data Preparation:**

1.  Extract frames from raw videos using `ffmpeg`.
    ```bash
    ffmpeg -i raw_video_1.mp4 -q:v 1 -start_number 0 path/StableAvatar/talking_face_data/rec/speech/00001/images/frame_%d.png
    ```
2.  Extract human face masks (refer to [StableAnimator repo](https://github.com/Francis-Rings/StableAnimator)).
3.  Extract lip masks:
    ```bash
    pip install mediapipe
    python lip_mask_extractor.py --folder_root="path/StableAvatar/talking_face_data/rec/singing" --start=1 --end=500
    ```
4.  Extract audio (see Audio Extraction section).

Training commands:

```bash
# Training on a single resolution (512x512) on a single machine
bash train_1B_square.sh
# Training on a single resolution (512x512) on multiple machines
bash train_1B_square_64.sh
# Training on mixed resolutions (480x832 and 832x480) on a single machine
bash train_1B_rec_vec.sh
# Training on mixed resolutions (480x832 and 832x480) on multiple machines
bash train_1B_rec_vec_64.sh
```

Key parameters:

*   Modify `CUDA_VISIBLE_DEVICES` for GPU selection.
*   `--pretrained_model_name_or_path`, `--pretrained_wav2vec_path`, `--output_dir`:  Pretrained weights paths and the output checkpoint path.
*   `--train_data_square_dir`, `--train_data_rec_dir`, `--train_data_vec_dir`: Training data paths.
*   `--validation_reference_path`, `--validation_driven_audio_path`: Validation paths.
*   `--video_sample_n_frames`: Frames per batch.
*   `--num_train_epochs`: Training epochs.

**Training requires approximately 50GB VRAM (mixed-resolution) or 40GB (512x512).**
Ensure static backgrounds and clear audio.

To train Wan2.1-14B-based StableAvatar:

```bash
# Training on mixed resolutions (480x832, 832x480, and 512x512) on multiple machines
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./checkpoints/Wan2.1-I2V-14B-480P
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-720P --local-dir ./checkpoints/Wan2.1-I2V-14B-720P # Optional
bash train_14B.sh
```

Use deepspeed configuration in `path/StableAvatar/deepspeed_config/zero_stage2_config.json` for 14B models.

To run LoRA training:

```bash
# Lora Training StableAvatar-1.3B on a mixed resolution setting (480x832 and 832x480) in a single machine
bash train_1B_rec_vec_lora.sh
# Training StableAvatar-1.3B on a mixed resolution setting (480x832 and 832x480) in multiple machines
bash train_1B_rec_vec_lora_64.sh
# Lora-Training StableAvatar-14B on a mixed resolution setting (480x832, 832x480, and 512x512) in multiple machines
bash train_14B_lora.sh
```
Control LoRA quality by adjusting `--rank` and `--network_alpha`.

### 🧱 Model Finetuning

To finetune or lora finetune:

*   Full Finetuning -  Add `--transformer_path="path/StableAvatar/checkpoints/StableAvatar-1.3B/transformer3d-square.pt"` to training scripts.
*   LoRA finetuning - Add `--transformer_path="path/StableAvatar/checkpoints/StableAvatar-1.3B/transformer3d-square.pt"` to the train_1B_rec_vec_lora.sh

Control LoRA quality by adjusting `--rank` and `--network_alpha`.

### 🧱 VRAM and Runtime

For a 5s video (480x832, fps=25), the basic model (--GPU_memory_mode="model_full_load") requires approximately 18GB VRAM and takes ~3 minutes on a 4090 GPU.

**StableAvatar can theoretically generate hours-long videos.** Consider running the VAE decoder on the CPU if memory is an issue.

## Contact

For suggestions or assistance, contact me: francisshuyuan@gmail.com

**If you find this project useful, please star the repository and cite it:**

```bib
@article{tu2025stableavatar,
  title={StableAvatar: Infinite-Length Audio-Driven Avatar Video Generation},
  author={Tu, Shuyuan and Pan, Yueming and Huang, Yinming and Han, Xintong and Xing, Zhen and Dai, Qi and Luo, Chong and Wu, Zuxuan and Jiang Yu-Gang},
  journal={arXiv preprint arXiv:2508.08248},
  year={2025}
}
```