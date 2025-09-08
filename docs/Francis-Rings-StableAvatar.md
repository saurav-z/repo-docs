# StableAvatar: Generate Infinite-Length Audio-Driven Avatar Videos

[![Project Page](https://img.shields.io/badge/Project-Page-Green)](https://francis-rings.github.io/StableAvatar)
[![Arxiv](https://img.shields.io/badge/Paper-Arxiv-red)](https://arxiv.org/abs/2508.08248)
[![HuggingFace Model](https://img.shields.io/badge/HuggingFace-Model-orange)](https://huggingface.co/FrancisRing/StableAvatar/tree/main)
[![Hugging Face Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue)](https://huggingface.co/spaces/YinmingHuang/StableAvatar)
[![YouTube](https://img.shields.io/badge/YouTube-Watch-red?style=flat-square&logo=youtube)](https://www.youtube.com/watch?v=6lhvmbzvv3Y)
[![Bilibili](https://img.shields.io/badge/Bilibili-Watch-blue?style=flat-square&logo=bilibili)](https://www.bilibili.com/video/BV1hUt9z4EoQ)

**StableAvatar generates high-fidelity, identity-preserving, infinite-length avatar videos directly from audio input.**

**Key Features:**

*   **Infinite-Length Video Generation:** Synthesize videos of any length without compromising quality or identity consistency.
*   **Audio-Driven Animation:** Animate avatars based on audio input, ensuring realistic lip-sync and expression.
*   **Identity Preservation:** Maintain the original identity of the avatar throughout the entire video.
*   **End-to-End Synthesis:** Generate videos directly, without the need for face-swapping or post-processing tools.
*   **High-Fidelity Output:** Produce videos with superior quality and natural-looking animations.
*   **Flexible Resolution:** Supports multiple output resolutions including 512x512, 480x832, and 832x480.
*   **Customization:** Fine-tuning and LoRA training options for model customization.

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/d7eca208-6a14-46af-b337-fb4d2b66ba8d" width="320" controls loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/b15784b1-c013-4126-a764-10c844341a4e" width="320" controls loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/87faa5c1-a118-4a03-a071-45f18e87e6a0" width="320" controls loop></video>
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

<p style="text-align: justify;">
  <span>StableAvatar's ability to create <b>infinite-length</b> and <b>ID-preserving videos</b>. All videos are <b>directly synthesized by StableAvatar without the use of any face-related post-processing tools</b>.</span>
</p>

<p align="center">
  <video src="https://github.com/user-attachments/assets/90691318-311e-40b9-9bd9-62db83ab1492" width="768" autoplay loop muted playsinline></video>
  <br/>
  <span>Comparison results between StableAvatar and state-of-the-art (SOTA) audio-driven avatar video generation models highlight the superior performance of StableAvatar in delivering <b>infinite-length, high-fidelity, identity-preserving avatar animation</b>.</span>
</p>

## Overview

<p align="center">
  <img src="assets/figures/framework.jpg" alt="model architecture" width="1280"/>
  </br>
  <i>The overview of the framework of StableAvatar.</i>
</p>

StableAvatar overcomes the limitations of current models by synthesizing long videos with natural audio synchronization and identity consistency. It's the first end-to-end video diffusion transformer for infinite-length video generation, enabling high-quality results without post-processing.  The framework includes a Time-step-aware Audio Adapter and an Audio Native Guidance Mechanism to refine audio synchronization. A Dynamic Weighted Sliding-window Strategy further ensures smooth transitions.  [Visit the original repo](https://github.com/Francis-Rings/StableAvatar) for more information.

## News

*   `[2025-8-29]`: 🔥 StableAvatar public demo is now live on [Hugging Face Spaces](https://huggingface.co/spaces/YinmingHuang/StableAvatar).(Note: due to the long video generation time, the demo is currently accessible to <b>Hugging Face Pro</b> users only.)
*   `[2025-8-18]`: 🔥 StableAvatar can run on [ComfyUI](https://github.com/smthemex/ComfyUI_StableAvatar) in just 10 steps, making it 3x faster. Thanks @[smthemex](https://github.com/smthemex) for the contribution.
*   `[2025-8-16]`: 🔥 We release the finetuning codes and lora training/finetuning codes! Other codes will be public as soon as possible. Stay tuned!
*   `[2025-8-15]`: 🔥 StableAvatar can run on Gradio Interface. Thanks @[gluttony-10](https://space.bilibili.com/893892) for the contribution!
*   `[2025-8-15]`: 🔥 StableAvatar can run on [ComfyUI](https://github.com/smthemex/ComfyUI_StableAvatar). Thanks @[smthemex](https://github.com/smthemex) for the contribution.
*   `[2025-8-13]`: 🔥 Added changes to run StableAvatar on the new Blackwell series Nvidia chips, including the RTX 6000 Pro.
*   `[2025-8-11]`: 🔥 The project page, code, technical report and [a basic model checkpoint](https://huggingface.co/FrancisRing/StableAvatar/tree/main) are released. Further lora training codes, the evaluation dataset and StableAvatar-pro will be released very soon. Stay tuned!

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

The basic model checkpoint (Wan2.1-1.3B-based) supports generating **infinite-length videos at 480x832, 832x480, or 512x512 resolutions**. Adjust frame counts or resolution to address memory issues.

### 🧱 Environment Setup

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
# Optional to install flash_attn to accelerate attention computation
pip install flash_attn
```

### 🧱 Environment Setup for Blackwell series chips

```bash
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
# Optional to install flash_attn to accelerate attention computation
pip install flash_attn
```

### 🧱 Download Weights

If you encounter connection issues with Hugging Face, set the environment variable: `export HF_ENDPOINT=https://hf-mirror.com`.

Download weights manually:

```bash
pip install "huggingface_hub[cli]"
cd StableAvatar
mkdir checkpoints
huggingface-cli download FrancisRing/StableAvatar --local-dir ./checkpoints
```

The file structure should be:

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

Extract audio from video files:

```bash
python audio_extractor.py --video_path="path/test/video.mp4" --saved_audio_path="path/test/audio.wav"
```

### 🧱 Vocal Separation

Separate vocals from audio:

```bash
pip install audio-separator[gpu]
python vocal_seperator.py --audio_separator_model_file="path/StableAvatar/checkpoints/Kim_Vocal_2.onnx" --audio_file_path="path/test/audio.wav" --saved_vocal_path="path/test/vocal.wav"
```

### 🧱 Base Model Inference

Use `inference.sh` as a sample configuration, modifying parameters like resolution, paths to resources, and prompts.

```bash
bash inference.sh
```

Key parameters:

*   `--width`, `--height`: Output resolution (512x512, 480x832, 832x480).
*   `--output_dir`: Output video path.
*   `--validation_reference_path`, `--validation_driven_audio_path`, `--validation_prompts`: Paths to reference image, audio, and text prompts.
*   `--pretrained_model_name_or_path`, `--pretrained_wav2vec_path`, `--transformer_path`: Paths to model weights.
*   `--sample_steps`, `--overlap_window_length`, `--clip_sample_n_frames`: Inference settings.
*   `--sample_text_guide_scale`, `--sample_audio_guide_scale`: Guidance scales.

Prompts are crucial: `[Description of first frame]-[Description of human behavior]-[Description of background (optional)]`.

You can also run a Gradio interface:

```bash
python app.py
```

Example videos are in `path/StableAvatar/examples`.

#### 💡Tips

*   Model versions: `transformer3d-square.pt` and `transformer3d-rec-vec.pt`.
*   GPU memory mode: `--GPU_memory_mode`: `model_full_load`, `sequential_cpu_offload`, `model_cpu_offload_and_qfloat8`, `model_cpu_offload`.
*   Multi-GPU inference: `--ulysses_degree` and `--ring_degree` in `inference.sh`.
*   Combine video and audio with ffmpeg:

```bash
ffmpeg -i video_without_audio.mp4 -i /path/audio.wav -c:v copy -c:a aac -shortest /path/output_with_audio.mp4
```

### 🧱 Model Training

**🔥🔥This tutorial is also useful for training conditioned Video Diffusion Transformer (DiT) models, such as Wan2.1.🔥🔥**

Training dataset structure:

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
*   Each folder has: `speech`, `singing`, and `dancing`.
*   `images`: RGB frames (`frame_i.png`).
*   `face_masks`, `lip_masks`: Corresponding masks.
*   `sub_clip.mp4`, `audio.wav`: Video and audio.
*   `video_square_path.txt`, `video_rec_path.txt`, `video_vec_path.txt`: Text files with video paths.

Extract frames with `ffmpeg`:

```bash
ffmpeg -i raw_video_1.mp4 -q:v 1 -start_number 0 path/StableAvatar/talking_face_data/rec/speech/00001/images/frame_%d.png
```

Extract face masks (see [StableAnimator repo](https://github.com/Francis-Rings/StableAnimator)) and lip masks (using `lip_mask_extractor.py`).

Run training commands, selecting the script based on the desired resolution and machine setup (`train_1B_square.sh`, `train_1B_rec_vec.sh`, `train_1B_square_64.sh`, `train_1B_rec_vec_64.sh`):

```bash
# Training StableAvatar on a single resolution setting (512x512) in a single machine
bash train_1B_square.sh
# Training StableAvatar on a single resolution setting (512x512) in multiple machines
bash train_1B_square_64.sh
# Training StableAvatar on a mixed resolution setting (480x832 and 832x480) in a single machine
bash train_1B_rec_vec.sh
# Training StableAvatar on a mixed resolution setting (480x832 and 832x480) in multiple machines
bash train_1B_rec_vec_64.sh
```

Key parameters:

*   `CUDA_VISIBLE_DEVICES`: GPU devices.
*   `--pretrained_model_name_or_path`, `--pretrained_wav2vec_path`, `--output_dir`: Model, Wav2Vec2.0, and output paths.
*   `--train_data_square_dir`, `--train_data_rec_dir`, `--train_data_vec_dir`: Training data paths.
*   `--validation_reference_path`, `--validation_driven_audio_path`: Validation data paths.
*   `--video_sample_n_frames`: Frames per batch.
*   `--num_train_epochs`: Training epochs.

**VRAM Requirements:** 50GB VRAM (mixed-resolution) or 40GB (512x512).

Ensure static backgrounds and clear audio for optimal results.

Training Wan2.1-14B:

```bash
# Training StableAvatar on a mixed resolution setting (480x832, 832x480, and 512x512) in multiple machines
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./checkpoints/Wan2.1-I2V-14B-480P
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-720P --local-dir ./checkpoints/Wan2.1-I2V-14B-720P # Optional
bash train_14B.sh
```

LoRA training commands:  (`train_1B_rec_vec_lora.sh`, `train_1B_rec_vec_lora_64.sh`, `train_14B_lora.sh`)

```bash
# Training StableAvatar-1.3B on a mixed resolution setting (480x832 and 832x480) in a single machine
bash train_1B_rec_vec_lora.sh
# Training StableAvatar-1.3B on a mixed resolution setting (480x832 and 832x480) in multiple machines
bash train_1B_rec_vec_lora_64.sh
# Lora-Training StableAvatar-14B on a mixed resolution setting (480x832, 832x480, and 512x512) in multiple machines
bash train_14B_lora.sh
```

Modify `--rank` and `--network_alpha` for LoRA quality control.

### 🧱 Model Finetuning

Finetune by adding `--transformer_path` to the training command with the checkpoint path (e.g., `transformer3d-square.pt`).

```bash
# Finetuning StableAvatar on a mixed resolution setting (480x832 and 832x480) in a single machine
bash train_1B_rec_vec.sh --transformer_path="path/StableAvatar/checkpoints/StableAvatar-1.3B/transformer3d-square.pt"
# Finetuning StableAvatar on a mixed resolution setting (480x832 and 832x480) in multiple machines
bash train_1B_rec_vec_64.sh --transformer_path="path/StableAvatar/checkpoints/StableAvatar-1.3B/transformer3d-square.pt"
```

Lora finetuning, apply the same principle, with `--transformer_path` to the lora training command.

### 🧱 VRAM and Runtime

*   5s video (480x832, fps=25): ~18GB VRAM, 3 minutes (4090 GPU, `--GPU_memory_mode="model_full_load"`).
*   VAE decoder can run on CPU.

## Contact

For suggestions or help, contact francisshuyuan@gmail.com.

If you find this work helpful, please star ⭐ this repository and cite the paper:

```bib
@article{tu2025stableavatar,
  title={StableAvatar: Infinite-Length Audio-Driven Avatar Video Generation},
  author={Tu, Shuyuan and Pan, Yueming and Huang, Yinming and Han, Xintong and Xing, Zhen and Dai, Qi and Luo, Chong and Wu, Zuxuan and Jiang Yu-Gang},
  journal={arXiv preprint arXiv:2508.08248},
  year={2025}
}
```