# StableAvatar: Generate Infinite-Length, Audio-Driven Avatar Videos

**Create stunning, infinite-length avatar videos driven by audio with StableAvatar, the cutting-edge solution for realistic and engaging visual content.**

[![Project Page](https://img.shields.io/badge/Project-Page-Green)](https://francis-rings.github.io/StableAvatar)
[![Arxiv](https://img.shields.io/badge/Paper-Arxiv-red)](https://arxiv.org/abs/2508.08248)
[![Hugging Face Model](https://img.shields.io/badge/HuggingFace-Model-orange)](https://huggingface.co/FrancisRing/StableAvatar/tree/main)
[![Hugging Face Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue)](https://huggingface.co/spaces/YinmingHuang/StableAvatar)
[![YouTube](https://img.shields.io/badge/YouTube-Watch-red?style=flat-square&logo=youtube)](https://www.youtube.com/watch?v=6lhvmbzvv3Y)
[![Bilibili](https://img.shields.io/badge/Bilibili-Watch-blue?style=flat-square&logo=bilibili)](https://www.bilibili.com/video/BV1hUt9z4EoQ)

StableAvatar introduces a novel approach to audio-driven avatar video generation, enabling the creation of high-quality, infinitely long videos with seamless audio synchronization and identity preservation. Developed by Shuyuan Tu et al. (Fudan University, Microsoft Research Asia, Xi'an Jiaotong University, Tencent Inc), StableAvatar overcomes the limitations of existing models, delivering exceptional results without the need for post-processing tools like face-swapping or restoration.

**(See video examples below showcasing StableAvatar's capabilities)**

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

## Key Features

*   **Infinite-Length Video Generation:**  Generate videos of virtually unlimited duration.
*   **Identity Preservation:** Ensures consistent appearance and identity throughout the video.
*   **High-Fidelity Results:** Produces high-quality videos without the need for external face-related post-processing.
*   **Audio Synchronization:**  Seamlessly synchronizes the avatar's movements with the provided audio.
*   **End-to-End Solution:**  Operates as a complete, self-contained solution for generating avatar videos.

**(Comparison video showing StableAvatar's superiority over existing models)**

<p align="center">
  <video src="https://github.com/user-attachments/assets/90691318-311e-40b9-9bd9-62db83ab1492" width="768" autoplay loop muted playsinline></video>
  <br/>
  <span>Comparison results between StableAvatar and state-of-the-art (SOTA) audio-driven avatar video generation models highlight the superior performance of StableAvatar in delivering <b>infinite-length, high-fidelity, identity-preserving avatar animation</b>.</span>
</p>

## Overview

StableAvatar utilizes a novel framework, depicted below, to achieve its impressive results.

<p align="center">
  <img src="assets/figures/framework.jpg" alt="model architecture" width="1280"/>
  </br>
  <i>The overview of the framework of StableAvatar.</i>
</p>

The core innovation lies in the introduction of a Time-step-aware Audio Adapter and an Audio Native Guidance Mechanism, along with a Dynamic Weighted Sliding-window Strategy, to prevent error accumulation and improve audio synchronization. This leads to the generation of long, smooth, and high-quality videos.

## News & Updates

*   **[2025-9-8]**: 🔥  Brand new demo available! Check out the generated videos on [YouTube](https://www.youtube.com/watch?v=GH4hrxIis3Q) and [Bilibili](https://www.bilibili.com/video/BV1jGYPzqEux).
*   **[2025-8-29]**: 🔥 Public demo live on [Hugging Face Spaces](https://huggingface.co/spaces/YinmingHuang/StableAvatar) (Hugging Face Pro users only).
*   **[2025-8-18]**: 🔥 StableAvatar now runs on [ComfyUI](https://github.com/smthemex/ComfyUI_StableAvatar), speeding up processing by 3x! Thanks @[smthemex](https://github.com/smthemex).
*   **[2025-8-16]**: 🔥 Finetuning and LoRA training/finetuning codes released!
*   **[2025-8-15]**: 🔥 StableAvatar runs on Gradio Interface (thanks to @[gluttony-10](https://space.bilibili.com/893892)!).
*   **[2025-8-15]**: 🔥 StableAvatar integration with [ComfyUI](https://github.com/smthemex/ComfyUI_StableAvatar) by @[smthemex](https://github.com/smthemex).
*   **[2025-8-13]**: 🔥 Updated for compatibility with new Blackwell series Nvidia chips (e.g., RTX 6000 Pro).
*   **[2025-8-11]**: 🔥 Project page, code, technical report, and [basic model checkpoint](https://huggingface.co/FrancisRing/StableAvatar/tree/main) released!

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

Get started generating infinite-length videos with the basic version of the model (Wan2.1-1.3B-based). Supports resolutions of 480x832, 832x480, or 512x512. Adjust frame count/resolution for memory optimization.

### 🧱 Environment Setup

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
# Optional: Install flash_attn for faster attention computation
pip install flash_attn
```

### 🧱 Environment Setup for Blackwell Series Chips

```bash
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
# Optional: Install flash_attn for faster attention computation
pip install flash_attn
```

### 🧱 Download Weights

If you face connection issues with Hugging Face, set the environment variable: `export HF_ENDPOINT=https://hf-mirror.com`.

Manual download:

```bash
pip install "huggingface_hub[cli]"
cd StableAvatar
mkdir checkpoints
huggingface-cli download FrancisRing/StableAvatar --local-dir ./checkpoints
```

Organize weights as follows:

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

Extract audio from a video (.mp4):

```bash
python audio_extractor.py --video_path="path/test/video.mp4" --saved_audio_path="path/test/audio.wav"
```

### 🧱 Vocal Separation

Separate vocals from audio (.wav):

```bash
pip install audio-separator[gpu]
python vocal_seperator.py --audio_separator_model_file="path/StableAvatar/checkpoints/Kim_Vocal_2.onnx" --audio_file_path="path/test/audio.wav" --saved_vocal_path="path/test/vocal.wav"
```

### 🧱 Base Model Inference

Example `inference.sh` configuration:

```bash
#Example bash inference.sh
bash inference.sh
```

StableAvatar (Wan2.1-1.3B-based) supports 512x512, 480x832, and 832x480 resolutions.  Modify "--width" and "--height" in `inference.sh`.  "--output_dir" is the save path.  "--validation_reference_path", "--validation_driven_audio_path", and "--validation_prompts" specify input paths.

Prompts are key: `[Description of first frame]-[Description of human behavior]-[Description of background (optional)]`.

"--pretrained_model_name_or_path", "--pretrained_wav2vec_path", and "--transformer_path" are model paths.  Adjust "--sample_steps", "--overlap_window_length", and "--clip_sample_n_frames" for quality and speed. Recommended text and audio CFG is `[3-6]`.

Launch Gradio interface:

```bash
python app.py
```

Example videos are in `path/StableAvatar/examples`.

#### 💡 Tips

*   Two versions of the Wan2.1-1.3B-based weights are available: `transformer3d-square.pt` and `transformer3d-rec-vec.pt`.  Use `--transformer_path` in `inference.sh` to switch between them.
*   For limited GPU resources, use `--GPU_memory_mode` in `inference.sh`: `model_full_load`, `sequential_cpu_offload`, `model_cpu_offload_and_qfloat8`, or `model_cpu_offload`.
*   Use Multi-GPU with `--ulysses_degree` and `--ring_degree` in `inference.sh`. Also, add `--fsdp_dit` to reduce GPU memory consumption.
*   For audio with video files, use ffmpeg:

```bash
ffmpeg -i video_without_audio.mp4 -i /path/audio.wav -c:v copy -c:a aac -shortest /path/output_with_audio.mp4
```

### 🧱 Model Training

**🔥🔥This tutorial is valuable if you're training a conditioned Video Diffusion Transformer (DiT) model (e.g., Wan2.1)🔥🔥**

Dataset format:

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

Videos of 512x512 are in `talking_face_data/square`, 480x832 in `talking_face_data/vec`, and 832x480 in `talking_face_data/rec`.  Each folder contains subfolders for different video types (speech, singing, dancing).

`images`, `face_masks`, and `lip_masks` store frames, face masks, and lip masks, respectively.  `sub_clip.mp4` and `audio.wav` are the video and audio files.

`video_square_path.txt`, `video_rec_path.txt`, and `video_vec_path.txt` list the video paths.

Extract frames with ffmpeg:

```bash
ffmpeg -i raw_video_1.mp4 -q:v 1 -start_number 0 path/StableAvatar/talking_face_data/rec/speech/00001/images/frame_%d.png
```

Extract face masks from the [StableAnimator repo](https://github.com/Francis-Rings/StableAnimator).

Extract lip masks with mediapipe:

```bash
pip install mediapipe
python lip_mask_extractor.py --folder_root="path/StableAvatar/talking_face_data/rec/singing" --start=1 --end=500
```

Follow the Audio Extraction section for extracting audio.

Training command:

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

Modify the parameters in `train_1B_square.sh` and `train_1B_rec_vec.sh` (e.g., `CUDA_VISIBLE_DEVICES`, `--pretrained_model_name_or_path`, `--pretrained_wav2vec_path`, `--output_dir`, etc.).

The training setup is shown as follows:
```
StableAvatar/
├── accelerate_config
├── deepspeed_config
├── talking_face_data
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

**Note:** Requires ~50GB VRAM due to mixed resolution training, or ~40GB for 512x512. The background should be static, and the audio should be free of noise.

Regarding training Wan2.1-14B-based StableAvatar, you can run the following command:
```bash
# Training StableAvatar on a mixed resolution setting (480x832, 832x480, and 512x512) in multiple machines
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./checkpoints/Wan2.1-I2V-14B-480P
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-720P --local-dir ./checkpoints/Wan2.1-I2V-14B-720P # Optional
bash train_14B.sh
```

### 🧱 Model Finetuning

To finetune, use the following:

```bash
# Finetuning StableAvatar on a mixed resolution setting (480x832 and 832x480) in a single machine
bash train_1B_rec_vec.sh
# Finetuning StableAvatar on a mixed resolution setting (480x832 and 832x480) in multiple machines
bash train_1B_rec_vec_64.sh
```
You can modify `--rank` and `--network_alpha` to control the quality of your lora training/finetuning.

### 🧱 VRAM and Runtime

The basic model (480x832, fps=25,  `--GPU_memory_mode="model_full_load"`) uses ~18GB VRAM and takes ~3 minutes per 5-second video on a 4090 GPU.

**🔥🔥StableAvatar can theoretically generate hours of video without significant quality loss. The 3D VAE decoder requires a large amount of GPU memory, especially when decoding 10k+ frames. You can run the VAE on the CPU.🔥🔥**

## Contact

For suggestions or help, contact:

Email: francisshuyuan@gmail.com

If you find our work useful, **please give a star ⭐ to this GitHub repository and cite it:**

```bib
@article{tu2025stableavatar,
  title={Stableavatar: Infinite-length audio-driven avatar video generation},
  author={Tu, Shuyuan and Pan, Yueming and Huang, Yinming and Han, Xintong and Xing, Zhen and Dai, Qi and Luo, Chong and Wu, Zuxuan and Jiang, Yu-Gang},
  journal={arXiv preprint arXiv:2508.08248},
  year={2025}
}
```

**[Visit the original repository](https://github.com/Francis-Rings/StableAvatar) for the latest updates and more details.**