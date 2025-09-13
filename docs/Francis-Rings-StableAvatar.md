# StableAvatar: Generate Infinite-Length Audio-Driven Avatar Videos

StableAvatar unlocks the creation of endless, high-quality avatar videos driven by audio, pushing the boundaries of AI-powered video generation. [Visit the original repository](https://github.com/Francis-Rings/StableAvatar) for the latest updates and code.

[![Project Page](https://img.shields.io/badge/Project-Page-Green)](https://francis-rings.github.io/StableAvatar)
[![Paper](https://img.shields.io/badge/Paper-Arxiv-red)](https://arxiv.org/abs/2508.08248)
[![HuggingFace Model](https://img.shields.io/badge/HuggingFace-Model-orange)](https://huggingface.co/FrancisRing/StableAvatar/tree/main)
[![Hugging Face Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue)](https://huggingface.co/spaces/YinmingHuang/StableAvatar)
[![YouTube](https://img.shields.io/badge/YouTube-Watch-red?style=flat-square&logo=youtube)](https://www.youtube.com/watch?v=6lhvmbzvv3Y)
[![Bilibili](https://img.shields.io/badge/Bilibili-Watch-blue?style=flat-square&logo=bilibili)](https://www.bilibili.com/video/BV1hUt9z4EoQ)

**Key Features:**

*   **Infinite-Length Video Generation:** Synthesize videos of arbitrary length without degradation.
*   **Identity Preservation:** Maintains the subject's identity throughout the video.
*   **High Fidelity:** Produces high-quality video output.
*   **End-to-End Synthesis:** Generates videos directly without requiring face-related post-processing like face swapping or restoration.
*   **Time-step-aware Audio Adapter:** Prevents error accumulation to generate long videos.
*   **Audio Native Guidance Mechanism:** Enhances audio synchronization.
*   **Dynamic Weighted Sliding-window Strategy:** Smooths infinite-length videos.

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
  <span>StableAvatar's audio-driven avatar videos demonstrate its ability to synthesize <b>infinite-length</b> and <b>ID-preserving videos</b>. These videos are generated <b>directly by StableAvatar, without face-related post-processing</b>.</span>
</p>

<p align="center">
  <video src="https://github.com/user-attachments/assets/90691318-311e-40b9-9bd9-62db83ab1492" width="768" autoplay loop muted playsinline></video>
  <br/>
  <span>Comparison of StableAvatar with state-of-the-art (SOTA) audio-driven avatar video generation models highlights StableAvatar's superior performance in delivering <b>infinite-length, high-fidelity, identity-preserving avatar animation</b>.</span>
</p>

## Overview

<p align="center">
  <img src="assets/figures/framework.jpg" alt="model architecture" width="1280"/>
  </br>
  <i>The overview of the framework of StableAvatar.</i>
</p>

StableAvatar is a pioneering end-to-end video diffusion transformer addressing the limitations of existing models, which struggle to generate long videos with consistent audio synchronization and identity. It leverages tailored training and inference modules to enable infinite-length video generation conditioned on a reference image and audio. The core innovation lies in a novel Time-step-aware Audio Adapter and an Audio Native Guidance Mechanism.
Experiments on benchmarks demonstrate the qualitative and quantitative effectiveness of StableAvatar.

## News

*   `[2025-9-8]`: 🔥 Exciting new demo videos on [YouTube](https://www.youtube.com/watch?v=GH4hrxIis3Q) and [Bilibili](https://www.bilibili.com/video/BV1jGYPzqEux)!
*   `[2025-8-29]`: 🔥 Public demo now live on [Hugging Face Spaces](https://huggingface.co/spaces/YinmingHuang/StableAvatar) (for Hugging Face Pro users).
*   `[2025-8-18]`: 🔥 StableAvatar runs on [ComfyUI](https://github.com/smthemex/ComfyUI_StableAvatar) for faster generation (3x speedup).
*   `[2025-8-16]`: 🔥 Finetuning and LoRA training/finetuning codes released!
*   `[2025-8-15]`: 🔥 StableAvatar runs on Gradio Interface (thanks @[gluttony-10](https://space.bilibili.com/893892)) and [ComfyUI](https://github.com/smthemex/ComfyUI_StableAvatar).
*   `[2025-8-13]`: 🔥 Updates for new Blackwell series Nvidia chips.
*   `[2025-8-11]`: 🔥 Project page, code, technical report, and basic model checkpoint ([HuggingFace](https://huggingface.co/FrancisRing/StableAvatar/tree/main)) released.

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

The basic model checkpoint (Wan2.1-1.3B-based) supports generating infinite-length videos at 480x832, 832x480, or 512x512 resolutions.

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

If you face connection issues with Hugging Face, set the environment variable: `export HF_ENDPOINT=https://hf-mirror.com`.

Download weights manually:

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
│   ├── Kim_Vocal_2.onnx
│   ├── wav2vec2-base-960h
│   ├── Wan2.1-Fun-V1.1-1.3B-InP
│   └── StableAvatar-1.3B
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

Extract audio from video:

```bash
python audio_extractor.py --video_path="path/test/video.mp4" --saved_audio_path="path/test/audio.wav"
```

### 🧱 Vocal Separation

Separate vocals from audio (requires `audio-separator[gpu]`):

```bash
pip install audio-separator[gpu]
python vocal_seperator.py --audio_separator_model_file="path/StableAvatar/checkpoints/Kim_Vocal_2.onnx" --audio_file_path="path/test/audio.wav" --saved_vocal_path="path/test/vocal.wav"
```

### 🧱 Base Model Inference

Use `inference.sh` for testing (adjust parameters as needed):

```bash
bash inference.sh
```

*   Wan2.1-1.3B supports: 512x512, 480x832, 832x480 resolutions. Modify `--width` and `--height` in `inference.sh`.
*   `--output_dir`: Output video path.
*   `--validation_reference_path`, `--validation_driven_audio_path`, `--validation_prompts`: Reference image, audio, and text prompts paths.
*   Prompts are important: `[Description of first frame]-[Description of human behavior]-[Description of background (optional)]`.
*   `--pretrained_model_name_or_path`, `--pretrained_wav2vec_path`, `--transformer_path`: Pretrained model paths.
*   `--sample_steps`: Inference steps (30-50 recommended for quality).
*   `--overlap_window_length`: Overlapping context length (5-15 recommended for quality).
*   `--clip_sample_n_frames`: Frames per batch/context window.
*   `--sample_text_guide_scale`, `--sample_audio_guide_scale`: CFG scales (3-6 recommended).

Run Gradio interface:

```bash
python app.py
```

See examples in `path/StableAvatar/examples`.

#### 💡 Tips

*   Two Wan2.1-1.3B versions: `transformer3d-square.pt` and `transformer3d-rec-vec.pt`.
*   `--GPU_memory_mode`: Adjust GPU memory usage (`model_full_load`, `sequential_cpu_offload`, `model_cpu_offload_and_qfloat8`, `model_cpu_offload`).
*   Multi-GPU inference: `--ulysses_degree` and `--ring_degree`.
*   Add `--fsdp_dit` to enable FSDP in DiT. Run: `bash multiple_gpu_inference.sh` (example).
*   Convert output video to MP4 with audio: `ffmpeg -i video_without_audio.mp4 -i /path/audio.wav -c:v copy -c:a aac -shortest /path/output_with_audio.mp4`.

### 🧱 Model Training

<b>🔥🔥 Training tutorial for conditioned Video Diffusion Transformer (DiT) models like Wan2.1.🔥🔥</b>

Dataset structure:

```
talking_face_data/
├── rec
│   ├──speech
│   │   ├──00001
│   │   │   ├──sub_clip.mp4
│   │   │   ├──audio.wav
│   │   │   ├──images
│   │   │   │   ├──frame_0.png
│   │   │   │   ├──frame_1.png
│   │   │   │   ├──frame_2.png
│   │   │   │   ├──...
│   │   │   ├──face_masks
│   │   │   │   ├──frame_0.png
│   │   │   │   ├──frame_1.png
│   │   │   │   ├──frame_2.png
│   │   │   │   ├──...
│   │   │   ├──lip_masks
│   │   │   │   ├──frame_0.png
│   │   │   │   ├──frame_1.png
│   │   │   │   ├──frame_2.png
│   │   │   │   ├──...
│   │   ├──00002
│   │   │   ├──sub_clip.mp4
│   │   │   ├──audio.wav
│   │   │   ├──images
│   │   │   ├──face_masks
│   │   │   ├──lip_masks
│   │   └──...
│   ├──singing
│   │   ├──00001
│   │   │   ├──sub_clip.mp4
│   │   │   ├──audio.wav
│   │   │   ├──images
│   │   │   ├──face_masks
│   │   │   ├──lip_masks
│   │   └──...
│   ├──dancing
│   │   ├──00001
│   │   │   ├──sub_clip.mp4
│   │   │   ├──audio.wav
│   │   │   ├──images
│   │   │   ├──face_masks
│   │   │   ├──lip_masks
│   │   └──...
├── vec
│   ├──speech
│   │   ├──00001
│   │   │   ├──sub_clip.mp4
│   │   │   ├──audio.wav
│   │   │   ├──images
│   │   │   ├──face_masks
│   │   │   ├──lip_masks
│   │   └──...
│   ├──singing
│   │   ├──00001
│   │   │   ├──sub_clip.mp4
│   │   │   ├──audio.wav
│   │   │   ├──images
│   │   │   ├──face_masks
│   │   │   ├──lip_masks
│   │   └──...
│   ├──dancing
│   │   ├──00001
│   │   │   ├──sub_clip.mp4
│   │   │   ├──audio.wav
│   │   │   ├──images
│   │   │   ├──face_masks
│   │   │   ├──lip_masks
│   │   └──...
├── square
│   ├──speech
│   │   ├──00001
│   │   │   ├──sub_clip.mp4
│   │   │   ├──audio.wav
│   │   │   ├──images
│   │   │   ├──face_masks
│   │   │   ├──lip_masks
│   │   └──...
│   ├──singing
│   │   ├──00001
│   │   │   ├──sub_clip.mp4
│   │   │   ├──audio.wav
│   │   │   ├──images
│   │   │   ├──face_masks
│   │   │   ├──lip_masks
│   │   └──...
│   ├──dancing
│   │   ├──00001
│   │   │   ├──sub_clip.mp4
│   │   │   ├──audio.wav
│   │   │   ├──images
│   │   │   ├──face_masks
│   │   │   ├──lip_masks
│   │   └──...
├── video_rec_path.txt
├── video_square_path.txt
└── video_vec_path.txt
```

*   `talking_face_data/square`: 512x512 videos.
*   `talking_face_data/vec`: 480x832 videos.
*   `talking_face_data/rec`: 832x480 videos.
*   Subfolders within `square`, `rec`, `vec`: speech, singing, dancing.
*   `frame_i.png`: Image frames.
*   `00001`, `00002`: Video segments.
*   `images`, `face_masks`, `lip_masks`: RGB frames, face masks, lip masks.
*   `sub_clip.mp4`, `audio.wav`: Video and audio files.
*   `video_square_path.txt`, `video_rec_path.txt`, `video_vec_path.txt`: Text files listing video paths.

Extract frames from videos using `ffmpeg`:

```bash
ffmpeg -i raw_video_1.mp4 -q:v 1 -start_number 0 path/StableAvatar/talking_face_data/rec/speech/00001/images/frame_%d.png
```

Extract face masks (refer to [StableAnimator repo](https://github.com/Francis-Rings/StableAnimator)).

Extract lip masks (requires `mediapipe`):

```bash
pip install mediapipe
python lip_mask_extractor.py --folder_root="path/StableAvatar/talking_face_data/rec/singing" --start=1 --end=500
```

Refer to the Audio Extraction section for audio extraction.

Train with this command, based on the dataset structure:

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

*   `CUDA_VISIBLE_DEVICES`: GPU devices.
*   `--pretrained_model_name_or_path`, `--pretrained_wav2vec_path`, `--output_dir`: Paths to pretrained Wan2.1-1.3B, Wav2Vec2.0, and the output directory.
*   `--train_data_square_dir`, `--train_data_rec_dir`, `--train_data_vec_dir`: Paths to video path lists.
*   `--validation_reference_path`, `--validation_driven_audio_path`: Validation image and audio paths.
*   `--video_sample_n_frames`: Frames per batch.
*   `--num_train_epochs`: Training epochs (default: infinite - stop manually).

Training requires ~50GB VRAM for mixed resolutions, ~40GB for 512x512. Static backgrounds and clear audio are recommended.

For Wan2.1-14B training:

```bash
# Training StableAvatar on a mixed resolution setting (480x832, 832x480, and 512x512) in multiple machines
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./checkpoints/Wan2.1-I2V-14B-480P
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-720P --local-dir ./checkpoints/Wan2.1-I2V-14B-720P # Optional
bash train_14B.sh
```

LoRA training:

```bash
# Training StableAvatar-1.3B on a mixed resolution setting (480x832 and 832x480) in a single machine
bash train_1B_rec_vec_lora.sh
# Training StableAvatar-1.3B on a mixed resolution setting (480x832 and 832x480) in multiple machines
bash train_1B_rec_vec_lora_64.sh
# Lora-Training StableAvatar-14B on a mixed resolution setting (480x832, 832x480, and 512x512) in multiple machines
bash train_14B_lora.sh
```

### 🧱 Model Finetuning

Finetune by adding the `--transformer_path` argument to the training scripts:

```bash
# Finetuning StableAvatar on a mixed resolution setting (480x832 and 832x480) in a single machine
bash train_1B_rec_vec.sh
# Finetuning StableAvatar on a mixed resolution setting (480x832 and 832x480) in multiple machines
bash train_1B_rec_vec_64.sh
```

LoRA finetuning:

```bash
# Lora-Finetuning StableAvatar-1.3B on a mixed resolution setting (480x832 and 832x480) in a single machine
bash train_1B_rec_vec_lora.sh
```

### 🧱 VRAM requirement and Runtime

5s video (480x832, 25 fps) requires ~18GB VRAM and ~3 minutes on a 4090 GPU (basic model, `model_full_load`).

<b>🔥StableAvatar is capable of generating long videos; however, the 3D VAE decoder demands significant GPU memory. You have the option to run the VAE decoder on CPU.🔥</b>

## Contact

For suggestions or assistance, contact:

Email: francisshuyuan@gmail.com

If you find this work helpful, please star ⭐ and cite it ❤️:

```bib
@article{tu2025stableavatar,
  title={StableAvatar: Infinite-Length Audio-Driven Avatar Video Generation},
  author={Tu, Shuyuan and Pan, Yueming and Huang, Yinming and Han, Xintong and Xing, Zhen and Dai, Qi and Luo, Chong and Wu, Zuxuan and Jiang Yu-Gang},
  journal={arXiv preprint arXiv:2508.08248},
  year={2025}
}
```