# StableAvatar: Generate Infinite-Length Audio-Driven Avatar Videos

> Create stunning, infinite-length avatar videos from audio with **StableAvatar**, the cutting-edge solution for lifelike digital characters.
[Explore the original repository](https://github.com/Francis-Rings/StableAvatar)

[![Project Page](https://img.shields.io/badge/Project-Page-Green)](https://francis-rings.github.io/StableAvatar)
[![Arxiv](https://img.shields.io/badge/Paper-Arxiv-red)](https://arxiv.org/abs/2508.08248)
[![Hugging Face Model](https://img.shields.io/badge/HuggingFace-Model-orange)](https://huggingface.co/FrancisRing/StableAvatar/tree/main)
[![Hugging Face Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue)](https://huggingface.co/spaces/YinmingHuang/StableAvatar)
[![YouTube](https://img.shields.io/badge/YouTube-Watch-red?style=flat-square&logo=youtube)](https://www.youtube.com/watch?v=6lhvmbzvv3Y)
[![Bilibili](https://img.shields.io/badge/Bilibili-Watch-blue?style=flat-square&logo=bilibili)](https://www.bilibili.com/video/BV1hUt9z4EoQ)

StableAvatar is a groundbreaking model for generating high-quality, infinite-length avatar videos driven by audio input. It excels in producing long videos with exceptional audio synchronization and identity preservation, eliminating the need for post-processing.

**Key Features:**

*   **Infinite-Length Video Generation:** Synthesize videos of any length without degradation.
*   **ID Preservation:** Maintains consistent identity throughout the entire video sequence.
*   **End-to-End Synthesis:** Generates videos directly, without the need for face-swapping or restoration tools.
*   **High Fidelity:** Delivers high-quality, realistic avatar animations.
*   **Audio Synchronization:** Provides exceptional lip-sync and audio-visual coherence.

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/d7eca208-6a14-46af-b337-fb4d2b66ba8d" width="320" controls loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/b5902ac4-8188-4da8-b9e6-6df280690ed1" width="320" controls loop></video>
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

**Comparison:** StableAvatar outperforms other audio-driven avatar video generation models, generating infinite-length, high-fidelity, and identity-preserving animations.

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

StableAvatar introduces a novel Time-step-aware Audio Adapter to prevent error accumulation and an Audio Native Guidance Mechanism to enhance audio synchronization. A Dynamic Weighted Sliding-window Strategy is used to enhance smoothness.

## News

*   `[2025-8-29]`:🔥 StableAvatar public demo is now live on [Hugging Face Spaces](https://huggingface.co/spaces/YinmingHuang/StableAvatar).(Note: due to the long video generation time, the demo is currently accessible to <b>Hugging Face Pro</b> users only.)
*   `[2025-8-18]`:🔥 StableAvatar can run on [ComfyUI](https://github.com/smthemex/ComfyUI_StableAvatar) in just 10 steps, making it 3x faster. Thanks @[smthemex](https://github.com/smthemex) for the contribution.
*   `[2025-8-16]`:🔥 We release the finetuning codes and lora training/finetuning codes! Other codes will be public as soon as possible. Stay tuned!
*   `[2025-8-15]`:🔥 StableAvatar can run on Gradio Interface. Thanks @[gluttony-10](https://space.bilibili.com/893892) for the contribution!
*   `[2025-8-15]`:🔥 StableAvatar can run on [ComfyUI](https://github.com/smthemex/ComfyUI_StableAvatar). Thanks @[smthemex](https://github.com/smthemex) for the contribution.
*   `[2025-8-13]`:🔥 Added changes to run StableAvatar on the new Blackwell series Nvidia chips, including the RTX 6000 Pro.
*   `[2025-8-11]`:🔥 The project page, code, technical report and [a basic model checkpoint](https://huggingface.co/FrancisRing/StableAvatar/tree/main) are released. Further lora training codes, the evaluation dataset and StableAvatar-pro will be released very soon. Stay tuned!

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

The basic version of the model checkpoint (Wan2.1-1.3B-based) supports generating infinite-length videos at a 480x832, 832x480 or 512x512 resolution. Reduce animation frames or output resolution to resolve memory issues.

### 🧱 Environment setup

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
# Optional to install flash_attn to accelerate attention computation
pip install flash_attn
```

### 🧱 Environment setup for Blackwell series chips

```bash
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
# Optional to install flash_attn to accelerate attention computation
pip install flash_attn
```

### 🧱 Download weights

If you encounter connection issues with Hugging Face, you can utilize the mirror endpoint by setting the environment variable: `export HF_ENDPOINT=https://hf-mirror.com`.
Please download weights manually as follows:

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

Extract audio from a video:

```bash
python audio_extractor.py --video_path="path/test/video.mp4" --saved_audio_path="path/test/audio.wav"
```

### 🧱 Vocal Separation

Separate vocals from audio:

```bash
pip install audio-separator[gpu]
python vocal_seperator.py --audio_separator_model_file="path/StableAvatar/checkpoints/Kim_Vocal_2.onnx" --audio_file_path="path/test/audio.wav" --saved_vocal_path="path/test/vocal.wav"
```

### 🧱 Base Model inference

Run inference:

```bash
bash inference.sh
```

*   Supports 512x512, 480x832, and 832x480 resolutions.
*   Modify `--width` and `--height` in `inference.sh` to change resolution.
*   `--output_dir`: output directory.
*   `--validation_reference_path`, `--validation_driven_audio_path`, and `--validation_prompts`: paths for reference image, audio, and text prompts.
*   Prompts: `[Description of first frame]-[Description of human behavior]-[Description of background (optional)]`.
*   `--pretrained_model_name_or_path`, `--pretrained_wav2vec_path`, and `--transformer_path`: paths for pretrained Wan2.1-1.3B weights, Wav2Vec2.0 weights, and StableAvatar weights.
*   `--sample_steps`, `--overlap_window_length`, and `--clip_sample_n_frames`: steps, context length overlap, and frames per batch.
*   Recommended `--sample_steps`: \[30-50].
*   Recommended `--overlap_window_length`: \[5-15].
*   `--sample_text_guide_scale` and `--sample_audio_guide_scale`: Text and audio CFG scales, recommended range \[3-6].

Run a Gradio interface:

```bash
python app.py
```

Examples are in `path/StableAvatar/examples`.

#### 💡Tips

*   Two versions of Wan2.1-1.3B-based StableAvatar weights: `transformer3d-square.pt` and `transformer3d-rec-vec.pt`.  Modify `--transformer_path` to switch.
*   GPU memory modes via `--GPU_memory_mode`:  `model_full_load`, `sequential_cpu_offload`, `model_cpu_offload_and_qfloat8`, `model_cpu_offload`.
*   Multi-GPU inference with `--ulysses_degree` and `--ring_degree`.
*   Use `ffmpeg` to add audio to the output:

```bash
ffmpeg -i video_without_audio.mp4 -i /path/audio.wav -c:v copy -c:a aac -shortest /path/output_with_audio.mp4
```

### 🧱 Model Training

**🔥🔥If you are looking to train a conditioned Video Diffusion Transformer (DiT) model, such as Wan2.1, this training tutorial will also be helpful.🔥🔥**
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

*   `square`: 512x512 videos.
*   `vec`: 480x832 videos.
*   `rec`: 832x480 videos.
*   `images`:  RGB frames (frame\_i.png).
*   `face_masks`: Face masks (frame\_i.png).
*   `lip_masks`: Lip masks (frame\_i.png).
*   `sub_clip.mp4`: Video clip.
*   `audio.wav`: Audio file.
*   `video_square_path.txt`, `video_rec_path.txt`, and `video_vec_path.txt`: Lists paths to video folders.

Extract frames using ffmpeg:

```bash
ffmpeg -i raw_video_1.mp4 -q:v 1 -start_number 0 path/StableAvatar/talking_face_data/rec/speech/00001/images/frame_%d.png
```

Extract face masks from [StableAnimator repo](https://github.com/Francis-Rings/StableAnimator).

Extract lip masks:

```bash
pip install mediapipe
python lip_mask_extractor.py --folder_root="path/StableAvatar/talking_face_data/rec/singing" --start=1 --end=500
```

Training with:

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

*   Modify the scripts for the training parameters.
*   `CUDA_VISIBLE_DEVICES`: GPU device(s).
*   `--pretrained_model_name_or_path`, `--pretrained_wav2vec_path`, and `--output_dir`: pretrained model path, wav2vec path, and output path.
*   `--train_data_square_dir`, `--train_data_rec_dir`, and `--train_data_vec_dir`: data paths.
*   `--validation_reference_path` and `--validation_driven_audio_path`: Validation reference image and audio paths.
*   `--video_sample_n_frames`: frames per batch.
*   `--num_train_epochs`: Training epochs (set to infinite).

File structure during training:

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

**VRAM:** Requires approximately 50GB VRAM for mixed-resolution training (reduced to 40GB for 512x512). The background of the selected training videos should remain static and the audio should be clear.

Regarding training Wan2.1-14B-based StableAvatar, you can run the following command:
```
# Training StableAvatar on a mixed resolution setting (480x832, 832x480, and 512x512) in multiple machines
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./checkpoints/Wan2.1-I2V-14B-480P
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-720P --local-dir ./checkpoints/Wan2.1-I2V-14B-720P # Optional
bash train_14B.sh
```
We utilize deepspeed stage-2 to train Wan2.1-14B-based StableAvatar. The GPU configuration can be modified in `path/StableAvatar/accelerate_config/accelerate_config_machine_14B_multiple.yaml`.
The deepspeed optimization configuration and deepspeed scheduler configuration are in `path/StableAvatar/deepspeed_config/zero_stage2_config.json`.
Notably, we observe that Wan2.1-1.3B-based StableAvatar is already capable of synthesizing infinite-length high quality avatar videos. The Wan2.1-14B backbone significantly increase the inference latency and GPU memory consumption during training, indicating limited efficiency in terms of performance-to-resource ratio.

You can also run the following commands to perform lora training:
```
# Training StableAvatar-1.3B on a mixed resolution setting (480x832 and 832x480) in a single machine
bash train_1B_rec_vec_lora.sh
# Training StableAvatar-1.3B on a mixed resolution setting (480x832 and 832x480) in multiple machines
bash train_1B_rec_vec_lora_64.sh
# Lora-Training StableAvatar-14B on a mixed resolution setting (480x832, 832x480, and 512x512) in multiple machines
bash train_14B_lora.sh
```
You can modify `--rank` and `--network_alpha` to control the quality of your lora training/finetuning.

If you want to train 720P Wan2.1-1.3B-based or Wan2.1-14B-based StableAvatar, you can directly modify the height and width of the dataloader (480p-->720p) in `train_1B_square.py`/`train_1B_vec_rec.py`/`train_14B.py`.

### 🧱 Model Finetuning
Regarding fully finetuning StableAvatar, you can add `--transformer_path="path/StableAvatar/checkpoints/StableAvatar-1.3B/transformer3d-square.pt"` to the `train_1B_rec_vec.sh` or `train_1B_rec_vec_64.sh`:
```
# Finetuning StableAvatar on a mixed resolution setting (480x832 and 832x480) in a single machine
bash train_1B_rec_vec.sh
# Finetuning StableAvatar on a mixed resolution setting (480x832 and 832x480) in multiple machines
bash train_1B_rec_vec_64.sh
```
In terms of lora finetuning StableAvatar, you can add `--transformer_path="path/StableAvatar/checkpoints/StableAvatar-1.3B/transformer3d-square.pt"` to the `train_1B_rec_vec_lora.sh`:
```
# Lora-Finetuning StableAvatar-1.3B on a mixed resolution setting (480x832 and 832x480) in a single machine
bash train_1B_rec_vec_lora.sh
```
You can modify `--rank` and `--network_alpha` to control the quality of your lora training/finetuning.

### 🧱 VRAM requirement and Runtime

For the 5s video (480x832, fps=25), the basic model (--GPU_memory_mode="model_full_load") requires approximately 18GB VRAM and finishes in 3 minutes on a 4090 GPU.

<b>🔥🔥Theoretically, StableAvatar is capable of synthesizing hours of video without significant quality degradation; however, the 3D VAE decoder demands significant GPU memory, especially when decoding 10k+ frames. You have the option to run the VAE decoder on CPU.🔥🔥</b>

## Contact

For suggestions or help, contact:

*   Email: francisshuyuan@gmail.com

If you find our work useful, <b>please consider giving a star ⭐ to this github repository and citing it ❤️</b>:

```bib
@article{tu2025stableavatar,
  title={StableAvatar: Infinite-Length Audio-Driven Avatar Video Generation},
  author={Tu, Shuyuan and Pan, Yueming and Huang, Yinming and Han, Xintong and Xing, Zhen and Dai, Qi and Luo, Chong and Wu, Zuxuan and Jiang Yu-Gang},
  journal={arXiv preprint arXiv:2508.08248},
  year={2025}
}
```