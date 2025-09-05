# StableAvatar: Generate Infinite-Length Audio-Driven Avatar Videos

**Create stunning, realistic, and infinite-length avatar videos driven by audio using StableAvatar!** Check out the original repository: [https://github.com/Francis-Rings/StableAvatar](https://github.com/Francis-Rings/StableAvatar)

[![Project Page](https://img.shields.io/badge/Project-Page-Green)](https://francis-rings.github.io/StableAvatar)
[![Arxiv](https://img.shields.io/badge/Paper-Arxiv-red)](https://arxiv.org/abs/2508.08248)
[![HuggingFace Model](https://img.shields.io/badge/HuggingFace-Model-orange)](https://huggingface.co/FrancisRing/StableAvatar/tree/main)
[![Hugging Face Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue)](https://huggingface.co/spaces/YinmingHuang/StableAvatar)
[![YouTube](https://img.shields.io/badge/YouTube-Watch-red?style=flat-square&logo=youtube)](https://www.youtube.com/watch?v=6lhvmbzvv3Y)
[![Bilibili](https://img.shields.io/badge/Bilibili-Watch-blue?style=flat-square&logo=bilibili)](https://www.bilibili.com/video/BV1hUt9z4EoQ)

**Authors:** Shuyuan Tu<sup>1</sup>, Yueming Pan<sup>3</sup>, Yinming Huang<sup>1</sup>, Xintong Han<sup>4</sup>, Zhen Xing<sup>1</sup>, Qi Dai<sup>2</sup>, Chong Luo<sup>2</sup>, Zuxuan Wu<sup>1</sup>, Yu-Gang Jiang<sup>1</sup>
[<sup>1</sup>Fudan University; <sup>2</sup>Microsoft Research Asia; <sup>3</sup>Xi'an Jiaotong University; <sup>4</sup>Tencent Inc]

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

<p style="text-align: justify;">
  StableAvatar generates audio-driven avatar videos, demonstrating its capability to synthesize <b>infinite-length</b> and <b>identity-preserving videos</b>. All videos are <b>generated directly by StableAvatar, eliminating the need for face-related post-processing tools</b> like FaceFusion or face restoration models such as GFP-GAN and CodeFormer.
</p>

<p align="center">
  <video src="https://github.com/user-attachments/assets/90691318-311e-40b9-9bd9-62db83ab1492" width="768" autoplay loop muted playsinline></video>
  <br/>
  Comparison results between StableAvatar and state-of-the-art (SOTA) audio-driven avatar video generation models highlight the superior performance of StableAvatar in delivering <b>infinite-length, high-fidelity, identity-preserving avatar animation</b>.
</p>

## Key Features

*   **Infinite-Length Video Generation:** Create videos of virtually unlimited length.
*   **Identity Preservation:** Maintain the subject's identity throughout the video.
*   **High-Fidelity Results:** Generate realistic and detailed avatar videos.
*   **End-to-End Solution:** No need for post-processing tools.
*   **Audio-Driven Animation:** Synchronize avatar movements with audio input.
*   **End-to-End Video Diffusion Transformer:** StableAvatar is the first end-to-end video diffusion transformer that synthesizes high-quality videos without post-processing.

## Overview
<p align="center">
  <img src="assets/figures/framework.jpg" alt="model architecture" width="1280"/>
  </br>
  <i>The overview of the framework of StableAvatar.</i>
</p>

StableAvatar overcomes limitations in existing diffusion models for audio-driven avatar video generation by introducing a novel Time-step-aware Audio Adapter and an Audio Native Guidance Mechanism. This allows for generating long videos with seamless audio synchronization and consistent identity.

## News

*   `[2025-8-29]`: 🔥 StableAvatar public demo is now live on [Hugging Face Spaces](https://huggingface.co/spaces/YinmingHuang/StableAvatar).(Note: due to the long video generation time, the demo is currently accessible to **Hugging Face Pro** users only.)
*   `[2025-8-18]`: 🔥 StableAvatar can run on [ComfyUI](https://github.com/smthemex/ComfyUI_StableAvatar) in just 10 steps, making it 3x faster. Thanks @[smthemex](https://github.com/smthemex) for the contribution.
*   `[2025-8-16]`: 🔥 We release the finetuning codes and lora training/finetuning codes! Other codes will be public as soon as possible. Stay tuned!
*   `[2025-8-15]`: 🔥 StableAvatar can run on Gradio Interface. Thanks @[gluttony-10](https://space.bilibili.com/893892) for the contribution!
*   `[2025-8-15]`: 🔥 StableAvatar can run on [ComfyUI](https://github.com/smthemex/ComfyUI_StableAvatar). Thanks @[smthemex](https://github.com/smthemex) for the contribution.
*   `[2025-8-13]`: 🔥 Added changes to run StableAvatar on the new Blackwell series Nvidia chips, including the RTX 6000 Pro.
*   `[2025-8-11]`: 🔥 The project page, code, technical report and [a basic model checkpoint](https://huggingface.co/FrancisRing/StableAvatar/tree/main) are released. Further lora training codes, the evaluation dataset and StableAvatar-pro will be released very soon. Stay tuned!

## To-Do List

*   [x] StableAvatar-1.3B-basic
*   [x] Inference Code
*   [x] Data Pre-Processing Code (Audio Extraction)
*   [x] Data Pre-Processing Code (Vocal Separation)
*   [x] Training Code
*   [x] Full Finetuning Code
*   [x] Lora Training Code
*   [x] Lora Finetuning Code
*   [ ] Inference Code with Audio Native Guidance
*   [ ] StableAvatar-pro

## Quickstart Guide

The basic version of the model checkpoint (Wan2.1-1.3B-based) supports generating infinite-length videos at 480x832, 832x480, or 512x512 resolutions.

### Environment Setup

**For standard GPUs:**

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
# Optional to install flash_attn to accelerate attention computation
pip install flash_attn
```

**For Blackwell series chips:**

```bash
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
# Optional to install flash_attn to accelerate attention computation
pip install flash_attn
```

### Download Weights

If you encounter connection issues, use the mirror endpoint: `export HF_ENDPOINT=https://hf-mirror.com`. Then, download the weights:

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

### Audio Extraction

Extract audio from video (.mp4) to .wav:

```bash
python audio_extractor.py --video_path="path/test/video.mp4" --saved_audio_path="path/test/audio.wav"
```

### Vocal Separation

Separate vocals from audio (.wav):

```bash
pip install audio-separator[gpu]
python vocal_seperator.py --audio_separator_model_file="path/StableAvatar/checkpoints/Kim_Vocal_2.onnx" --audio_file_path="path/test/audio.wav" --saved_vocal_path="path/test/vocal.wav"
```

### Base Model Inference

Modify `inference.sh` with your configurations. Example:

```bash
bash inference.sh
```

Key parameters in `inference.sh`:

*   `--width` and `--height`: Output resolution.
*   `--output_dir`: Save path for the generated animation.
*   `--validation_reference_path`, `--validation_driven_audio_path`, `--validation_prompts`: Paths for reference image, audio, and prompts.
*   `--pretrained_model_name_or_path`, `--pretrained_wav2vec_path`, `--transformer_path`: Pretrained model paths.
*   `--sample_steps`, `--overlap_window_length`, `--clip_sample_n_frames`: Inference settings.
*   `--sample_text_guide_scale` and `--sample_audio_guide_scale`: CFG scales.

Example prompts: `[Description of first frame]-[Description of human behavior]-[Description of background (optional)]`.

Gradio interface:

```bash
python app.py
```

Examples are in `path/StableAvatar/examples`.

#### Tips

*   Use `transformer3d-square.pt` or `transformer3d-rec-vec.pt` from  `--transformer_path`.
*   Use `--GPU_memory_mode` to manage GPU memory.
*   Use `--ulysses_degree`, `--ring_degree`, and `--fsdp_dit` for multi-GPU inference.

To add the audio to the generated video:

```bash
ffmpeg -i video_without_audio.mp4 -i /path/audio.wav -c:v copy -c:a aac -shortest /path/output_with_audio.mp4
```

### Model Training

**Training tutorials will also be helpful if you’re looking to train a conditioned Video Diffusion Transformer (DiT) model, such as Wan2.1.**

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
*   `images`: Frames extracted from `.mp4` files.
*   `face_masks`: Face masks.  Use [StableAnimator repo](https://github.com/Francis-Rings/StableAnimator).
*   `lip_masks`: Lip masks.  Run `python lip_mask_extractor.py ...`.
*   `video_square_path.txt`, `video_rec_path.txt`, and `video_vec_path.txt`:  Paths to video data.

Extract frames (using ffmpeg):

```bash
ffmpeg -i raw_video_1.mp4 -q:v 1 -start_number 0 path/StableAvatar/talking_face_data/rec/speech/00001/images/frame_%d.png
```

Extract lip masks (requires mediapipe):

```bash
pip install mediapipe
python lip_mask_extractor.py --folder_root="path/StableAvatar/talking_face_data/rec/singing" --start=1 --end=500
```

Run training commands:

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

Parameters:

*   `CUDA_VISIBLE_DEVICES`: GPU devices.
*   `--pretrained_model_name_or_path`, `--pretrained_wav2vec_path`, `--output_dir`: Pretrained model paths and output directory.
*   `--train_data_square_dir`, `--train_data_rec_dir`, `--train_data_vec_dir`: Training data paths.
*   `--validation_reference_path`, `--validation_driven_audio_path`: Validation data paths.
*   `--video_sample_n_frames`: Frames per batch.
*   `--num_train_epochs`: Training epochs.

Training `StableAvatar` requires approximately 50GB of VRAM. If you train exclusively on 512x512 videos, the VRAM requirement is reduced to approximately 40GB.

For Wan2.1-14B training:

```bash
# Training StableAvatar on a mixed resolution setting (480x832, 832x480, and 512x512) in multiple machines
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./checkpoints/Wan2.1-I2V-14B-480P
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-720P --local-dir ./checkpoints/Wan2.1-I2V-14B-720P # Optional
bash train_14B.sh
```

For lora training:

```bash
# Training StableAvatar-1.3B on a mixed resolution setting (480x832 and 832x480) in a single machine
bash train_1B_rec_vec_lora.sh
# Training StableAvatar-1.3B on a mixed resolution setting (480x832 and 832x480) in multiple machines
bash train_1B_rec_vec_lora_64.sh
# Lora-Training StableAvatar-14B on a mixed resolution setting (480x832, 832x480, and 512x512) in multiple machines
bash train_14B_lora.sh
```
*   Modify `--rank` and `--network_alpha` to control the quality of your lora training/finetuning.

### Model Finetuning

Fully finetuning StableAvatar: Add `--transformer_path` to training scripts.

```bash
# Finetuning StableAvatar on a mixed resolution setting (480x832 and 832x480) in a single machine
bash train_1B_rec_vec.sh
# Finetuning StableAvatar on a mixed resolution setting (480x832 and 832x480) in multiple machines
bash train_1B_rec_vec_64.sh
```

Lora finetuning StableAvatar: Add `--transformer_path` to the lora training script.

```bash
# Lora-Finetuning StableAvatar-1.3B on a mixed resolution setting (480x832 and 832x480) in a single machine
bash train_1B_rec_vec_lora.sh
```

*   Modify `--rank` and `--network_alpha` to control the quality of your lora training/finetuning.

### VRAM Requirement and Runtime

For a 5-second video (480x832, fps=25), the basic model requires ~18GB VRAM and finishes in 3 minutes on a 4090 GPU.

**StableAvatar can theoretically generate hours of video without significant quality loss. You have the option to run the VAE decoder on the CPU.**

## Contact

Contact the author with questions or suggestions:

*   Email: francisshuyuan@gmail.com

If you find this work useful, please give a star ⭐ to this GitHub repository and cite it ❤️:

```bib
@article{tu2025stableavatar,
  title={StableAvatar: Infinite-Length Audio-Driven Avatar Video Generation},
  author={Tu, Shuyuan and Pan, Yueming and Huang, Yinming and Han, Xintong and Xing, Zhen and Dai, Qi and Luo, Chong and Wu, Zuxuan and Jiang Yu-Gang},
  journal={arXiv preprint arXiv:2508.08248},
  year={2025}
}
```