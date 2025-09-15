# StableAvatar: Generate Infinite-Length Audio-Driven Avatar Videos

**Create stunning, infinite-length avatar videos driven by audio with StableAvatar!** Explore the [original repository](https://github.com/Francis-Rings/StableAvatar) for the full experience.

[![Project Page](https://img.shields.io/badge/Project-Page-Green)](https://francis-rings.github.io/StableAvatar)
[![Arxiv](https://img.shields.io/badge/Paper-Arxiv-red)](https://arxiv.org/abs/2508.08248)
[![HuggingFace Model](https://img.shields.io/badge/HuggingFace-Model-orange)](https://huggingface.co/FrancisRing/StableAvatar/tree/main)
[![Hugging Face Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue)](https://huggingface.co/spaces/YinmingHuang/StableAvatar)
[![YouTube](https://img.shields.io/badge/YouTube-Watch-red?style=flat-square&logo=youtube)](https://www.youtube.com/watch?v=6lhvmbzvv3Y)
[![Bilibili](https://img.shields.io/badge/Bilibili-Watch-blue?style=flat-square&logo=bilibili)](https://www.bilibili.com/video/BV1hUt9z4EoQ)

StableAvatar is a groundbreaking end-to-end video diffusion transformer that produces high-quality, infinite-length avatar videos driven by audio input.  This innovative model ensures identity preservation and natural audio synchronization without the need for post-processing.

## Key Features:

*   **Infinite-Length Video Generation:**  Generate videos of any length without quality degradation.
*   **Identity Preservation:** Maintains the original identity throughout the entire video sequence.
*   **High-Fidelity Output:** Produces videos with superior quality and detail.
*   **End-to-End Solution:** Avoids the need for face-swapping or other post-processing tools.
*   **Audio-Driven Animation:** Synchronizes avatar movements seamlessly with the provided audio.
*   **Time-step-aware Audio Adapter**: Prevent error accumulation via time-step-aware modulation.
*   **Audio Native Guidance Mechanism**: Further enhance the audio synchronization by leveraging the diffusion’s own evolving joint audio-latent prediction as a dynamic guidance signal.
*   **Dynamic Weighted Sliding-window Strategy**: Enhance the smoothness of the infinite-length videos.

## Demo Videos

[Include the video thumbnails and corresponding captions from the original README here.]

## Overview

[Include the model architecture image from the original README here.]

StableAvatar overcomes the limitations of existing audio-driven avatar generation models by introducing novel techniques for audio modeling and video synthesis. It addresses the common issue of error accumulation in long videos by incorporating a Time-step-aware Audio Adapter.  Furthermore, the Audio Native Guidance Mechanism and Dynamic Weighted Sliding-window Strategy improve audio synchronization and video smoothness, leading to superior results.

## News

*   `[2025-9-8]`: 🔥  New demo videos are available on [YouTube](https://www.youtube.com/watch?v=GH4hrxIis3Q) and [Bilibili](https://www.bilibili.com/video/BV1jGYPzqEux).
*   `[2025-8-29]`: 🔥 StableAvatar public demo is live on [Hugging Face Spaces](https://huggingface.co/spaces/YinmingHuang/StableAvatar). (Note: access is currently limited to Hugging Face Pro users.)
*   `[2025-8-18]`: 🔥 StableAvatar runs on [ComfyUI](https://github.com/smthemex/ComfyUI_StableAvatar) in just 10 steps, thanks to @[smthemex](https://github.com/smthemex).
*   `[2025-8-16]`: 🔥 Finetuning and LoRA training/finetuning codes are released.
*   `[2025-8-15]`: 🔥 StableAvatar runs on a Gradio Interface and ComfyUI (contributions from @[gluttony-10](https://space.bilibili.com/893892) and @[smthemex](https://github.com/smthemex), respectively).
*   `[2025-8-13]`: 🔥 Support added for new Blackwell series Nvidia chips.
*   `[2025-8-11]`: 🔥 Project page, code, technical report, and basic model checkpoint are released.

## 🛠️ To-Do List

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

## 🔑 Quickstart

This quickstart guide will help you generate infinite-length videos at 480x832 or 832x480 or 512x512 resolution using the basic version of the model checkpoint (Wan2.1-1.3B-based). Reduce frame count or resolution for memory optimization.

### 🧱 Environment Setup

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

### 🧱 Download Weights

Download weights manually using Hugging Face CLI:

```bash
pip install "huggingface_hub[cli]"
cd StableAvatar
mkdir checkpoints
huggingface-cli download FrancisRing/StableAvatar --local-dir ./checkpoints
```

Organize weights within the project's file structure:

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

Extract audio from a video file (.mp4):

```bash
python audio_extractor.py --video_path="path/test/video.mp4" --saved_audio_path="path/test/audio.wav"
```

### 🧱 Vocal Separation

Separate vocals from audio for improved lip sync:

```bash
pip install audio-separator[gpu]
python vocal_seperator.py --audio_separator_model_file="path/StableAvatar/checkpoints/Kim_Vocal_2.onnx" --audio_file_path="path/test/audio.wav" --saved_vocal_path="path/test/vocal.wav"
```

### 🧱 Base Model Inference

Run the inference script using the provided example configuration:

```bash
bash inference.sh
```

Adjust resolution and paths in `inference.sh`:

*   `--width` and `--height`: Set animation resolution (512x512, 480x832, or 832x480).
*   `--output_dir`: Specify the output directory.
*   `--validation_reference_path`, `--validation_driven_audio_path`, and `--validation_prompts`: Set paths for the reference image, audio, and text prompts.

Prompts are crucial, and should follow the format: `[Description of first frame]-[Description of human behavior]-[Description of background (optional)]`.

*   `--pretrained_model_name_or_path`, `--pretrained_wav2vec_path`, and `--transformer_path`: Set paths for pretrained weights.
*   `--sample_steps`, `--overlap_window_length`, and `--clip_sample_n_frames`: Configure inference parameters.
*   `--sample_text_guide_scale` and `--sample_audio_guide_scale`: Control guidance scales.

Run the Gradio interface:

```bash
python app.py
```

See the example videos in `path/StableAvatar/examples`.

#### 💡 Tips

*   Use `transformer3d-square.pt` or `transformer3d-rec-vec.pt` by modifying `--transformer_path`.
*   Optimize GPU memory usage with `--GPU_memory_mode`.  Options are: `model_full_load`, `sequential_cpu_offload`, `model_cpu_offload_and_qfloat8`, and `model_cpu_offload`.
*   Use Multi-GPU inference via `--ulysses_degree` and `--ring_degree`.
*   Use ffmpeg to add audio to generated video using the following command:
   ```bash
   ffmpeg -i video_without_audio.mp4 -i /path/audio.wav -c:v copy -c:a aac -shortest /path/output_with_audio.mp4
   ```

### 🧱 Model Training

**🔥 Training guide for conditioned Video Diffusion Transformer (DiT) models, such as Wan2.1. 🔥**

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

*   `talking_face_data/square`, `talking_face_data/rec`, and `talking_face_data/vec`: Contain video data in 512x512, 480x832, and 832x480 resolutions, respectively.  Each folder contains subfolders `speech`, `singing`, and `dancing`.
*   `.png` frames:  Stored as `frame_i.png`.
*   `images`, `face_masks`, and `lip_masks`:  Contain RGB frames, face masks, and lip masks, respectively.
*   `sub_clip.mp4` and `audio.wav`: Corresponds to the RGB video and audio.
*   `video_square_path.txt`, `video_rec_path.txt`, and `video_vec_path.txt`: Text files listing video paths.

Extract frames from raw videos (speech):

```bash
ffmpeg -i raw_video_1.mp4 -q:v 1 -start_number 0 path/StableAvatar/talking_face_data/rec/speech/00001/images/frame_%d.png
```

Extract face masks (refer to [StableAnimator repo](https://github.com/Francis-Rings/StableAnimator)).

Extract lip masks:

```bash
pip install mediapipe
python lip_mask_extractor.py --folder_root="path/StableAvatar/talking_face_data/rec/singing" --start=1 --end=500
```

For details of corresponding audio, please refer to the Audio Extraction section.

Train your Wan2.1-1.3B-based StableAvatar:

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

*   CUDA\_VISIBLE\_DEVICES: GPU devices.
*   `--pretrained_model_name_or_path`, `--pretrained_wav2vec_path`, and `--output_dir`: Pretrained model, Wav2Vec2.0, and output paths.
*   `--train_data_square_dir`, `--train_data_rec_dir`, and `--train_data_vec_dir`:  Paths for video data lists.
*   `--validation_reference_path` and `--validation_driven_audio_path`: Validation assets.
*   `--video_sample_n_frames`: Batch processing size.
*   `--num_train_epochs`: Training epochs (set to infinite for manual termination).

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

**Training requires approximately 50GB of VRAM for mixed-resolution or 40GB for 512x512.** Static backgrounds and clear audio are recommended.

Train Wan2.1-14B-based StableAvatar:

```bash
# Training StableAvatar on a mixed resolution setting (480x832, 832x480, and 512x512) in multiple machines
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./checkpoints/Wan2.1-I2V-14B-480P
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-720P --local-dir ./checkpoints/Wan2.1-I2V-14B-720P # Optional
bash train_14B.sh
```

Use deepspeed stage-2. Modify GPU configuration in `accelerate_config/accelerate_config_machine_14B_multiple.yaml`. Deepspeed optimization and scheduler configuration in `deepspeed_config/zero_stage2_config.json`. Note that Wan2.1-1.3B-based StableAvatar is already capable of synthesizing infinite-length high quality avatar videos. The Wan2.1-14B backbone significantly increase the inference latency and GPU memory consumption during training, indicating limited efficiency in terms of performance-to-resource ratio.

Lora training:

```bash
# Training StableAvatar-1.3B on a mixed resolution setting (480x832 and 832x480) in a single machine
bash train_1B_rec_vec_lora.sh
# Training StableAvatar-1.3B on a mixed resolution setting (480x832 and 832x480) in multiple machines
bash train_1B_rec_vec_lora_64.sh
# Lora-Training StableAvatar-14B on a mixed resolution setting (480x832, 832x480, and 512x512) in multiple machines
bash train_14B_lora.sh
```

Control quality using `--rank` and `--network_alpha`.

For 720P training, modify the dataloader height and width in `train_1B_square.py`/`train_1B_vec_rec.py`/`train_14B.py`.

### 🧱 Model Finetuning

Finetune StableAvatar. Add `--transformer_path` to `train_1B_rec_vec.sh` or `train_1B_rec_vec_64.sh`.

```bash
# Finetuning StableAvatar on a mixed resolution setting (480x832 and 832x480) in a single machine
bash train_1B_rec_vec.sh
# Finetuning StableAvatar on a mixed resolution setting (480x832 and 832x480) in multiple machines
bash train_1B_rec_vec_64.sh
```

Lora Finetuning, add `--transformer_path` to the  `train_1B_rec_vec_lora.sh`:
```bash
# Lora-Finetuning StableAvatar-1.3B on a mixed resolution setting (480x832 and 832x480) in a single machine
bash train_1B_rec_vec_lora.sh
```

Control quality via `--rank` and `--network_alpha`.

### 🧱 VRAM Requirements and Runtime

5s video (480x832, fps=25) with the basic model (--GPU\_memory\_mode="model\_full\_load") requires ~18GB VRAM and finishes in ~3 minutes on a 4090 GPU.

**Theoretically, StableAvatar can generate hours of video without significant quality degradation; however, the 3D VAE decoder demands significant GPU memory, especially when decoding 10k+ frames.  You have the option to run the VAE decoder on CPU.**

## Contact

For suggestions or assistance, please contact:

*   Email: francisshuyuan@gmail.com

If you found this project helpful, please consider giving a star to this repository and citing it:

```bib
@article{tu2025stableavatar,
  title={StableAvatar: Infinite-Length Audio-Driven Avatar Video Generation},
  author={Tu, Shuyuan and Pan, Yueming and Huang, Yinming and Han, Xintong and Xing, Zhen and Dai, Qi and Luo, Chong and Wu, Zuxuan and Jiang Yu-Gang},
  journal={arXiv preprint arXiv:2508.08248},
  year={2025}
}
```