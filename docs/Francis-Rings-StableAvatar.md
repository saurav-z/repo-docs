# StableAvatar: Generate Infinite-Length Audio-Driven Avatar Videos

**Create stunning, endlessly flowing avatar videos from just audio with StableAvatar!** [Explore the original repo](https://github.com/Francis-Rings/StableAvatar)

[![Project Page](https://img.shields.io/badge/Project-Page-Green)](https://francis-rings.github.io/StableAvatar)
[![Arxiv](https://img.shields.io/badge/Paper-Arxiv-red)](https://arxiv.org/abs/2508.08248)
[![HuggingFace Model](https://img.shields.io/badge/HuggingFace-Model-orange)](https://huggingface.co/FrancisRing/StableAvatar/tree/main)
[![Hugging Face Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue)](https://huggingface.co/spaces/YinmingHuang/StableAvatar)
[![YouTube](https://img.shields.io/badge/YouTube-Watch-red?style=flat-square&logo=youtube)](https://www.youtube.com/watch?v=6lhvmbzvv3Y)
[![Bilibili](https://img.shields.io/badge/Bilibili-Watch-blue?style=flat-square&logo=bilibili)](https://www.bilibili.com/video/BV1hUt9z4EoQ)

StableAvatar is a groundbreaking approach to audio-driven avatar video generation, enabling the creation of **infinite-length, high-fidelity, and identity-preserving videos** without the need for post-processing. This innovative system utilizes a novel Time-step-aware Audio Adapter and an Audio Native Guidance Mechanism to ensure seamless audio synchronization and identity consistency throughout the generated videos.

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
  <span>StableAvatar generates audio-driven avatar videos that can run for **infinite length** and are <b>ID-preserving</b>, without the need for post-processing tools.</span>
</p>

<p align="center">
  <video src="https://github.com/user-attachments/assets/90691318-311e-40b9-9bd9-62db83ab1492" width="768" autoplay loop muted playsinline></video>
  <br/>
  <span>Comparison results between StableAvatar and state-of-the-art (SOTA) audio-driven avatar video generation models highlight the superior performance of StableAvatar in delivering <b>infinite-length, high-fidelity, identity-preserving avatar animation</b>.</span>
</p>


## Key Features

*   **Infinite-Length Video Generation:** Create videos of any duration without degradation.
*   **High-Fidelity Results:** Produce realistic and detailed avatar animations.
*   **Identity Preservation:** Maintain a consistent identity throughout the video.
*   **End-to-End Solution:** No post-processing or face-swapping required.
*   **Time-Step-Aware Audio Adapter:** Prevent error accumulation for long videos.
*   **Audio Native Guidance Mechanism:** Enhances audio synchronization.
*   **Dynamic Weighted Sliding-window Strategy:** Ensures smooth transitions.

## Overview

<p align="center">
  <img src="assets/figures/framework.jpg" alt="model architecture" width="1280"/>
  </br>
  <i>The overview of the framework of StableAvatar.</i>
</p>

StableAvatar addresses the limitations of current diffusion models by introducing innovative solutions for long-form video generation. It overcomes the issue of audio modeling in existing models, preventing latent distribution error accumulation and ensuring superior performance. The core innovations include a Time-step-aware Audio Adapter, an Audio Native Guidance Mechanism, and a Dynamic Weighted Sliding-window Strategy, which collectively enable the generation of high-quality, long-lasting avatar videos.

## What's New

*   `[2025-9-8]`: 🔥 Brand new demo released! Videos available on [YouTube](https://www.youtube.com/watch?v=GH4hrxIis3Q) and [Bilibili](https://www.bilibili.com/video/BV1jGYPzqEux).
*   `[2025-8-29]`: 🔥 Public demo now live on [Hugging Face Spaces](https://huggingface.co/spaces/YinmingHuang/StableAvatar) (Pro users only).
*   `[2025-8-18]`: 🔥 StableAvatar runs on [ComfyUI](https://github.com/smthemex/ComfyUI_StableAvatar).
*   `[2025-8-16]`: 🔥 Finetuning and LoRA training/finetuning codes released.
*   `[2025-8-15]`: 🔥 StableAvatar runs on Gradio interface.
*   `[2025-8-15]`: 🔥 StableAvatar runs on [ComfyUI](https://github.com/smthemex/ComfyUI_StableAvatar).
*   `[2025-8-13]`: 🔥 Support for new Blackwell series Nvidia chips.
*   `[2025-8-11]`: 🔥 Project page, code, technical report, and basic model checkpoint released.

## To-Do List

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

## Quickstart

### Environment Setup

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install flash_attn  # Optional to accelerate attention computation
```

### Environment Setup for Blackwell Series Chips

```bash
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install flash_attn  # Optional to accelerate attention computation
```

### Download Weights

```bash
pip install "huggingface_hub[cli]"
cd StableAvatar
mkdir checkpoints
huggingface-cli download FrancisRing/StableAvatar --local-dir ./checkpoints
```

The overall file structure should be:

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

```bash
python audio_extractor.py --video_path="path/test/video.mp4" --saved_audio_path="path/test/audio.wav"
```

### Vocal Separation

```bash
pip install audio-separator[gpu]
python vocal_seperator.py --audio_separator_model_file="path/StableAvatar/checkpoints/Kim_Vocal_2.onnx" --audio_file_path="path/test/audio.wav" --saved_vocal_path="path/test/vocal.wav"
```

### Base Model Inference

Modify `inference.sh` with the following:

*   `--width` and `--height`: Set the animation resolution (512x512, 480x832, 832x480).
*   `--output_dir`:  Specify the output path.
*   `--validation_reference_path`, `--validation_driven_audio_path`, `--validation_prompts`:  Paths for the reference image, audio, and prompts.
*   `--pretrained_model_name_or_path`, `--pretrained_wav2vec_path`, `--transformer_path`: Model weight paths.
*   `--sample_steps`, `--overlap_window_length`, `--clip_sample_n_frames`: Inference parameters. Recommended `--sample_steps` is [30-50] and `--overlap_window_length` is [5-15].
*   `--sample_text_guide_scale` and `--sample_audio_guide_scale`: CFG scales. Recommended range is `[3-6]`.

Run the inference with:

```bash
bash inference.sh
```

Additionally, you can also launch a Gradio interface:
```bash
python app.py
```
#### Tips

*   Wan2.1-1.3B-based StableAvatar weights have two versions: `transformer3d-square.pt` and `transformer3d-rec-vec.pt`, which are trained on two video datasets in two different resolution settings. Two versions both support generating audio-driven avatar video at three different resolution settings: 512x512, 480x832, and 832x480. You can modify `--transformer_path` in `inference.sh` to switch these two versions.
*   If you have limited GPU resources, you can change the loading mode of StableAvatar by modifying "--GPU_memory_mode" in `inference.sh`. The options of "--GPU_memory_mode" are `model_full_load`, `sequential_cpu_offload`, `model_cpu_offload_and_qfloat8`, and `model_cpu_offload`. In particular, when you set `--GPU_memory_mode` to `sequential_cpu_offload`, the total GPU memory consumption is approximately 3G with slower inference speed.
*   If you have multiple Gpus, you can run Multi-GPU inference to speed up by modifying "--ulysses_degree" and "--ring_degree" in `inference.sh`. For example, if you have 8 GPUs, you can set `--ulysses_degree=4` and `--ring_degree=2`. Notably, you have to ensure ulysses_degree*ring_degree=total GPU number/world-size. Moreover, you can also add `--fsdp_dit` in `inference.sh` to activate FSDP in DiT to further reduce GPU memory consumption.
*   If you want to obtain the high quality MP4 file with audio, we recommend you to leverage ffmpeg on the <b>output_path</b> as follows:
```
ffmpeg -i video_without_audio.mp4 -i /path/audio.wav -c:v copy -c:a aac -shortest /path/output_with_audio.mp4
```

### Model Training

Follow the dataset format as follows:

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

*   Videos are stored in `talking_face_data/square`, `talking_face_data/vec`, and `talking_face_data/rec`.
*   Each folder contains subfolders for different video types (speech, singing, and dancing).
*   Files are named consistently (e.g., `frame_0.png`).
*   Use `ffmpeg` to extract frames and audio if needed.
*   Human face and lip masks are also required.

Run the appropriate training script:

```bash
# Single resolution (512x512), single machine
bash train_1B_square.sh
# Single resolution (512x512), multi-machine
bash train_1B_square_64.sh
# Mixed resolutions (480x832 and 832x480), single machine
bash train_1B_rec_vec.sh
# Mixed resolutions (480x832 and 832x480), multi-machine
bash train_1B_rec_vec_64.sh
```

See the scripts for detailed parameter descriptions.

### Model Finetuning

To finetune, add `--transformer_path` to your training command:

```bash
# Finetuning StableAvatar, single machine
bash train_1B_rec_vec.sh --transformer_path="path/StableAvatar/checkpoints/StableAvatar-1.3B/transformer3d-square.pt"
```

### VRAM Requirement and Runtime

*   Basic model (480x832, 25 fps): ~18GB VRAM, 3 minutes on a 4090 GPU.
*   Runtime depends on the number of steps and the length of the video.

## Contact

For any questions, suggestions, or support, please reach out to:

*   Email: francisshuyuan@gmail.com

If you find this work helpful, please consider giving a star to this repository and citing it:

```bib
@article{tu2025stableavatar,
  title={Stableavatar: Infinite-length audio-driven avatar video generation},
  author={Tu, Shuyuan and Pan, Yueming and Huang, Yinming and Han, Xintong and Xing, Zhen and Dai, Qi and Luo, Chong and Wu, Zuxuan and Jiang, Yu-Gang},
  journal={arXiv preprint arXiv:2508.08248},
  year={2025}
}
```