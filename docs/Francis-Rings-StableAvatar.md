# StableAvatar: Generate Infinite-Length, Audio-Driven Avatar Videos

**Unleash the power of StableAvatar to create stunning, infinite-length avatar videos driven by audio, without face-swapping or post-processing!** [View the original repository](https://github.com/Francis-Rings/StableAvatar)

[![Project Page](https://img.shields.io/badge/Project-Page-Green)](https://francis-rings.github.io/StableAvatar)
[![Arxiv](https://img.shields.io/badge/Paper-Arxiv-red)](https://arxiv.org/abs/2508.08248)
[![HuggingFace Model](https://img.shields.io/badge/HuggingFace-Model-orange)](https://huggingface.co/FrancisRing/StableAvatar/tree/main)
[![YouTube Demo](https://img.shields.io/badge/YouTube-Watch-red?style=flat-square&logo=youtube)](https://www.youtube.com/watch?v=6lhvmbzvv3Y)
[![Bilibili Demo](https://img.shields.io/badge/Bilibili-Watch-blue?style=flat-square&logo=bilibili)](https://www.bilibili.com/video/BV1hUt9z4EoQ)

StableAvatar is a cutting-edge AI model that generates high-fidelity, identity-preserving avatar videos driven by audio.  It overcomes the limitations of existing models, offering the ability to synthesize **infinite-length** videos directly, without relying on face-related post-processing.

**Key Features:**

*   **Infinite-Length Video Generation:** Generate videos of any length without degradation in quality or identity consistency.
*   **Audio-Driven Animation:** Synchronizes avatar movements precisely to the provided audio input.
*   **Identity Preservation:**  Maintains the consistent identity of the avatar throughout the video.
*   **End-to-End Solution:**  Generates videos without the need for external face-swapping or restoration tools.
*   **High-Fidelity Results:** Produces videos with exceptional visual quality and natural movement.

**Demonstration Videos:**

*(Embedded video examples showcasing StableAvatar's capabilities)*

*(Insert example videos here - replace the placeholders)*

<video src="https://github.com/user-attachments/assets/d7eca208-6a14-46af-b337-fb4d2b66ba8d" width="320" controls loop></video>
<video src="https://github.com/user-attachments/assets/b5902ac4-8188-4da8-b9e6-6df280690ed1" width="320" controls loop></video>
<video src="https://github.com/user-attachments/assets/87faa5c1-a118-4a03-a071-45f18e87e6a0" width="320" controls loop></video>

<video src="https://github.com/user-attachments/assets/531eb413-8993-4f8f-9804-e3c5ec5794d4" width="320" controls loop></video>
<video src="https://github.com/user-attachments/assets/cdc603e2-df46-4cf8-a14e-1575053f996f" width="320" controls loop></video>
<video src="https://github.com/user-attachments/assets/7022dc93-f705-46e5-b8fc-3a3fb755795c" width="320" controls loop></video>

<video src="https://github.com/user-attachments/assets/0ba059eb-ff6f-4d94-80e6-f758c613b737" width="320" controls loop></video>
<video src="https://github.com/user-attachments/assets/03e6c1df-85c6-448d-b40d-aacb8add4e45" width="320" controls loop></video>
<video src="https://github.com/user-attachments/assets/90b78154-dda0-4eaa-91fd-b5485b718a7f" width="320" controls loop></video>
<br/>

*A comparison video demonstrating StableAvatar's superior performance.*

<p align="center">
  <video src="https://github.com/user-attachments/assets/90691318-311e-40b9-9bd9-62db83ab1492" width="768" autoplay loop muted playsinline></video>
  <br/>
</p>

## Overview

*(Insert the architectural diagram of StableAvatar)*

<p align="center">
  <img src="assets/figures/framework.jpg" alt="model architecture" width="1280"/>
</br>
  <i>The overview of the framework of StableAvatar.</i>
</p>

StableAvatar tackles the challenge of long-form audio-driven avatar video generation by introducing a novel Time-step-aware Audio Adapter and an Audio Native Guidance Mechanism, preventing error accumulation and enhancing audio synchronization. A Dynamic Weighted Sliding-window Strategy is implemented to enhance the smoothness of the infinite-length videos.

## News

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

The basic model checkpoint (Wan2.1-1.3B-based) supports generating infinite-length videos at 480x832, 832x480, or 512x512 resolutions. Reduce the number of animated frames or the output resolution if you encounter memory issues.

### 🧱 Environment Setup

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
# Optional to install flash_attn to accelerate attention computation
pip install flash_attn
```

### 🧱 Environment Setup for Blackwell Series Chips

```bash
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
# Optional to install flash_attn to accelerate attention computation
pip install flash_attn
```

### 🧱 Download Weights

If you have connection issues with Hugging Face, use the mirror endpoint: `export HF_ENDPOINT=https://hf-mirror.com`.

```bash
pip install "huggingface_hub[cli]"
cd StableAvatar
mkdir checkpoints
huggingface-cli download FrancisRing/StableAvatar --local-dir ./checkpoints
```

The file structure should be:

```
StableAvatar/
├── ...
├── checkpoints
│   ├── Kim_Vocal_2.onnx
│   ├── wav2vec2-base-960h
│   ├── Wan2.1-Fun-V1.1-1.3B-InP
│   └── StableAvatar-1.3B
├── ...
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

Run inference using `inference.sh`:

```bash
bash inference.sh
```

*   Modify `--width`, `--height` in `inference.sh` to set resolution (512x512, 480x832, or 832x480).
*   `--output_dir`:  Output directory for generated animation.
*   `--validation_reference_path`, `--validation_driven_audio_path`, `--validation_prompts`: Paths to the reference image, audio, and text prompts, respectively.
*   Prompts: Recommended format `[Description of first frame]-[Description of human behavior]-[Description of background (optional)]`.
*   `--pretrained_model_name_or_path`, `--pretrained_wav2vec_path`, `--transformer_path`:  Paths to pretrained model weights.
*   `--sample_steps`, `--overlap_window_length`, `--clip_sample_n_frames`:  Inference parameters.
    *   Recommended `--sample_steps`: \[30-50]
    *   Recommended `--overlap_window_length`: \[5-15]
*   `--sample_text_guide_scale`, `--sample_audio_guide_scale`: CFG scales.  Recommended range: \[3-6]. Increase audio CFG for better lip sync.

Gradio Interface:

```bash
python app.py
```

Example videos are provided in `path/StableAvatar/examples`.

#### 💡 Tips
- The two versions of Wan2.1-1.3B-based StableAvatar weights: `transformer3d-square.pt` and `transformer3d-rec-vec.pt`. You can modify `--transformer_path` in `inference.sh` to switch these two versions.

- You can change the loading mode of StableAvatar by modifying "--GPU_memory_mode" in `inference.sh`. The options of "--GPU_memory_mode" are `model_full_load`, `sequential_cpu_offload`, `model_cpu_offload_and_qfloat8`, and `model_cpu_offload`. In particular, when you set `--GPU_memory_mode` to `sequential_cpu_offload`, the total GPU memory consumption is approximately 3G with slower inference speed.
Setting `--GPU_memory_mode` to `model_cpu_offload` can significantly cut GPU memory usage, reducing it by roughly half compared to `model_full_load` mode.

- If you have multiple Gpus, you can run Multi-GPU inference to speed up by modifying "--ulysses_degree" and "--ring_degree" in `inference.sh`. For example, if you have 8 GPUs, you can set `--ulysses_degree=4` and `--ring_degree=2`. Notably, you have to ensure ulysses_degree*ring_degree=total GPU number/world-size. Moreover, you can also add `--fsdp_dit` in `inference.sh` to activate FSDP in DiT to further reduce GPU memory consumption.
You can fun the following command:
```
bash multiple_gpu_inference.sh
```

Use ffmpeg to add audio to the output:

```bash
ffmpeg -i video_without_audio.mp4 -i /path/audio.wav -c:v copy -c:a aac -shortest /path/output_with_audio.mp4
```

### 🧱 Model Training

**Training Tutorial for Video Diffusion Transformer (DiT) models (Wan2.1) and StableAvatar.**

Dataset Structure:

```
talking_face_data/
├── rec
│   │  ├──speech
│   │  │  ├──00001
│   │  │  │  ├──sub_clip.mp4
│   │  │  │  ├──audio.wav
│   │  │  │  ├──images
│   │  │  │  ├──frame_0.png
│   │  │  │  ├──frame_1.png
│   │  │  │  ├──frame_2.png
│   │  │  │  ├──...
│   │  │  │  ├──face_masks
│   │  │  │  ├──frame_0.png
│   │  │  │  ├──frame_1.png
│   │  │  │  ├──frame_2.png
│   │  │  │  ├──...
│   │  │  │  ├──lip_masks
│   │  │  │  ├──frame_0.png
│   │  │  │  ├──frame_1.png
│   │  │  │  ├──frame_2.png
│   │  │  │  ├──...
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
*   Mixed-resolution videos:  `talking_face_data/square` (512x512), `talking_face_data/vec` (480x832), and `talking_face_data/rec` (832x480).
*   Each subfolder contains different types of videos (speech, singing, and dancing).
*   `.png` image files are named `frame_i.png`.
*   `images`, `face_masks`, and `lip_masks`:  RGB frames, face masks, and lip masks, respectively.
*   `sub_clip.mp4` and `audio.wav`: Video and audio files.
*   `video_square_path.txt`, `video_rec_path.txt`, `video_vec_path.txt`:  Text files listing the paths to the video data.

Use ffmpeg to extract frames from videos:

```bash
ffmpeg -i raw_video_1.mp4 -q:v 1 -start_number 0 path/StableAvatar/talking_face_data/rec/speech/00001/images/frame_%d.png
```

*   For human face masks, refer to [StableAnimator repo](https://github.com/Francis-Rings/StableAnimator).
*   For human lip masks:

```bash
pip install mediapipe
python lip_mask_extractor.py --folder_root="path/StableAvatar/talking_face_data/rec/singing" --start=1 --end=500
```

*   For Audio Extraction, refer to the Audio Extraction section.

Training Commands:

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
Parameter details for `train_1B_square.sh` and `train_1B_rec_vec.sh`:
*  `CUDA_VISIBLE_DEVICES`: gpu devices
*  `--pretrained_model_name_or_path`, `--pretrained_wav2vec_path`, and `--output_dir` : pretrained Wan2.1-1.3B path, pretrained Wav2Vec2.0 path, and the checkpoint saved path of the trained StableAvatar.
* `--train_data_square_dir`, `--train_data_rec_dir`, and `--train_data_vec_dir` are the paths of `video_square_path.txt`, `video_rec_path.txt`, and `video_vec_path.txt`, respectively.
*   `--validation_reference_path` and `--validation_driven_audio_path`: Paths to the validation reference image and validation driven audio.
*   `--video_sample_n_frames`:  The number of frames processed in a single batch.
*   `--num_train_epochs`: The training epoch number.
For the parameter details of `train_1B_square_64.sh` and `train_1B_rec_vec_64.sh`, we set the GPU configuration in `path/StableAvatar/accelerate_config/accelerate_config_machine_1B_multiple.yaml`. In my setting, the training setup consists of 8 nodes, each equipped with 8 NVIDIA A100 80GB GPUs, for training StableAvatar.

The overall file structure of StableAvatar at training is shown as follows:
```
StableAvatar/
├── ...
├── checkpoints
│   ├── Kim_Vocal_2.onnx
│   ├── wav2vec2-base-960h
│   ├── Wan2.1-Fun-V1.1-1.3B-InP
│   └── StableAvatar-1.3B
├── ...
```

**Important Notes:**

*   Training StableAvatar requires approximately 50GB of VRAM (mixed resolution) or 40GB (512x512 only).
*   Use static backgrounds in training videos for accurate loss calculation.
*   Use clear audio without excessive noise.

Wan2.1-14B-based StableAvatar training:
```bash
# Training StableAvatar on a mixed resolution setting (480x832, 832x480, and 512x512) in multiple machines
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./checkpoints/Wan2.1-I2V-14B-480P
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-720P --local-dir ./checkpoints/Wan2.1-I2V-14B-720P # Optional
bash train_14B.sh
```
Deepspeed configurations in `path/StableAvatar/deepspeed_config/zero_stage2_config.json`.

Lora Training:
```bash
# Training StableAvatar-1.3B on a mixed resolution setting (480x832 and 832x480) in a single machine
bash train_1B_rec_vec_lora.sh
# Training StableAvatar-1.3B on a mixed resolution setting (480x832 and 832x480) in multiple machines
bash train_1B_rec_vec_lora_64.sh
# Lora-Training StableAvatar-14B on a mixed resolution setting (480x832, 832x480, and 512x512) in multiple machines
bash train_14B_lora.sh
```
You can modify `--rank` and `--network_alpha` to control the quality of your lora training/finetuning.

To train 720P Wan2.1-1.3B-based or Wan2.1-14B-based StableAvatar, you can directly modify the height and width of the dataloader (480p-->720p) in `train_1B_square.py`/`train_1B_vec_rec.py`/`train_14B.py`.

### 🧱 Model Finetuning

Finetune:  Add `--transformer_path` to the training script:
```bash
# Finetuning StableAvatar on a mixed resolution setting (480x832 and 832x480) in a single machine
bash train_1B_rec_vec.sh --transformer_path="path/StableAvatar/checkpoints/StableAvatar-1.3B/transformer3d-square.pt"
# Finetuning StableAvatar on a mixed resolution setting (480x832 and 832x480) in multiple machines
bash train_1B_rec_vec_64.sh --transformer_path="path/StableAvatar/checkpoints/StableAvatar-1.3B/transformer3d-square.pt"
```
Lora Finetuning:  Add `--transformer_path`:

```bash
# Lora-Finetuning StableAvatar-1.3B on a mixed resolution setting (480x832 and 832x480) in a single machine
bash train_1B_rec_vec_lora.sh --transformer_path="path/StableAvatar/checkpoints/StableAvatar-1.3B/transformer3d-square.pt"
```

Modify `--rank` and `--network_alpha` for Lora quality.

### 🧱 VRAM Requirement and Runtime

For a 5-second video (480x832, 25 fps), the basic model requires approximately 18GB VRAM and finishes in 3 minutes on a 4090 GPU.

**StableAvatar can theoretically generate hours of video. You can run the VAE decoder on the CPU if memory is limited.**

## Contact

For suggestions or help, contact:  francisshuyuan@gmail.com

If you find our work useful, consider giving a star to this repository and citing it:

```bib
@article{tu2025stableavatar,
  title={StableAvatar: Infinite-Length Audio-Driven Avatar Video Generation},
  author={Tu, Shuyuan and Pan, Yueming and Huang, Yinming and Han, Xintong and Xing, Zhen and Dai, Qi and Luo, Chong and Wu, Zuxuan and Jiang Yu-Gang},
  journal={arXiv preprint arXiv:2508.08248},
  year={2025}
}
```