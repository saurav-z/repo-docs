# StableAvatar: Generate Infinite-Length Audio-Driven Avatar Videos

**Transform audio into stunning, infinite-length avatar videos with StableAvatar!**  Explore the cutting-edge technology behind StableAvatar, a groundbreaking approach to audio-driven avatar video generation, and unlock the potential for realistic and engaging visual content.  [View the original repository on GitHub](https://github.com/Francis-Rings/StableAvatar)

[![Project Page](https://img.shields.io/badge/Project-Page-Green)](https://francis-rings.github.io/StableAvatar)
[![Arxiv Paper](https://img.shields.io/badge/Paper-Arxiv-red)](https://arxiv.org/abs/2508.08248)
[![HuggingFace Model](https://img.shields.io/badge/HuggingFace-Model-orange)](https://huggingface.co/FrancisRing/StableAvatar/tree/main)
[![Hugging Face Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue)](https://huggingface.co/spaces/YinmingHuang/StableAvatar)
[![YouTube](https://img.shields.io/badge/YouTube-Watch-red?style=flat-square&logo=youtube)](https://www.youtube.com/watch?v=6lhvmbzvv3Y)
[![Bilibili](https://img.shields.io/badge/Bilibili-Watch-blue?style=flat-square&logo=bilibili)](https://www.bilibili.com/video/BV1hUt9z4EoQ)

*   **Authors:** Shuyuan Tu, Yueming Pan, Yinming Huang, Xintong Han, Zhen Xing, Qi Dai, Chong Luo, Zuxuan Wu, Yu-Gang Jiang
*   **Affiliations:** Fudan University, Microsoft Research Asia, Xi'an Jiaotong University, Tencent Inc

## Key Features:

*   **Infinite-Length Video Generation:** Synthesize videos of any length with StableAvatar's innovative framework.
*   **ID Preservation:** Maintain the identity of your avatar throughout the entire video, ensuring consistent and recognizable visuals.
*   **High-Fidelity Results:** Generate high-quality videos with natural audio synchronization and lifelike movements, avoiding post-processing.
*   **End-to-End Solution:** StableAvatar eliminates the need for face-swapping or restoration tools, providing a seamless workflow.
*   **Time-step-aware Audio Adapter:** Prevents error accumulation in long videos via time-step-aware modulation.
*   **Audio Native Guidance Mechanism:** Improves audio synchronization with diffusion's own evolving joint audio-latent prediction as a dynamic guidance signal.
*   **Dynamic Weighted Sliding-window Strategy:** Improves the smoothness of infinite-length videos.

## Showcase

<!-- Removed the table of videos and consolidated into a single video example for better readability on the main page. -->

<p align="center">
  <video src="https://github.com/user-attachments/assets/90691318-311e-40b9-9bd9-62db83ab1492" width="768" autoplay loop muted playsinline></video>
  <br/>
  <span>Comparison of StableAvatar and state-of-the-art (SOTA) audio-driven avatar video generation models demonstrate superior performance in delivering <b>infinite-length, high-fidelity, identity-preserving avatar animation</b>.</span>
</p>

## Overview

<!-- Moved overview image closer to the overview text -->

<p align="center">
  <img src="assets/figures/framework.jpg" alt="model architecture" width="1280"/>
  </br>
  <i>The overview of the framework of StableAvatar.</i>
</p>

StableAvatar addresses the limitations of existing diffusion models by introducing novel modules for superior performance. The core of StableAvatar's framework is a video diffusion transformer that generates infinite-length, high-quality videos directly from audio and a reference image.  The main reason why existing models struggle to generate long videos lies in their audio modeling. To address this, StableAvatar introduces a novel Time-step-aware Audio Adapter that prevents error accumulation via time-step-aware modulation. During inference, we propose a novel Audio Native Guidance Mechanism to further enhance the audio synchronization by leveraging the diffusion’s own evolving joint audio-latent prediction as a dynamic guidance signal. To enhance the smoothness of the infinite-length videos, we introduce a Dynamic Weighted Sliding-window Strategy that fuses latent over time. Experiments on benchmarks show the effectiveness of StableAvatar both qualitatively and quantitatively.

## News

*   **[2025-9-8]:** 🔥  Brand new demo released! Check out the generated videos on [YouTube](https://www.youtube.com/watch?v=GH4hrxIis3Q) and [Bilibili](https://www.bilibili.com/video/BV1jGYPzqEux).
*   **[2025-8-29]:** 🔥 StableAvatar public demo live on [Hugging Face Spaces](https://huggingface.co/spaces/YinmingHuang/StableAvatar). (Note: Due to the long video generation time, the demo is accessible to **Hugging Face Pro** users only.)
*   **[2025-8-18]:** 🔥 StableAvatar runs on [ComfyUI](https://github.com/smthemex/ComfyUI_StableAvatar) in just 10 steps, 3x faster. Thanks @[smthemex](https://github.com/smthemex) for the contribution.
*   **[2025-8-16]:** 🔥 Finetuning and LoRA training/finetuning codes released! Other codes will be public as soon as possible.
*   **[2025-8-15]:** 🔥 StableAvatar runs on Gradio Interface. Thanks @[gluttony-10](https://space.bilibili.com/893892) for the contribution!
*   **[2025-8-15]:** 🔥 StableAvatar runs on [ComfyUI](https://github.com/smthemex/ComfyUI_StableAvatar). Thanks @[smthemex](https://github.com/smthemex) for the contribution.
*   **[2025-8-13]:** 🔥 Added changes to run StableAvatar on the new Blackwell series Nvidia chips, including the RTX 6000 Pro.
*   **[2025-8-11]:** 🔥 Project page, code, technical report, and [basic model checkpoint](https://huggingface.co/FrancisRing/StableAvatar/tree/main) released. Further LoRA training codes, the evaluation dataset, and StableAvatar-pro are coming soon.

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

The basic version of the model checkpoint (Wan2.1-1.3B-based) supports generating <b>infinite-length videos at a 480x832 or 832x480 or 512x512 resolution</b>. Reduce the number of animated frames or the resolution of the output if you encounter insufficient memory issues.

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

If you encounter connection issues with Hugging Face, set the environment variable: `export HF_ENDPOINT=https://hf-mirror.com`.

Download weights manually:

```bash
pip install "huggingface_hub[cli]"
cd StableAvatar
mkdir checkpoints
huggingface-cli download FrancisRing/StableAvatar --local-dir ./checkpoints
```

Ensure the weights are organized as follows:

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

A sample configuration is provided in `inference.sh`.

```bash
bash inference.sh
```

Modify resolution using `--width` and `--height`.  Set the output path with `--output_dir`. Use `--validation_reference_path`, `--validation_driven_audio_path`, and `--validation_prompts` to specify the reference image, audio, and prompts, respectively.
Prompts: `[Description of first frame]-[Description of human behavior]-[Description of background (optional)]`.
Modify paths for `--pretrained_model_name_or_path`, `--pretrained_wav2vec_path`, and `--transformer_path`.
Configure inference steps with `--sample_steps`, `--overlap_window_length`, and `--clip_sample_n_frames`.  Recommended `--sample_steps` is [30-50] and `--overlap_window_length` is [5-15].
Use `--sample_text_guide_scale` and `--sample_audio_guide_scale` to control the Classify-Free-Guidance scale of text prompt and audio, where the recommended range is `[3-6]`.

Alternatively, launch a Gradio interface:

```bash
python app.py
```

Example videos are available in `path/StableAvatar/examples`.  Enjoy!

#### 💡 Tips

*   Use `--transformer_path` to switch between `transformer3d-square.pt` and `transformer3d-rec-vec.pt` models.
*   Use `--GPU_memory_mode` (`model_full_load`, `sequential_cpu_offload`, `model_cpu_offload_and_qfloat8`, `model_cpu_offload`) to manage GPU memory. `sequential_cpu_offload` approximately 3G VRAM, `model_cpu_offload` will approximately halve GPU memory usage.
*   Use `--ulysses_degree` and `--ring_degree` for multi-GPU inference.  Ensure `ulysses_degree * ring_degree = total GPU number / world-size`. Add `--fsdp_dit` for FSDP in DiT.

```bash
bash multiple_gpu_inference.sh
```

To add audio to the generated video:

```bash
ffmpeg -i video_without_audio.mp4 -i /path/audio.wav -c:v copy -c:a aac -shortest /path/output_with_audio.mp4
```

### 🧱 Model Training

**🔥🔥If you're looking to train a conditioned Video Diffusion Transformer (DiT) model, such as Wan2.1, this training tutorial will also be helpful.🔥🔥**

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

*   `square`: 512x512 videos.
*   `vec`: 480x832 videos.
*   `rec`: 832x480 videos.
*   Each folder contains subfolders for "speech", "singing", and "dancing".
*   `images`, `face_masks`, and `lip_masks` store the frames, face masks, and lip masks, respectively.
*   `sub_clip.mp4` is the video and `audio.wav` is the audio.
*   `video_square_path.txt`, `video_rec_path.txt`, and `video_vec_path.txt` list the video folder paths.

Extract frames from raw videos:

```bash
ffmpeg -i raw_video_1.mp4 -q:v 1 -start_number 0 path/StableAvatar/talking_face_data/rec/speech/00001/images/frame_%d.png
```

Extract face masks (refer to [StableAnimator repo](https://github.com/Francis-Rings/StableAnimator)).

Extract lip masks:

```bash
pip install mediapipe
python lip_mask_extractor.py --folder_root="path/StableAvatar/talking_face_data/rec/singing" --start=1 --end=500
```

For audio extraction, refer to the Audio Extraction section.

Training:

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

Modify parameters in training scripts.  `CUDA_VISIBLE_DEVICES` specifies GPUs.  `--pretrained_model_name_or_path`, `--pretrained_wav2vec_path`, and `--output_dir` are the paths for pre-trained Wan2.1-1.3B, Wav2Vec2.0, and the output checkpoint.  `--train_data_square_dir`, `--train_data_rec_dir`, and `--train_data_vec_dir` point to the data paths.  `--video_sample_n_frames` is the batch size.  `--num_train_epochs` is the training epoch.

The overall file structure of StableAvatar at training is shown as follows:
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
**Training requires approximately 50GB of VRAM for the mixed resolution.  512x512 training needs ~40GB VRAM.** Static backgrounds and clear audio are recommended for dataset quality.

For Wan2.1-14B:

```bash
# Training StableAvatar on a mixed resolution setting (480x832, 832x480, and 512x512) in multiple machines
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./checkpoints/Wan2.1-I2V-14B-480P
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-720P --local-dir ./checkpoints/Wan2.1-I2V-14B-720P # Optional
bash train_14B.sh
```
Use deepspeed stage-2, modifying the GPU configuration in `path/StableAvatar/accelerate_config/accelerate_config_machine_14B_multiple.yaml` and deepspeed config in `path/StableAvatar/deepspeed_config/zero_stage2_config.json`.

LoRA Training:

```bash
# Training StableAvatar-1.3B on a mixed resolution setting (480x832 and 832x480) in a single machine
bash train_1B_rec_vec_lora.sh
# Training StableAvatar-1.3B on a mixed resolution setting (480x832 and 832x480) in multiple machines
bash train_1B_rec_vec_lora_64.sh
# Lora-Training StableAvatar-14B on a mixed resolution setting (480x832, 832x480, and 512x512) in multiple machines
bash train_14B_lora.sh
```

Use `--rank` and `--network_alpha` to tune LoRA quality.

For 720P training, modify the dataloader height and width.

### 🧱 Model Finetuning

Finetune by adding `--transformer_path` (e.g., `"path/StableAvatar/checkpoints/StableAvatar-1.3B/transformer3d-square.pt"`) to the training scripts:

```bash
# Finetuning StableAvatar on a mixed resolution setting (480x832 and 832x480) in a single machine
bash train_1B_rec_vec.sh
# Finetuning StableAvatar on a mixed resolution setting (480x832 and 832x480) in multiple machines
bash train_1B_rec_vec_64.sh
```

LoRA finetuning similarly:

```bash
# Lora-Finetuning StableAvatar-1.3B on a mixed resolution setting (480x832 and 832x480) in a single machine
bash train_1B_rec_vec_lora.sh
```

Use `--rank` and `--network_alpha` to tune LoRA quality.

### 🧱 VRAM Requirement and Runtime

For 5s video (480x832, fps=25), the basic model requires ~18GB VRAM and finishes in 3 minutes on a 4090 GPU.

**StableAvatar can generate hours-long videos; however, the 3D VAE decoder can demand significant GPU memory, especially when decoding 10k+ frames. You can run the VAE decoder on CPU to mitigate this issue.**

## Contact

Feel free to contact the authors with suggestions or inquiries.

*   Email: francisshuyuan@gmail.com

**If you find this work helpful, please consider giving a star ⭐ to this GitHub repository and citing it ❤️:**

```bib
@article{tu2025stableavatar,
  title={StableAvatar: Infinite-Length Audio-Driven Avatar Video Generation},
  author={Tu, Shuyuan and Pan, Yueming and Huang, Yinming and Han, Xintong and Xing, Zhen and Dai, Qi and Luo, Chong and Wu, Zuxuan and Jiang Yu-Gang},
  journal={arXiv preprint arXiv:2508.08248},
  year={2025}
}
```