# StableAvatar: Generate Infinite-Length Audio-Driven Avatar Videos

StableAvatar empowers you to create high-quality, **infinite-length** avatar videos from just audio! Check out the original repo for more details: [https://github.com/Francis-Rings/StableAvatar](https://github.com/Francis-Rings/StableAvatar).

[![Project Page](https://img.shields.io/badge/Project-Page-Green)](https://francis-rings.github.io/StableAvatar)
[![Paper](https://img.shields.io/badge/Paper-Arxiv-red)](https://arxiv.org/abs/2508.08248)
[![HuggingFace Model](https://img.shields.io/badge/HuggingFace-Model-orange)](https://huggingface.co/FrancisRing/StableAvatar/tree/main)
[![Hugging Face Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue)](https://huggingface.co/spaces/YinmingHuang/StableAvatar)
[![YouTube](https://img.shields.io/badge/YouTube-Watch-red?style=flat-square&logo=youtube)](https://www.youtube.com/watch?v=6lhvmbzvv3Y)
[![Bilibili](https://img.shields.io/badge/Bilibili-Watch-blue?style=flat-square&logo=bilibili)](https://www.bilibili.com/video/BV1hUt9z4EoQ)

## Key Features

*   **Infinite-Length Video Generation:** Produce avatar videos of arbitrary length without quality degradation.
*   **Identity Preservation:** Maintain consistent identity throughout the entire video.
*   **High Fidelity:** Generate videos with impressive visual quality.
*   **End-to-End Synthesis:**  No post-processing tools (e.g., face swapping, restoration) are required.
*   **Audio-Driven:** Generate videos synchronized to an audio track.
*   **Versatile Resolution Support:**  Generate videos at various resolutions (512x512, 480x832, 832x480).
*   **Multiple GPU Support** Support multi-GPU inference to speed up.

## Demo Videos

[Include the example videos here, ensuring they are responsive and display correctly. Use the same code provided.]

## Overview

[Include the framework image and description, maintaining clarity.]

StableAvatar is a groundbreaking video diffusion transformer designed to overcome limitations in existing audio-driven avatar video generation models. Unlike previous approaches, StableAvatar synthesizes high-quality, infinite-length videos without the need for post-processing. It introduces innovations such as a Time-step-aware Audio Adapter and an Audio Native Guidance Mechanism to prevent error accumulation and enhance audio synchronization. A Dynamic Weighted Sliding-window Strategy is also utilized. This approach results in superior performance in terms of length, fidelity, and identity preservation.

## News

*   **\[2025-8-29]:** 🔥 Public demo is now live on [Hugging Face Spaces](https://huggingface.co/spaces/YinmingHuang/StableAvatar) (for Hugging Face Pro users).
*   **\[2025-8-18]:** 🔥 StableAvatar can run on [ComfyUI](https://github.com/smthemex/ComfyUI_StableAvatar) in just 10 steps, making it 3x faster. Thanks @[smthemex](https://github.com/smthemex).
*   **\[2025-8-16]:** 🔥 Finetuning and lora training/finetuning codes are released.
*   **\[2025-8-15]:** 🔥 StableAvatar can run on Gradio Interface. Thanks @[gluttony-10](https://space.bilibili.com/893892).
*   **\[2025-8-15]:** 🔥 StableAvatar can run on [ComfyUI](https://github.com/smthemex/ComfyUI_StableAvatar). Thanks @[smthemex](https://github.com/smthemex).
*   **\[2025-8-13]:** 🔥 Added changes to run StableAvatar on the new Blackwell series Nvidia chips, including the RTX 6000 Pro.
*   **\[2025-8-11]:** 🔥 Project page, code, technical report, and [a basic model checkpoint](https://huggingface.co/FrancisRing/StableAvatar/tree/main) are released.

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

If you encounter connection issues with Hugging Face, you can utilize the mirror endpoint by setting the environment variable: `export HF_ENDPOINT=https://hf-mirror.com`.

```bash
pip install "huggingface_hub[cli]"
cd StableAvatar
mkdir checkpoints
huggingface-cli download FrancisRing/StableAvatar --local-dir ./checkpoints
```

All the weights should be organized in models as follows:
The overall file structure of this project should be organized as follows:
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

```bash
bash inference.sh
```

Modify parameters in `inference.sh` such as `--width`, `--height`, `--output_dir`, `--validation_reference_path`, `--validation_driven_audio_path`, `--validation_prompts`, `--pretrained_model_name_or_path`, `--pretrained_wav2vec_path`, `--transformer_path`, `--sample_steps`, `--overlap_window_length`, `--clip_sample_n_frames`, `--sample_text_guide_scale`, and `--sample_audio_guide_scale` to customize your video generation. Prompts are crucial; recommended format: `[Description of first frame]-[Description of human behavior]-[Description of background (optional)]`.

To launch a Gradio interface:

```bash
python app.py
```

### 💡 Tips

*   Use `transformer3d-square.pt` or `transformer3d-rec-vec.pt` models.
*   Adjust `--GPU_memory_mode` to manage VRAM usage (`model_full_load`, `sequential_cpu_offload`, `model_cpu_offload_and_qfloat8`, and `model_cpu_offload`).
*   Use `--ulysses_degree` and `--ring_degree` for multi-GPU inference and add `--fsdp_dit` for FSDP.
*   Use `ffmpeg` to add audio to video.

#### Training and Finetuning

Detailed instructions for data organization, training, and finetuning are provided, including dataset structure, example commands for training and finetuning (single/multi-GPU), and parameters.  Key points:

*   Follow the specific data structure requirements for training.
*   Use `ffmpeg` to extract frames if needed.
*   Extract face and lip masks for datasets.
*   Adapt the appropriate `train_*.sh` scripts for your setup.
*   Modify training parameters based on the instructions.

#### Performance and Memory

*   5s video (480x832, fps=25) requires approximately 18GB VRAM and finishes in 3 minutes on a 4090 GPU (using the basic model).
*   Theoretically, StableAvatar can generate hours of video, but the 3D VAE decoder demands significant GPU memory, so consider running the decoder on the CPU.

## Contact

Email: francisshuyuan@gmail.com

If you find our work useful, **please consider giving a star ⭐ to this github repository and citing it ❤️**:

```bib
@article{tu2025stableavatar,
  title={StableAvatar: Infinite-Length Audio-Driven Avatar Video Generation},
  author={Tu, Shuyuan and Pan, Yueming and Huang, Yinming and Han, Xintong and Xing, Zhen and Dai, Qi and Luo, Chong and Wu, Zuxuan and Jiang Yu-Gang},
  journal={arXiv preprint arXiv:2508.08248},
  year={2025}
}
```