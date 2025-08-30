# StableAvatar: Generate Infinite-Length Audio-Driven Avatar Videos

**Create stunning, infinite-length avatar videos driven by audio with StableAvatar – the future of digital avatars!**  [View the original repository](https://github.com/Francis-Rings/StableAvatar)

[![Project Page](https://img.shields.io/badge/Project-Page-Green)](https://francis-rings.github.io/StableAvatar)
[![Arxiv](https://img.shields.io/badge/Paper-Arxiv-red)](https://arxiv.org/abs/2508.08248)
[![HuggingFace Model](https://img.shields.io/badge/HuggingFace-Model-orange)](https://huggingface.co/FrancisRing/StableAvatar/tree/main)
[![Hugging Face Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue)](https://huggingface.co/spaces/YinmingHuang/StableAvatar)
[![YouTube](https://img.shields.io/badge/YouTube-Watch-red?style=flat-square&logo=youtube)](https://www.youtube.com/watch?v=6lhvmbzvv3Y)
[![Bilibili](https://img.shields.io/badge/Bilibili-Watch-blue?style=flat-square&logo=bilibili)](https://www.bilibili.com/video/BV1hUt9z4EoQ)

StableAvatar is a groundbreaking innovation in audio-driven avatar video generation, enabling the creation of remarkably long and identity-preserving videos without post-processing.  It's the first end-to-end video diffusion transformer for this purpose!

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/d7eca208-6a14-46af-b337-fb4d2b66ba8d" width="320" controls loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/b5902ac8-8188-4da8-b9e6-6df280690ed1" width="320" controls loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/87faa5c1-a118-4a03-a071-45f18e6a0" width="320" controls loop></video>
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

*   **Infinite Length Videos:** Generate videos of any duration.
*   **Identity Preservation:** Maintains consistent character identity throughout.
*   **No Post-Processing:**  Achieved directly by StableAvatar, without external tools.

<p align="center">
  <video src="https://github.com/user-attachments/assets/90691318-311e-40b9-9bd9-62db83ab1492" width="768" autoplay loop muted playsinline></video>
  <br/>
  <span>Comparison results between StableAvatar and state-of-the-art (SOTA) audio-driven avatar video generation models highlight the superior performance of StableAvatar in delivering <b>infinite-length, high-fidelity, identity-preserving avatar animation</b>.</span>
</p>

## Key Features

*   **Time-step-aware Audio Adapter:** Prevents error accumulation in long videos.
*   **Audio Native Guidance Mechanism:** Enhances audio synchronization.
*   **Dynamic Weighted Sliding-window Strategy:** Smooths transitions for infinite-length videos.

## Overview

<p align="center">
  <img src="assets/figures/framework.jpg" alt="model architecture" width="1280"/>
  </br>
  <i>The overview of the framework of StableAvatar.</i>
</p>

StableAvatar addresses the limitations of existing models in generating long, synchronized avatar videos.  It utilizes a novel architecture that includes a Time-step-aware Audio Adapter, an Audio Native Guidance Mechanism, and a Dynamic Weighted Sliding-window Strategy. These innovations enable the creation of high-quality, infinite-length videos.

## What's New

*   **[2025-8-29]:** 🔥 StableAvatar public demo is now live on [Hugging Face Spaces](https://huggingface.co/spaces/YinmingHuang/StableAvatar) (Hugging Face Pro users only).
*   **[2025-8-18]:** 🔥 Runs on [ComfyUI](https://github.com/smthemex/ComfyUI_StableAvatar) in just 10 steps (3x faster).
*   **[2025-8-16]:** 🔥 Finetuning and LoRA training/finetuning codes released.
*   **[2025-8-15]:** 🔥 Runs on Gradio Interface and ComfyUI.
*   **[2025-8-13]:** 🔥 Updated for new Blackwell series Nvidia chips.
*   **[2025-8-11]:** 🔥 Project page, code, and model checkpoint released.

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

The basic version of the model checkpoint (Wan2.1-1.3B-based) supports generating infinite-length videos at a 480x832 or 832x480 or 512x512 resolution.
Please refer to the original repository for detailed instructions on setup and usage.

### Environment Setup

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
# Optional to install flash_attn to accelerate attention computation
pip install flash_attn
```
or
```bash
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
# Optional to install flash_attn to accelerate attention computation
pip install flash_attn
```
### Download Weights

```bash
pip install "huggingface_hub[cli]"
cd StableAvatar
mkdir checkpoints
huggingface-cli download FrancisRing/StableAvatar --local-dir ./checkpoints
```

### Audio Extraction, Vocal Separation, and Base Model Inference

Follow the detailed steps outlined in the original README.

## Training Information

Detailed instructions for model training, including data organization, training scripts, and environment setup, are provided in the original README. 

## Contact

For questions, suggestions, or if you find this project helpful:

*   **Email:** francisshuyuan@gmail.com

If you find our work useful, **please consider giving a star ⭐ to this github repository and citing it ❤️**:

```bib
@article{tu2025stableavatar,
  title={StableAvatar: Infinite-Length Audio-Driven Avatar Video Generation},
  author={Tu, Shuyuan and Pan, Yueming and Huang, Yinming and Han, Xintong and Xing, Zhen and Dai, Qi and Luo, Chong and Wu, Zuxuan and Jiang Yu-Gang},
  journal={arXiv preprint arXiv:2508.08248},
  year={2025}
}
```