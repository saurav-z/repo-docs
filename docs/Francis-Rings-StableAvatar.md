# StableAvatar: Generate Infinite-Length Audio-Driven Avatar Videos

StableAvatar revolutionizes audio-driven avatar video generation, enabling the creation of high-quality, **infinite-length videos** directly from audio input! 

[Project Page](https://francis-rings.github.io/StableAvatar) | [Paper (arXiv)](https://arxiv.org/abs/2508.08248) | [Hugging Face Model](https://huggingface.co/FrancisRing/StableAvatar/tree/main) | [Hugging Face Demo](https://huggingface.co/spaces/YinmingHuang/StableAvatar) | [YouTube Demo](https://www.youtube.com/watch?v=6lhvmbzvv3Y) | [Bilibili Demo](https://www.bilibili.com/video/BV1hUt9z4EoQ)

**(Original Repo: [https://github.com/Francis-Rings/StableAvatar](https://github.com/Francis-Rings/StableAvatar))**

**Key Features:**

*   **Infinite-Length Video Generation:** Create videos of any duration.
*   **High Fidelity:**  Generates high-quality avatar animations.
*   **Identity Preservation:** Maintains the original identity throughout the video.
*   **End-to-End Synthesis:**  No need for post-processing tools like face-swapping or restoration.
*   **Time-step-aware Audio Adapter:** Addresses error accumulation for long videos.
*   **Audio Native Guidance Mechanism:** Improves audio synchronization.
*   **Dynamic Weighted Sliding-window Strategy:** Enhances video smoothness.

## Overview

StableAvatar overcomes the limitations of existing models by introducing innovative techniques like the Time-step-aware Audio Adapter and Audio Native Guidance, enabling the generation of long, high-quality videos with natural audio synchronization and identity consistency.

[Model Architecture Diagram (link to image in original README)]

## News

*   **\[2025-9-8]**: 🔥  Brand new demo released! Watch on [YouTube](https://www.youtube.com/watch?v=GH4hrxIis3Q) and [Bilibili](https://www.bilibili.com/video/BV1jGYPzqEux).
*   **\[2025-8-29]**: 🔥 Public demo live on [Hugging Face Spaces](https://huggingface.co/spaces/YinmingHuang/StableAvatar) (for Hugging Face Pro users).
*   **\[2025-8-18]**: 🔥 Runs on [ComfyUI](https://github.com/smthemex/ComfyUI_StableAvatar) for faster performance (thanks, @[smthemex](https://github.com/smthemex)!).
*   **\[2025-8-16]**: 🔥 Finetuning and LoRA training codes released!
*   **\[2025-8-15]**: 🔥 Gradio interface available (thanks, @[gluttony-10](https://space.bilibili.com/893892)!).
*   **\[2025-8-15]**: 🔥 Runs on [ComfyUI](https://github.com/smthemex/ComfyUI_StableAvatar) (thanks, @[smthemex](https://github.com/smthemex)!).
*   **\[2025-8-13]**: 🔥 Support added for new Blackwell series Nvidia chips.
*   **\[2025-8-11]**: 🔥 Project page, code, and [basic model checkpoint](https://huggingface.co/FrancisRing/StableAvatar/tree/main) released.

## Quickstart Guide

This quickstart guide outlines the key steps for getting started with StableAvatar.

### Environment Setup

```bash
# For older PyTorch versions, check original README.
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install flash_attn  # Optional: to accelerate attention computation
```

```bash
# For Blackwell series chips:
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install flash_attn  # Optional: to accelerate attention computation
```

### Download Weights

If you encounter issues, use a mirror: `export HF_ENDPOINT=https://hf-mirror.com`.
Download weights:

```bash
pip install "huggingface_hub[cli]"
cd StableAvatar
mkdir checkpoints
huggingface-cli download FrancisRing/StableAvatar --local-dir ./checkpoints
```

File structure:

```
StableAvatar/
├── ...
├── checkpoints
│   ├── Kim_Vocal_2.onnx
│   ├── wav2vec2-base-960h
│   ├── Wan2.1-Fun-V1.1-1.3B-InP
│   └── StableAvatar-1.3B
├── inference.py
├── ...
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

Modify parameters in `inference.sh` (resolution, prompts, paths, etc.)

```bash
bash inference.sh
```

Example prompt: `[Description of first frame]-[Description of human behavior]-[Description of background (optional)]`.
Recommended `--sample_steps` (30-50), `--overlap_window_length` (5-15), and `--sample_text_guide_scale` / `--sample_audio_guide_scale` (3-6).

### Gradio Interface

```bash
python app.py
```

### Post-Processing (Optional)

```bash
ffmpeg -i video_without_audio.mp4 -i /path/audio.wav -c:v copy -c:a aac -shortest /path/output_with_audio.mp4
```

### Tips

*   Adjust `--GPU_memory_mode` for GPU memory optimization.
*   Use multiple GPUs for faster inference.
*   Experiment with different model versions (Wan2.1-1.3B, etc.).

## Training

[Training Instructions (Detailed instructions from original README)]

## Contact

Feel free to reach out for suggestions or help!

Email: francisshuyuan@gmail.com

If you find this work helpful, give it a star and cite:

```bib
@article{tu2025stableavatar,
  title={StableAvatar: Infinite-Length Audio-Driven Avatar Video Generation},
  author={Tu, Shuyuan and Pan, Yueming and Huang, Yinming and Han, Xintong and Xing, Zhen and Dai, Qi and Luo, Chong and Wu, Zuxuan and Jiang Yu-Gang},
  journal={arXiv preprint arXiv:2508.08248},
  year={2025}
}