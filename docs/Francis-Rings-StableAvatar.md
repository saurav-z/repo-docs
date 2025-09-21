# StableAvatar: Generate Infinite-Length Audio-Driven Avatar Videos

**Create stunning, infinite-length avatar videos effortlessly using just audio and a reference image with StableAvatar!**  [Go to the Original Repo](https://github.com/Francis-Rings/StableAvatar)

[![Project Page](https://img.shields.io/badge/Project-Page-Green)](https://francis-rings.github.io/StableAvatar)
[![Arxiv](https://img.shields.io/badge/Paper-Arxiv-red)](https://arxiv.org/abs/2508.08248)
[![Hugging Face Model](https://img.shields.io/badge/HuggingFace-Model-orange)](https://huggingface.co/FrancisRing/StableAvatar/tree/main)
[![Hugging Face Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue)](https://huggingface.co/spaces/YinmingHuang/StableAvatar)
[![YouTube](https://img.shields.io/badge/YouTube-Watch-red?style=flat-square&logo=youtube)](https://www.youtube.com/watch?v=6lhvmbzvv3Y)
[![Bilibili](https://img.shields.io/badge/Bilibili-Watch-blue?style=flat-square&logo=bilibili)](https://www.bilibili.com/video/BV1hUt9z4EoQ)

StableAvatar, developed by Shuyuan Tu et al., is a groundbreaking video diffusion transformer that allows you to generate high-quality, infinite-length avatar videos driven by audio. It excels at maintaining identity and delivering natural audio synchronization without the need for post-processing.

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
  StableAvatar generates audio-driven avatar videos capable of synthesizing <b>infinite-length</b> and <b>ID-preserving videos</b>. All videos are <b>directly synthesized by StableAvatar without the use of any face-related post-processing tools</b>, such as the face-swapping tool FaceFusion or face restoration models like GFP-GAN and CodeFormer.
</p>

<p align="center">
  <video src="https://github.com/user-attachments/assets/90691318-311e-40b9-9bd9-62db83ab1492" width="768" autoplay loop muted playsinline></video>
  <br/>
  <span>Comparison results between StableAvatar and state-of-the-art (SOTA) audio-driven avatar video generation models highlight the superior performance of StableAvatar in delivering <b>infinite-length, high-fidelity, identity-preserving avatar animation</b>.</span>
</p>


## Key Features

*   **Infinite-Length Video Generation:** Generate videos of any duration.
*   **Identity Preservation:** Maintain consistent identity throughout the video.
*   **End-to-End Solution:** No need for external post-processing tools.
*   **High-Quality Output:** Delivers realistic and detailed avatar animations.
*   **Audio Synchronization:** Natural and seamless audio-visual alignment.

## Technical Overview

<p align="center">
  <img src="assets/figures/framework.jpg" alt="model architecture" width="1280"/>
  </br>
  <i>The overview of the framework of StableAvatar.</i>
</p>

StableAvatar addresses the limitations of existing models by introducing a novel Time-step-aware Audio Adapter to prevent error accumulation and an Audio Native Guidance Mechanism to improve audio synchronization. It also introduces a Dynamic Weighted Sliding-window Strategy for smoother transitions in infinite-length videos.

## News and Updates

*   **[2025-9-8]:** 🔥  New demo released! Check out the generated videos on [YouTube](https://www.youtube.com/watch?v=GH4hrxIis3Q) and [Bilibili](https://www.bilibili.com/video/BV1jGYPzqEux).
*   **[2025-8-29]:** 🔥 StableAvatar public demo is now live on [Hugging Face Spaces](https://huggingface.co/spaces/YinmingHuang/StableAvatar). (Note: due to the long video generation time, the demo is currently accessible to <b>Hugging Face Pro</b> users only.)
*   **[2025-8-18]:** 🔥 StableAvatar can run on [ComfyUI](https://github.com/smthemex/ComfyUI_StableAvatar) in just 10 steps, making it 3x faster. Thanks @[smthemex](https://github.com/smthemex) for the contribution.
*   **[2025-8-16]:** 🔥 Finetuning and LoRA training/finetuning codes are released!
*   **[2025-8-15]:** 🔥 StableAvatar can run on Gradio Interface. Thanks @[gluttony-10](https://space.bilibili.com/893892) for the contribution!
*   **[2025-8-15]:** 🔥 StableAvatar can run on [ComfyUI](https://github.com/smthemex/ComfyUI_StableAvatar). Thanks @[smthemex](https://github.com/smthemex) for the contribution.
*   **[2025-8-13]:** 🔥 Added changes to run StableAvatar on the new Blackwell series Nvidia chips, including the RTX 6000 Pro.
*   **[2025-8-11]:** 🔥 Project page, code, technical report and [a basic model checkpoint](https://huggingface.co/FrancisRing/StableAvatar/tree/main) are released.

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

The basic model checkpoint (Wan2.1-1.3B-based) supports generating infinite-length videos at resolutions of 480x832, 832x480, or 512x512. Adjust the number of animated frames or the output resolution if you encounter memory issues.

### Environment Setup

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
# Optional to install flash_attn to accelerate attention computation
pip install flash_attn
```

### Environment Setup for Blackwell series chips

```bash
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
# Optional to install flash_attn to accelerate attention computation
pip install flash_attn
```

### Download Weights

Download the model weights:

```bash
pip install "huggingface_hub[cli]"
cd StableAvatar
mkdir checkpoints
huggingface-cli download FrancisRing/StableAvatar --local-dir ./checkpoints
```

The expected file structure:

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

Extract audio from a video:

```bash
python audio_extractor.py --video_path="path/test/video.mp4" --saved_audio_path="path/test/audio.wav"
```

### Vocal Separation (Optional)

Separate vocals from audio:

```bash
pip install audio-separator[gpu]
python vocal_seperator.py --audio_separator_model_file="path/StableAvatar/checkpoints/Kim_Vocal_2.onnx" --audio_file_path="path/test/audio.wav" --saved_vocal_path="path/test/vocal.wav"
```

### Base Model Inference

Run inference using `inference.sh`.  Modify parameters like resolution (`--width`, `--height`), output directory (`--output_dir`), reference image and audio paths (`--validation_reference_path`, `--validation_driven_audio_path`), text prompts (`--validation_prompts`), and model paths (`--pretrained_model_name_or_path`, `--pretrained_wav2vec_path`, `--transformer_path`).

Example prompt format: `[Description of first frame]-[Description of human behavior]-[Description of background (optional)]`.

```bash
bash inference.sh
```

You can also launch a Gradio interface:

```bash
python app.py
```

Example validation cases are available in `path/StableAvatar/examples`.

#### Tips

*   Choose between `transformer3d-square.pt` and `transformer3d-rec-vec.pt` based on your desired resolution.
*   Use `--GPU_memory_mode` in `inference.sh` to manage GPU memory usage (e.g., `sequential_cpu_offload` or `model_cpu_offload`).
*   Utilize Multi-GPU inference with `--ulysses_degree` and `--ring_degree`.  See the documentation for configuration details.
*   To obtain high quality MP4 files with audio, use ffmpeg:
    ```bash
    ffmpeg -i video_without_audio.mp4 -i /path/audio.wav -c:v copy -c:a aac -shortest /path/output_with_audio.mp4
    ```

### Model Training

**This section provides helpful information if you're training a Video Diffusion Transformer (DiT) model.**

Follow the dataset format:

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

*   `images`, `face_masks`, and `lip_masks` store image frames, face masks, and lip masks, respectively.
*   `sub_clip.mp4` and `audio.wav` are the video and audio files.
*   `video_square_path.txt`, `video_rec_path.txt`, and `video_vec_path.txt` list the paths to video folders.

Use `ffmpeg` to extract frames:

```bash
ffmpeg -i raw_video_1.mp4 -q:v 1 -start_number 0 path/StableAvatar/talking_face_data/rec/speech/00001/images/frame_%d.png
```

For human face and lip mask extraction, refer to the mentioned resources.

Example for lip mask extraction:

```bash
pip install mediapipe
python lip_mask_extractor.py --folder_root="path/StableAvatar/talking_face_data/rec/singing" --start=1 --end=500
```

Run the training scripts:

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

For training parameters, see `train_1B_square.sh` and `train_1B_rec_vec.sh` or the scripts for multi-machine training.  Use the accelerate_config for multi machine configs.

*   The backgrounds of the selected training videos should remain static, as this helps the diffusion model calculate accurate reconstruction loss.
*   The audio should be clear and free from excessive background noise.

### VRAM and Runtime

For a 5-second video (480x832, fps=25), the basic model (`--GPU_memory_mode="model_full_load"`) requires approximately 18GB VRAM and takes about 3 minutes on a 4090 GPU.

**Theoretically, StableAvatar can generate hours of video without significant quality degradation.  You can optionally run the VAE on CPU to manage memory.**

## Contact

For suggestions or assistance, please contact:

Email: francisshuyuan@gmail.com

If you find our work useful, **please consider giving a star ⭐ to this GitHub repository and citing it ❤️**:

```bib
@article{tu2025stableavatar,
  title={Stableavatar: Infinite-length audio-driven avatar video generation},
  author={Tu, Shuyuan and Pan, Yueming and Huang, Yinming and Han, Xintong and Xing, Zhen and Dai, Qi and Luo, Chong and Wu, Zuxuan and Jiang, Yu-Gang},
  journal={arXiv preprint arXiv:2508.08248},
  year={2025}
}
```