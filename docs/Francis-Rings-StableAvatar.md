# StableAvatar: Generate Infinite-Length Avatar Videos from Audio

**Bring your audio to life with StableAvatar, the groundbreaking model for generating high-quality, infinite-length avatar videos directly from audio, all without post-processing!** ([Original Repository](https://github.com/Francis-Rings/StableAvatar))

[![Project Page](https://img.shields.io/badge/Project-Page-Green)](https://francis-rings.github.io/StableAvatar)
[![Paper](https://img.shields.io/badge/Paper-Arxiv-red)](https://arxiv.org/abs/2508.08248)
[![HuggingFace Model](https://img.shields.io/badge/HuggingFace-Model-orange)](https://huggingface.co/FrancisRing/StableAvatar/tree/main)
[![YouTube](https://img.shields.io/badge/YouTube-Watch-red?style=flat-square&logo=youtube)](https://www.youtube.com/watch?v=6lhvmbzvv3Y)
[![Bilibili](https://img.shields.io/badge/Bilibili-Watch-blue?style=flat-square&logo=bilibili)](https://www.bilibili.com/video/BV1hUt9z4EoQ)

StableAvatar introduces a novel approach to audio-driven avatar video generation, enabling the creation of exceptionally long videos that preserve identity and maintain superior audio synchronization. This is achieved without the need for face-related post-processing tools, such as face swapping or restoration.

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
  <span>StableAvatar excels at synthesizing infinite-length and ID-preserving videos driven by audio. All videos are directly synthesized by StableAvatar without the use of any face-related post-processing tools, such as the face-swapping tool FaceFusion or face restoration models like GFP-GAN and CodeFormer.</span>
</p>

<p align="center">
  <video src="https://github.com/user-attachments/assets/90691318-311e-40b9-9bd9-62db83ab1492" width="768" autoplay loop muted playsinline></video>
  <br/>
  <span>Comparison results between StableAvatar and state-of-the-art (SOTA) audio-driven avatar video generation models highlight the superior performance of StableAvatar in delivering <b>infinite-length, high-fidelity, identity-preserving avatar animation</b>.</span>
</p>

## Key Features

*   **Infinite-Length Video Generation:** Create videos of any length without quality degradation.
*   **Identity Preservation:** Maintain consistent facial identity throughout the entire video.
*   **High-Fidelity Audio Synchronization:** Achieve natural and accurate lip-sync and audio alignment.
*   **End-to-End Processing:** Generate videos directly from audio without the need for post-processing tools.
*   **State-of-the-Art Performance:** Outperforms existing models in quality, length, and identity preservation.

## Overview

<p align="center">
  <img src="assets/figures/framework.jpg" alt="model architecture" width="1280"/>
  </br>
  <i>The overview of the framework of StableAvatar.</i>
</p>

Existing methods often struggle to generate long videos with consistent identity and natural audio sync. StableAvatar overcomes these limitations with innovative features like a Time-step-aware Audio Adapter and an Audio Native Guidance Mechanism, along with a Dynamic Weighted Sliding-window Strategy. These advancements prevent error accumulation and enhance audio synchronization, resulting in superior video generation.

## What's New

*   **[2025-08-18]:** StableAvatar now runs on [ComfyUI](https://github.com/smthemex/ComfyUI_StableAvatar) in just 10 steps, making it 3x faster! Thanks @[smthemex](https://github.com/smthemex) for the contribution.
*   **[2025-08-16]:** Finetuning codes and LoRA training/finetuning codes have been released.
*   **[2025-08-15]:** StableAvatar is now available on a Gradio Interface. Thanks @[gluttony-10](https://space.bilibili.com/893892) for the contribution!
*   **[2025-08-15]:** StableAvatar now runs on [ComfyUI](https://github.com/smthemex/ComfyUI_StableAvatar). Thanks @[smthemex](https://github.com/smthemex) for the contribution.
*   **[2025-08-13]:**  Added changes to run StableAvatar on the new Blackwell series Nvidia chips, including the RTX 6000 Pro.
*   **[2025-08-11]:** Project page, code, technical report, and a basic model checkpoint are released.

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

Generate infinite-length videos at 480x832, 832x480, or 512x512 resolution using the basic model checkpoint (Wan2.1-1.3B-based). Reduce the number of animated frames or output resolution if you encounter memory issues.

### Environment Setup

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
# Optional to install flash_attn to accelerate attention computation
pip install flash_attn
```

**For Blackwell Series Chips:**

```bash
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
# Optional to install flash_attn to accelerate attention computation
pip install flash_attn
```

### Download Weights

Set the environment variable if you encounter Hugging Face connection issues: `export HF_ENDPOINT=https://hf-mirror.com`.

```bash
pip install "huggingface_hub[cli]"
cd StableAvatar
mkdir checkpoints
huggingface-cli download FrancisRing/StableAvatar --local-dir ./checkpoints
```

Organize weights in the following structure:

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

Modify `inference.sh` to adjust settings like resolution, output path, and prompts.

```bash
bash inference.sh
```

Example prompts: `[Description of first frame]-[Description of human behavior]-[Description of background (optional)]`.

Use the recommended ranges for `--sample_steps` (30-50), `--overlap_window_length` (5-15), and `--sample_text_guide_scale` and `--sample_audio_guide_scale` (3-6).

Run a Gradio interface:

```bash
python app.py
```

Test with provided examples in `path/StableAvatar/examples`.

#### Tips

*   Use `--transformer_path` in `inference.sh` to switch between `transformer3d-square.pt` and `transformer3d-rec-vec.pt`.
*   Use `--GPU_memory_mode` in `inference.sh` for memory optimization (`model_full_load`, `sequential_cpu_offload`, `model_cpu_offload_and_qfloat8`, or `model_cpu_offload`).
*   Use `--ulysses_degree` and `--ring_degree` in `inference.sh` for multi-GPU inference.
*   Use FSDP with `--fsdp_dit` to reduce GPU memory consumption.
*   If you want to obtain the high quality MP4 file with audio, we recommend you to leverage ffmpeg on the <b>output_path</b> as follows:

```bash
ffmpeg -i video_without_audio.mp4 -i /path/audio.wav -c:v copy -c:a aac -shortest /path/output_with_audio.mp4
```

### Model Training

**This tutorial also helps train a conditioned Video Diffusion Transformer (DiT) model.**

Training dataset structure:

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
StableAvatar is trained on mixed-resolution videos, with 512x512 videos stored in `talking_face_data/square`, 480x832 videos stored in `talking_face_data/vec`, and 832x480 videos stored in `talking_face_data/rec`. Each folder in `talking_face_data/square` or `talking_face_data/rec` or `talking_face_data/vec` contains three subfolders which contains different types of videos (speech, singing, and dancing). 
All `.png` image files are named in the format `frame_i.png`, such as `frame_0.png`, `frame_1.png`, and so on.
`00001`, `00002`, `00003` indicate individual video information.
In terms of three subfolders, `images`, `face_masks`, and `lip_masks` store RGB frames, corresponding human face masks, and corresponding human lip masks, respectively.
`sub_clip.mp4` and `audio.wav` refer to the corresponding RGB video of `images` and the corresponding audio file.
`video_square_path.txt`, `video_rec_path.txt`, and `video_vec_path.txt` record folder paths of `talking_face_data/square`, `talking_face_data/rec`, and `talking_face_data/vec`, respectively.
For example, the content of `video_rec_path.txt` is shown as follows:
```
path/StableAvatar/talking_face_data/rec/speech/00001
path/StableAvatar/talking_face_data/rec/speech/00002
...
path/StableAvatar/talking_face_data/rec/singing/00003
path/StableAvatar/talking_face_data/rec/singing/00004
...
path/StableAvatar/talking_face_data/rec/dancing/00005
path/StableAvatar/talking_face_data/rec/dancing/00006
...
```
If you only have raw videos, you can leverage `ffmpeg` to extract frames from raw videos (speech) and store them in the subfolder `images`.
```
ffmpeg -i raw_video_1.mp4 -q:v 1 -start_number 0 path/StableAvatar/talking_face_data/rec/speech/00001/images/frame_%d.png
```
The obtained frames are saved in `path/StableAvatar/talking_face_data/rec/speech/00001/images`.

For extracting the human face masks, please refer to [StableAnimator repo](https://github.com/Francis-Rings/StableAnimator). The Human Face Mask Extraction section in the tutorial provides off-the-shelf codes.

For extracting the human lip masks, you can run the following command:
```
pip install mediapipe
python lip_mask_extractor.py --folder_root="path/StableAvatar/talking_face_data/rec/singing" --start=1 --end=500
```
`--folder_root` refers to the root path of training datasets.
`--start` and `--end`  specify the starting and ending indices of the selected training dataset. For example, `--start=1 --end=500` indicates that the human lip extraction will start at `path/StableAvatar/talking_face_data/rec/singing/00001` and end at `path/StableAvatar/talking_face_data/rec/singing/00500`.

For extraction details of corresponding audio, please refer to the Audio Extraction section.
When your dataset is organized exactly as outlined above, you can easily train your Wan2.1-1.3B-based StableAvatar by running the following command:
```
# Training StableAvatar on a single resolution setting (512x512) in a single machine
bash train_1B_square.sh
# Training StableAvatar on a single resolution setting (512x512) in multiple machines
bash train_1B_square_64.sh
# Training StableAvatar on a mixed resolution setting (480x832 and 832x480) in a single machine
bash train_1B_rec_vec.sh
# Training StableAvatar on a mixed resolution setting (480x832 and 832x480) in multiple machines
bash train_1B_rec_vec_64.sh
```
For the parameter details of `train_1B_square.sh` and `train_1B_rec_vec.sh`, `CUDA_VISIBLE_DEVICES` refers to gpu devices. In my setting, I use 4 NVIDIA A100 80G to train StableAvatar (`CUDA_VISIBLE_DEVICES=3,2,1,0`).
`--pretrained_model_name_or_path`, `--pretrained_wav2vec_path`, and `--output_dir` refer to the pretrained Wan2.1-1.3B path, pretrained Wav2Vec2.0 path, and the checkpoint saved path of the trained StableAvatar.
`--train_data_square_dir`, `--train_data_rec_dir`, and `--train_data_vec_dir` are the paths of `video_square_path.txt`, `video_rec_path.txt`, and `video_vec_path.txt`, respectively.
`--validation_reference_path` and `--validation_driven_audio_path` are paths of the validation reference image and the validation driven audio.
`--video_sample_n_frames` is the number of frames that StableAvatar processes in a single batch. 
`--num_train_epochs` is the training epoch number. It is worth noting that the default number of training epochs is set to infinite. You can manually terminate the training process once you observe that your StableAvatar has reached its peak performance.
For the parameter details of `train_1B_square_64.sh` and `train_1B_rec_vec_64.sh`, we set the GPU configuration in `path/StableAvatar/accelerate_config/accelerate_config_machine_1B_multiple.yaml`. In my setting, the training setup consists of 8 nodes, each equipped with 8 NVIDIA A100 80GB GPUs, for training StableAvatar.

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
**Training StableAvatar requires approximately 50GB of VRAM for mixed-resolution training and ~40GB VRAM for 512x512 video training.** The background of training videos should be static and audio should be clear.

Regarding training Wan2.1-14B-based StableAvatar, you can run the following command:
```
# Training StableAvatar on a mixed resolution setting (480x832, 832x480, and 512x512) in multiple machines
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./checkpoints/Wan2.1-I2V-14B-480P
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-720P --local-dir ./checkpoints/Wan2.1-I2V-14B-720P # Optional
bash train_14B.sh
```
Deepspeed stage-2 is used for Wan2.1-14B-based StableAvatar training. Modify GPU configuration in `path/StableAvatar/accelerate_config/accelerate_config_machine_14B_multiple.yaml`.  Deepspeed optimization and scheduler settings are in `path/StableAvatar/deepspeed_config/zero_stage2_config.json`. Wan2.1-1.3B-based StableAvatar can already synthesize high-quality videos; the 14B backbone increases inference latency and GPU memory consumption, showing limited performance-to-resource ratio efficiency.

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

### Model Finetuning

Add `--transformer_path` in `train_1B_rec_vec.sh` or `train_1B_rec_vec_64.sh` for full finetuning:
```bash
# Finetuning StableAvatar on a mixed resolution setting (480x832 and 832x480) in a single machine
bash train_1B_rec_vec.sh
# Finetuning StableAvatar on a mixed resolution setting (480x832 and 832x480) in multiple machines
bash train_1B_rec_vec_64.sh
```

Lora finetuning:
```bash
# Lora-Finetuning StableAvatar-1.3B on a mixed resolution setting (480x832 and 832x480) in a single machine
bash train_1B_rec_vec_lora.sh
```

### VRAM Requirement and Runtime

The basic model (480x832, fps=25) needs approximately 18GB VRAM and finishes in 3 minutes on a 4090 GPU.

**StableAvatar can theoretically synthesize hours of video, but the 3D VAE decoder demands significant GPU memory, especially when decoding 10k+ frames. The VAE decoder can be run on the CPU.**

## Contact

For suggestions or help, please contact:

Email: francisshuyuan@gmail.com

If you find our work useful, please consider giving a star ⭐ to this GitHub repository and citing it ❤️:

```bib
@article{tu2025stableavatar,
  title={StableAvatar: Infinite-Length Audio-Driven Avatar Video Generation},
  author={Tu, Shuyuan and Pan, Yueming and Huang, Yinming and Han, Xintong and Xing, Zhen and Dai, Qi and Luo, Chong and Wu, Zuxuan and Jiang Yu-Gang},
  journal={arXiv preprint arXiv:2508.08248},
  year={2025}
}
```