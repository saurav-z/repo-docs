# StableAvatar: Create Infinite-Length Avatar Videos with Audio (and No Post-Processing!)

**Generate stunning, infinite-length avatar videos driven by audio, surpassing existing models with superior fidelity and identity preservation!** Explore the [original repository](https://github.com/Francis-Rings/StableAvatar) for the latest updates and code.

[![Project Page](https://img.shields.io/badge/Project-Page-Green)](https://francis-rings.github.io/StableAvatar)
[![Arxiv](https://img.shields.io/badge/Paper-Arxiv-red)](https://arxiv.org/abs/2508.08248)
[![HuggingFace Model](https://img.shields.io/badge/HuggingFace-Model-orange)](https://huggingface.co/FrancisRing/StableAvatar/tree/main)
[![YouTube](https://img.shields.io/badge/YouTube-Watch-red?style=flat-square&logo=youtube)](https://www.youtube.com/watch?v=6lhvmbzvv3Y)
[![Bilibili](https://img.shields.io/badge/Bilibili-Watch-blue?style=flat-square&logo=bilibili)](https://www.bilibili.com/video/BV1hUt9z4EoQ)

StableAvatar is a groundbreaking approach to audio-driven avatar video generation, capable of producing videos of **unlimited length** while maintaining **perfect identity preservation** and **natural audio synchronization**.  It achieves this without relying on any face-related post-processing tools, making it a truly end-to-end solution.

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
  StableAvatar generates audio-driven avatar videos showcasing its power to synthesize <b>infinite-length</b> and <b>ID-preserving videos</b>.  All videos are synthesized directly by StableAvatar, <b>without the use of any face-related post-processing tools</b>.
</p>

<p align="center">
  <video src="https://github.com/user-attachments/assets/90691318-311e-40b9-9bd9-62db83ab1492" width="768" autoplay loop muted playsinline></video>
  <br/>
  Comparison results highlight StableAvatar's superior performance in delivering <b>infinite-length, high-fidelity, identity-preserving avatar animation</b> compared to other SOTA models.
</p>

## Key Features

*   **Infinite-Length Video Generation:** Produce videos of any duration.
*   **Identity Preservation:**  Guaranteed consistent identity throughout the video.
*   **High-Fidelity Output:**  Generate high-quality, realistic videos.
*   **End-to-End Solution:**  No need for external face-swapping or restoration tools.
*   **Audio Synchronization:**  Natural and accurate synchronization of audio and video.
*   **Time-step-aware Audio Adapter:** Prevents error accumulation, ensuring the quality of long videos.
*   **Audio Native Guidance Mechanism:** Enhances audio synchronization.
*   **Dynamic Weighted Sliding-window Strategy:** Improves the smoothness of infinite-length videos.

## Overview

<p align="center">
  <img src="assets/figures/framework.jpg" alt="model architecture" width="1280"/>
  </br>
  <i>The overview of the framework of StableAvatar.</i>
</p>

Existing models struggle to generate long videos with consistent identity and audio synchronization. StableAvatar overcomes these limitations with a novel video diffusion transformer that enables infinite-length video generation without post-processing.  The core innovation lies in addressing audio modeling, introducing a Time-step-aware Audio Adapter to prevent error accumulation and an Audio Native Guidance Mechanism for improved synchronization.  A Dynamic Weighted Sliding-window Strategy is also introduced to enhance video smoothness.

## What's New

*   `[2025-8-18]`:  🔥 StableAvatar runs on [ComfyUI](https://github.com/smthemex/ComfyUI_StableAvatar) for faster generation (3x speed increase!). Thanks @[smthemex](https://github.com/smthemex)!
*   `[2025-8-16]`:  🔥 Finetuning and LoRA training/finetuning codes are released! More codes will be public soon.
*   `[2025-8-15]`:  🔥 StableAvatar runs on Gradio Interface. Thanks @[gluttony-10](https://space.bilibili.com/893892)!
*   `[2025-8-15]`:  🔥 StableAvatar runs on [ComfyUI](https://github.com/smthemex/ComfyUI_StableAvatar). Thanks @[smthemex](https://github.com/smthemex)!
*   `[2025-8-13]`:  🔥 Support for new Blackwell series Nvidia chips, including the RTX 6000 Pro.
*   `[2025-8-11]`:  🔥 Project page, code, technical report, and a basic model checkpoint ([https://huggingface.co/FrancisRing/StableAvatar/tree/main](https://huggingface.co/FrancisRing/StableAvatar/tree/main)) are released. Further LoRA training codes, the evaluation dataset, and StableAvatar-pro are coming soon!

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

## Quickstart

The basic model checkpoint (Wan2.1-1.3B-based) supports generating **infinite-length videos** at **480x832, 832x480, or 512x512 resolution**.  Reduce the number of frames or the resolution if you encounter memory issues.

### Environment Setup

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
# Optional to install flash_attn to accelerate attention computation
pip install flash_attn
```

### Environment Setup for Blackwell Series Chips

```bash
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
# Optional to install flash_attn to accelerate attention computation
pip install flash_attn
```

### Download Weights

If you have issues connecting to Hugging Face, use the mirror endpoint: `export HF_ENDPOINT=https://hf-mirror.com`.

Download the weights manually:

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

Extract audio from a video file:

```bash
python audio_extractor.py --video_path="path/test/video.mp4" --saved_audio_path="path/test/audio.wav"
```

### Vocal Separation

Separate vocals from audio for better lip sync:

```bash
pip install audio-separator[gpu]
python vocal_seperator.py --audio_separator_model_file="path/StableAvatar/checkpoints/Kim_Vocal_2.onnx" --audio_file_path="path/test/audio.wav" --saved_vocal_path="path/test/vocal.wav"
```

### Base Model Inference

Example configuration in `inference.sh`.  Modify the script for your needs.

```bash
bash inference.sh
```

Wan2.1-1.3B supports 512x512, 480x832, and 832x480 resolutions. Adjust `--width` and `--height` in `inference.sh` to set the resolution.

*   `--output_dir`: Output directory for generated videos.
*   `--validation_reference_path`: Path to the reference image.
*   `--validation_driven_audio_path`: Path to the audio file.
*   `--validation_prompts`: Text prompts. Recommended format: `[Description of first frame]-[Description of human behavior]-[Description of background (optional)]`.
*   `--pretrained_model_name_or_path`: Pretrained Wan2.1-1.3B weights path.
*   `--pretrained_wav2vec_path`: Pretrained Wav2Vec2.0 weights path.
*   `--transformer_path`: Pretrained StableAvatar weights path.
*   `--sample_steps`: Number of inference steps (30-50 recommended).  More steps = higher quality.
*   `--overlap_window_length`: Overlapping context length (5-15 recommended).
*   `--clip_sample_n_frames`: Frames synthesized in a batch.
*   `--sample_text_guide_scale` and `--sample_audio_guide_scale`: Classifier-Free-Guidance scales for text and audio (3-6 recommended). Increase audio scale for lip-sync improvements.

You can also run a Gradio interface:

```bash
python app.py
```

Example videos are in `path/StableAvatar/examples`.  Enjoy generating infinite-length avatar videos!

#### Tips

*   Use `transformer3d-square.pt` or `transformer3d-rec-vec.pt` for the `--transformer_path`.
*   If you have limited GPU resources, you can change the loading mode by modifying "--GPU_memory_mode" in `inference.sh`. The options of "--GPU_memory_mode" are `model_full_load`, `sequential_cpu_offload`, `model_cpu_offload_and_qfloat8`, and `model_cpu_offload`.
*   If you have multiple GPUs, you can speed up inference by modifying "--ulysses_degree" and "--ring_degree" in `inference.sh`.
*   The video synthesized by StableAvatar is without audio. If you want to obtain the high quality MP4 file with audio, we recommend you to leverage ffmpeg on the output_path as follows:
```
ffmpeg -i video_without_audio.mp4 -i /path/audio.wav -c:v copy -c:a aac -shortest /path/output_with_audio.mp4
```

### Model Training

**🔥🔥If you want to train a conditioned Video Diffusion Transformer (DiT) model, such as Wan2.1, this training tutorial will also be helpful.🔥🔥**

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
*   `images`: RGB frames (named `frame_i.png`).
*   `face_masks`: Face masks.
*   `lip_masks`: Lip masks.
*   `sub_clip.mp4`: The original video.
*   `audio.wav`: The audio file.
*   `video_square_path.txt`, `video_rec_path.txt`, and `video_vec_path.txt`:  Text files containing the paths to the video folders.

If you only have raw videos, extract frames:

```bash
ffmpeg -i raw_video_1.mp4 -q:v 1 -start_number 0 path/StableAvatar/talking_face_data/rec/speech/00001/images/frame_%d.png
```

Use a face mask extraction tool (like [StableAnimator](https://github.com/Francis-Rings/StableAnimator)).

Extract lip masks:

```bash
pip install mediapipe
python lip_mask_extractor.py --folder_root="path/StableAvatar/talking_face_data/rec/singing" --start=1 --end=500
```

For audio extraction details, see the Audio Extraction section.

Training command examples:

```bash
# Single resolution (512x512) - Single Machine
bash train_1B_square.sh
# Single resolution (512x512) - Multi Machine
bash train_1B_square_64.sh
# Mixed resolution (480x832 and 832x480) - Single Machine
bash train_1B_rec_vec.sh
# Mixed resolution (480x832 and 832x480) - Multi Machine
bash train_1B_rec_vec_64.sh
```

*   Modify parameters in `train_1B_square.sh` and `train_1B_rec_vec.sh`:
    *   `CUDA_VISIBLE_DEVICES`: GPU devices.
    *   `--pretrained_model_name_or_path`: Pretrained Wan2.1-1.3B path.
    *   `--pretrained_wav2vec_path`: Pretrained Wav2Vec2.0 path.
    *   `--output_dir`: Checkpoint save path.
    *   `--train_data_square_dir`, `--train_data_rec_dir`, `--train_data_vec_dir`: Paths to the video path text files.
    *   `--validation_reference_path`: Validation reference image path.
    *   `--validation_driven_audio_path`: Validation audio path.
    *   `--video_sample_n_frames`: Frames per batch.
    *   `--num_train_epochs`: Training epochs (default is infinite; manually terminate when desired).
*   Modify parameters in `train_1B_square_64.sh` and `train_1B_rec_vec_64.sh`: GPU configuration in `path/StableAvatar/accelerate_config/accelerate_config_machine_1B_multiple.yaml`.

File Structure during training:

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

**Training StableAvatar requires approximately 50GB of VRAM for the mixed-resolution training pipeline. If you train exclusively on 512x512 videos, the VRAM requirement is reduced to approximately 40GB.** Backgrounds in training videos should remain static, and audio should be clear of noise.

Regarding training Wan2.1-14B-based StableAvatar, you can run the following command:
```
# Training StableAvatar on a mixed resolution setting (480x832, 832x480, and 512x512) in multiple machines
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./checkpoints/Wan2.1-I2V-14B-480P
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-720P --local-dir ./checkpoints/Wan2.1-I2V-14B-720P # Optional
bash train_14B.sh
```
We utilize deepspeed stage-2 to train Wan2.1-14B-based StableAvatar. The GPU configuration can be modified in `path/StableAvatar/accelerate_config/accelerate_config_machine_14B_multiple.yaml`.
The deepspeed optimization configuration and deepspeed scheduler configuration are in `path/StableAvatar/deepspeed_config/zero_stage2_config.json`.
Notably, we observe that Wan2.1-1.3B-based StableAvatar is already capable of synthesizing infinite-length high quality avatar videos. The Wan2.1-14B backbone significantly increase the inference latency and GPU memory consumption during training, indicating limited efficiency in terms of performance-to-resource ratio.

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

Finetune by including `--transformer_path`:

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

### VRAM and Runtime

The basic model (480x832, 25fps, `--GPU_memory_mode="model_full_load"`) requires approximately 18GB VRAM and takes 3 minutes per 5-second video on a 4090 GPU.

**StableAvatar can theoretically synthesize hours of video; however, the 3D VAE decoder demands significant GPU memory, especially when decoding 10k+ frames. You have the option to run the VAE decoder on CPU.**

## Contact

For suggestions or assistance, please contact:

*   Email: francisshuyuan@gmail.com

If you find this project useful, **please star the repository and cite our work**:

```bib
@article{tu2025stableavatar,
  title={StableAvatar: Infinite-Length Audio-Driven Avatar Video Generation},
  author={Tu, Shuyuan and Pan, Yueming and Huang, Yinming and Han, Xintong and Xing, Zhen and Dai, Qi and Luo, Chong and Wu, Zuxuan and Jiang Yu-Gang},
  journal={arXiv preprint arXiv:2508.08248},
  year={2025}
}
```