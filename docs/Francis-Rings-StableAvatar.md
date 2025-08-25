# StableAvatar: Generate Infinite-Length Audio-Driven Avatar Videos

**Create stunning, high-fidelity avatar videos that last forever with StableAvatar!**  [See the Original Repo](https://github.com/Francis-Rings/StableAvatar)

[![Project Page](https://img.shields.io/badge/Project-Page-Green)](https://francis-rings.github.io/StableAvatar)
[![Paper](https://img.shields.io/badge/Paper-Arxiv-red)](https://arxiv.org/abs/2508.08248)
[![HuggingFace Model](https://img.shields.io/badge/HuggingFace-Model-orange)](https://huggingface.co/FrancisRing/StableAvatar/tree/main)
[![YouTube](https://img.shields.io/badge/YouTube-Watch-red?style=flat-square&logo=youtube)](https://www.youtube.com/watch?v=6lhvmbzvv3Y)
[![Bilibili](https://img.shields.io/badge/Bilibili-Watch-blue?style=flat-square&logo=bilibili)](https://www.bilibili.com/video/BV1hUt9z4EoQ)

StableAvatar is a cutting-edge video diffusion transformer that revolutionizes audio-driven avatar video generation, allowing for the creation of high-quality, **infinite-length**, and **identity-preserving** videos without the need for face-related post-processing tools. It leverages a novel Time-step-aware Audio Adapter and an Audio Native Guidance Mechanism for superior audio synchronization.

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
  <span>StableAvatar generates audio-driven avatar videos, showcasing its ability to synthesize <b>infinite-length</b> and <b>ID-preserving videos</b>. All videos are <b>directly synthesized by StableAvatar without any face-related post-processing tools</b>, such as face-swapping or face restoration.</span>
</p>

<p align="center">
  <video src="https://github.com/user-attachments/assets/90691318-311e-40b9-9bd9-62db83ab1492" width="768" autoplay loop muted playsinline></video>
  <br/>
  <span>Comparison with existing SOTA models reveals StableAvatar's superior performance in creating <b>infinite-length, high-fidelity, identity-preserving avatar animation</b>.</span>
</p>

## Key Features

*   **Infinite-Length Video Generation:** Create videos of any duration.
*   **Identity Preservation:** Maintains the subject's identity throughout the video.
*   **High-Fidelity Output:** Produces videos with impressive visual quality.
*   **End-to-End Solution:** No face-related post-processing needed, simplifying the workflow.
*   **Superior Audio Synchronization:** Achieve precise lip-sync through innovative audio handling.
*   **Time-Step-Aware Audio Adapter:** Prevents error accumulation for longer video synthesis.
*   **Audio Native Guidance Mechanism:** Enhances audio synchronization during inference.
*   **Dynamic Weighted Sliding-window Strategy:** Improves smoothness for extended videos.

## Overview

<p align="center">
  <img src="assets/figures/framework.jpg" alt="model architecture" width="1280"/>
  </br>
  <i>The overview of the framework of StableAvatar.</i>
</p>

StableAvatar tackles the limitations of current audio-driven avatar generation models by providing a comprehensive end-to-end solution. Key innovations include the Time-step-aware Audio Adapter to prevent error accumulation and an Audio Native Guidance Mechanism to improve audio synchronization, resulting in longer, higher-quality videos.

## News

*   **[2025-8-18]:** 🔥 StableAvatar can now run on [ComfyUI](https://github.com/smthemex/ComfyUI_StableAvatar) for significantly faster performance (3x speed increase!). Thanks to @[smthemex](https://github.com/smthemex) for the contribution.
*   **[2025-8-16]:** 🔥 Finetuning and LoRA training/finetuning codes released! More codes coming soon.
*   **[2025-8-15]:** 🔥 StableAvatar can run on Gradio Interface. Thanks @[gluttony-10](https://space.bilibili.com/893892) for the contribution!
*   **[2025-8-15]:** 🔥 StableAvatar can run on [ComfyUI](https://github.com/smthemex/ComfyUI_StableAvatar). Thanks @[smthemex](https://github.com/smthemex) for the contribution.
*   **[2025-8-13]:** 🔥 Added support for the new Blackwell series Nvidia chips, including the RTX 6000 Pro.
*   **[2025-8-11]:** 🔥 Project page, code, technical report, and [basic model checkpoint](https://huggingface.co/FrancisRing/StableAvatar/tree/main) released. LoRA training codes, evaluation dataset, and StableAvatar-pro coming soon!

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

The basic Wan2.1-1.3B-based model checkpoint supports generating **infinite-length videos** at resolutions of **480x832, 832x480 or 512x512**. Reduce animated frames or the output resolution if you face memory issues.

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

If you encounter issues with Hugging Face, use the mirror endpoint: `export HF_ENDPOINT=https://hf-mirror.com`. Download the weights manually:

```bash
pip install "huggingface_hub[cli]"
cd StableAvatar
mkdir checkpoints
huggingface-cli download FrancisRing/StableAvatar --local-dir ./checkpoints
```

Ensure the file structure is as follows:

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

Extract audio from a video file:

```bash
python audio_extractor.py --video_path="path/test/video.mp4" --saved_audio_path="path/test/audio.wav"
```

### 🧱 Vocal Separation

Separate vocals from audio (optional for improved lip-sync):

```bash
pip install audio-separator[gpu]
python vocal_seperator.py --audio_separator_model_file="path/StableAvatar/checkpoints/Kim_Vocal_2.onnx" --audio_file_path="path/test/audio.wav" --saved_vocal_path="path/test/vocal.wav"
```

### 🧱 Base Model Inference

A sample configuration is provided in `inference.sh`.  Adjust the settings as needed.

```bash
bash inference.sh
```

StableAvatar supports 512x512, 480x832, and 832x480 resolutions. Modify `--width` and `--height` in `inference.sh` to set the resolution.  `--output_dir` sets the output path. `--validation_reference_path`, `--validation_driven_audio_path`, and `--validation_prompts` are the image, audio, and text prompts, respectively.

Prompts are critical: `[Description of first frame]-[Description of human behavior]-[Description of background (optional)]`.

`--pretrained_model_name_or_path`, `--pretrained_wav2vec_path`, and `--transformer_path` specify the model, Wav2Vec2.0, and StableAvatar weights, respectively.

`--sample_steps`, `--overlap_window_length`, and `--clip_sample_n_frames` control inference steps, overlap between windows, and frames per batch.  Recommend `--sample_steps` [30-50], `--overlap_window_length` [5-15].

`--sample_text_guide_scale` and `--sample_audio_guide_scale` are the CFG scales.  Recommended range is `[3-6]`.  Increase audio CFG for improved lip-sync.

Alternatively, run a Gradio interface:

```bash
python app.py
```

Check the `path/StableAvatar/examples` directory for 6 validation cases in different resolutions.

#### 💡Tips

*   Two Wan2.1-1.3B-based StableAvatar versions: `transformer3d-square.pt` and `transformer3d-rec-vec.pt`, trained on different datasets. Both support all three resolutions. Modify `--transformer_path` in `inference.sh` to switch between them.

*   For limited GPU resources, manage the loading mode with `--GPU_memory_mode` options: `model_full_load`, `sequential_cpu_offload`, `model_cpu_offload_and_qfloat8`, `model_cpu_offload`.  `sequential_cpu_offload` uses approximately 3G of GPU memory.  `model_cpu_offload` reduces GPU memory usage further.

*   Use multi-GPU inference to speed up with `--ulysses_degree` and `--ring_degree`.  For example, on 8 GPUs, set `--ulysses_degree=4` and `--ring_degree=2`. Ensure `ulysses_degree * ring_degree = total GPU number / world-size`.  Also add `--fsdp_dit` to activate FSDP in DiT.
```bash
bash multiple_gpu_inference.sh
```
(Example shows inference setup for 4 GPUs)

If you would like to convert video without audio to a high-quality MP4 with audio, you can use ffmpeg on the output_path to achieve the following:
```bash
ffmpeg -i video_without_audio.mp4 -i /path/audio.wav -c:v copy -c:a aac -shortest /path/output_with_audio.mp4
```

### 🧱 Model Training

**🔥🔥This training guide will also be helpful if you want to train a Video Diffusion Transformer (DiT) model like Wan2.1.🔥🔥**

The training dataset should be organized as shown in the original README.

StableAvatar is trained on mixed-resolution videos: 512x512 in `talking_face_data/square`, 480x832 in `talking_face_data/vec`, and 832x480 in `talking_face_data/rec`. Each directory contains subfolders for different video types (speech, singing, dancing)

All `.png` image files are named in the format `frame_i.png`.

`00001`, `00002`, `00003` are video examples.

The `images`, `face_masks`, and `lip_masks` folders store the RGB frames, face masks, and lip masks.

`sub_clip.mp4` and `audio.wav` refer to the RGB video and audio file.

`video_square_path.txt`, `video_rec_path.txt`, and `video_vec_path.txt` record directory paths.

For example, the content of `video_rec_path.txt` is shown in the original README.

If you have raw videos, use `ffmpeg` to extract frames:

```bash
ffmpeg -i raw_video_1.mp4 -q:v 1 -start_number 0 path/StableAvatar/talking_face_data/rec/speech/00001/images/frame_%d.png
```

For face mask extraction, see the [StableAnimator repo](https://github.com/Francis-Rings/StableAnimator).  Lip mask extraction:

```bash
pip install mediapipe
python lip_mask_extractor.py --folder_root="path/StableAvatar/talking_face_data/rec/singing" --start=1 --end=500
```

For audio extraction, see the Audio Extraction section.

When your dataset is organized as detailed, train your Wan2.1-1.3B-based StableAvatar by running:

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

See the original README for parameter details of `train_1B_square.sh`, `train_1B_rec_vec.sh`, `train_1B_square_64.sh`, and `train_1B_rec_vec_64.sh`.

The overall file structure of StableAvatar at training is shown in the original README.

**Training StableAvatar requires approximately 50GB of VRAM due to the mixed-resolution (480x832 and 832x480) training pipeline. If you train StableAvatar exclusively on 512x512 videos, the VRAM requirement is reduced to approximately 40GB.**

Backgrounds should be static to ensure accurate reconstruction loss calculation.  Audio should be clear, free of excessive noise.

Training Wan2.1-14B-based StableAvatar:

```bash
# Training StableAvatar on a mixed resolution setting (480x832, 832x480, and 512x512) in multiple machines
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./checkpoints/Wan2.1-I2V-14B-480P
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-720P --local-dir ./checkpoints/Wan2.1-I2V-14B-720P # Optional
bash train_14B.sh
```

We use deepspeed stage-2 for the Wan2.1-14B-based training. The GPU configuration can be modified in `path/StableAvatar/accelerate_config/accelerate_config_machine_14B_multiple.yaml`.

The deepspeed optimization and scheduler settings are in `path/StableAvatar/deepspeed_config/zero_stage2_config.json`.

We observe that Wan2.1-1.3B-based StableAvatar synthesizes high-quality avatar videos, while the Wan2.1-14B backbone increases inference latency and GPU memory consumption, indicating limited efficiency in terms of performance-to-resource ratio.

Lora training commands:

```bash
# Training StableAvatar-1.3B on a mixed resolution setting (480x832 and 832x480) in a single machine
bash train_1B_rec_vec_lora.sh
# Training StableAvatar-1.3B on a mixed resolution setting (480x832 and 832x480) in multiple machines
bash train_1B_rec_vec_lora_64.sh
# Lora-Training StableAvatar-14B on a mixed resolution setting (480x832, 832x480, and 512x512) in multiple machines
bash train_14B_lora.sh
```

Modify `--rank` and `--network_alpha` to control the LoRA training/finetuning quality.

If you would like to train 720P Wan2.1-1.3B-based or Wan2.1-14B-based StableAvatar, you can directly modify the height and width of the dataloader (480p-->720p) in `train_1B_square.py`/`train_1B_vec_rec.py`/`train_14B.py`.

### 🧱 Model Finetuning

For fully finetuning, add `--transformer_path="path/StableAvatar/checkpoints/StableAvatar-1.3B/transformer3d-square.pt"` to the `train_1B_rec_vec.sh` or `train_1B_rec_vec_64.sh`:

```bash
# Finetuning StableAvatar on a mixed resolution setting (480x832 and 832x480) in a single machine
bash train_1B_rec_vec.sh
# Finetuning StableAvatar on a mixed resolution setting (480x832 and 832x480) in multiple machines
bash train_1B_rec_vec_64.sh
```

For LoRA finetuning, add `--transformer_path="path/StableAvatar/checkpoints/StableAvatar-1.3B/transformer3d-square.pt"` to the `train_1B_rec_vec_lora.sh`:

```bash
# Lora-Finetuning StableAvatar-1.3B on a mixed resolution setting (480x832 and 832x480) in a single machine
bash train_1B_rec_vec_lora.sh
```

Modify `--rank` and `--network_alpha` to control the quality of LoRA training/finetuning.

### 🧱 VRAM Requirement and Runtime

The basic model (--GPU_memory_mode="model_full_load") requires approximately 18GB VRAM and finishes in 3 minutes on a 4090 GPU for a 5s video (480x832, fps=25).

**🔥🔥Theoretically, StableAvatar is capable of synthesizing hours of video without significant quality degradation; however, the 3D VAE decoder demands significant GPU memory, especially when decoding 10k+ frames. You have the option to run the VAE decoder on CPU.🔥🔥**

## Contact

For any suggestions or if you find the work helpful, feel free to contact me:

Email: francisshuyuan@gmail.com

If you find our work useful, <b>please consider giving a star ⭐ to this github repository and citing it ❤️</b>:

```bib
@article{tu2025stableavatar,
  title={StableAvatar: Infinite-Length Audio-Driven Avatar Video Generation},
  author={Tu, Shuyuan and Pan, Yueming and Huang, Yinming and Han, Xintong and Xing, Zhen and Dai, Qi and Luo, Chong and Wu, Zuxuan and Jiang Yu-Gang},
  journal={arXiv preprint arXiv:2508.08248},
  year={2025}
}
```