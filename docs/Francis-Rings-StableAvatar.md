# StableAvatar: Generate Infinite-Length, Audio-Driven Avatar Videos

**Create stunning, infinite-length avatar videos driven by audio!** [Explore the StableAvatar Repository](https://github.com/Francis-Rings/StableAvatar)

[![Project Page](https://img.shields.io/badge/Project-Page-Green)](https://francis-rings.github.io/StableAvatar)
[![Paper - Arxiv](https://img.shields.io/badge/Paper-Arxiv-red)](https://arxiv.org/abs/2508.08248)
[![HuggingFace Model](https://img.shields.io/badge/HuggingFace-Model-orange)](https://huggingface.co/FrancisRing/StableAvatar/tree/main)
[![Hugging Face Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue)](https://huggingface.co/spaces/YinmingHuang/StableAvatar)
[![YouTube Demo](https://img.shields.io/badge/YouTube-Watch-red?style=flat-square&logo=youtube)](https://www.youtube.com/watch?v=6lhvmbzvv3Y)
[![Bilibili Demo](https://img.shields.io/badge/Bilibili-Watch-blue?style=flat-square&logo=bilibili)](https://www.bilibili.com/video/BV1hUt9z4EoQ)

StableAvatar is a cutting-edge video diffusion transformer enabling the generation of infinite-length, high-quality avatar videos directly from audio input, without the need for post-processing. Developed by researchers at Fudan University, Microsoft Research Asia, Xi'an Jiaotong University, and Tencent Inc, StableAvatar excels at creating videos that maintain identity and achieve seamless audio-visual synchronization.

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
  StableAvatar excels in generating audio-driven avatar videos of <b>infinite length</b> while flawlessly preserving identity. These videos are generated <b>end-to-end</b>, without relying on any face-related post-processing tools.
</p>

<p align="center">
  <video src="https://github.com/user-attachments/assets/90691318-311e-40b9-9bd9-62db83ab1492" width="768" autoplay loop muted playsinline></video>
  <br/>
  <span>Comparison results highlight StableAvatar's superior performance, delivering <b>high-fidelity, identity-preserving avatar animation</b>.</span>
</p>


## Key Features

*   **Infinite-Length Video Generation:** Create avatar videos of any length, allowing for extended storytelling and content creation.
*   **Identity Preservation:** Ensures the generated avatar maintains a consistent identity throughout the video.
*   **High-Fidelity Results:** Delivers high-quality video output, enhancing the viewer experience.
*   **End-to-End Solution:** Generates videos directly from audio input, eliminating the need for post-processing.
*   **Seamless Audio Synchronization:** Achieves natural and precise synchronization between audio and video.
*   **Versatile Application:** Supports a wide range of use cases, including virtual assistants, content creation, and entertainment.

## Overview: How it Works

<p align="center">
  <img src="assets/figures/framework.jpg" alt="model architecture" width="1280"/>
  </br>
  <i>The framework of StableAvatar.</i>
</p>

StableAvatar addresses the challenges of generating long, high-quality avatar videos by:

*   **Time-step-aware Audio Adapter:** Prevents error accumulation in the audio processing pipeline.
*   **Audio Native Guidance Mechanism:** Enhances audio synchronization using the diffusion model's predictions.
*   **Dynamic Weighted Sliding-window Strategy:** Smooths transitions and improves overall video quality.

## What's New

*   `[2025-9-8]`: 🔥  New Demo released! [YouTube](https://www.youtube.com/watch?v=GH4hrxIis3Q) and [Bilibili](https://www.bilibili.com/video/BV1jGYPzqEux).
*   `[2025-8-29]`: 🔥 Public demo on [Hugging Face Spaces](https://huggingface.co/spaces/YinmingHuang/StableAvatar) (for Hugging Face Pro users).
*   `[2025-8-18]`: 🔥 ComfyUI support (3x faster!), thanks to @[smthemex](https://github.com/smthemex).
*   `[2025-8-16]`: 🔥 Finetuning and LoRA training codes released!
*   `[2025-8-15]`: 🔥 Gradio Interface support, thanks to @[gluttony-10](https://space.bilibili.com/893892)! & ComfyUI support.
*   `[2025-8-13]`: 🔥 Support for the new Blackwell series Nvidia chips, including the RTX 6000 Pro.
*   `[2025-8-11]`: 🔥 Project page, code, technical report, and basic model checkpoint released ([Hugging Face](https://huggingface.co/FrancisRing/StableAvatar/tree/main)).

## 🛠️ To-Do List
- [x] StableAvatar-1.3B-basic
- [x] Inference Code
- [x] Data Pre-Processing Code (Audio Extraction)
- [x] Data Pre-Processing Code (Vocal Separation)
- [x] Training Code
- [x] Full Finetuning Code
- [x] Lora Training Code
- [x] Lora Finetuning Code
- [ ] Inference Code with Audio Native Guidance
- [ ] StableAvatar-pro

## 🔑 Quickstart: Get Started Now

StableAvatar is ready to go! This quickstart guide will get you up and running with the basic model.

### 🧱 Environment Setup

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
# Optional to install flash_attn to accelerate attention computation
pip install flash_attn
```

### 🧱 Environment setup for Blackwell series chips

```
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
# Optional to install flash_attn to accelerate attention computation
pip install flash_attn
```

### 🧱 Download Weights
If you encounter connection issues with Hugging Face, you can utilize the mirror endpoint by setting the environment variable: `export HF_ENDPOINT=https://hf-mirror.com`.
Please download weights manually as follows:
```
pip install "huggingface_hub[cli]"
cd StableAvatar
mkdir checkpoints
huggingface-cli download FrancisRing/StableAvatar --local-dir ./checkpoints
```
All the weights should be organized in models as follows
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
Given the target video file (.mp4), you can use the following command to obtain the corresponding audio file (.wav):
```
python audio_extractor.py --video_path="path/test/video.mp4" --saved_audio_path="path/test/audio.wav"
```

### 🧱 Vocal Separation
As noisy background music may negatively impact the performance of StableAvatar to some extents, you can further separate the vocal from the audio file for better lip synchronization.
Given the path to an audio file (.wav), you can run the following command to extract the corresponding vocal signals:
```
pip install audio-separator[gpu]
python vocal_seperator.py --audio_separator_model_file="path/StableAvatar/checkpoints/Kim_Vocal_2.onnx" --audio_file_path="path/test/audio.wav" --saved_vocal_path="path/test/vocal.wav"
```

### 🧱 Base Model Inference
A sample configuration for testing is provided as `inference.sh`. You can also easily modify the various configurations according to your needs.

```bash
bash inference.sh
```
Wan2.1-1.3B-based StableAvatar supports audio-driven avatar video generation at three different resolution settings: 512x512, 480x832, and 832x480. You can modify "--width" and "--height" in `inference.sh` to set the resolution of the animation. "--output_dir" in `inference.sh` refers to the saved path of the generated animation. "--validation_reference_path", "--validation_driven_audio_path", and "--validation_prompts" in `inference.sh` refer to the path of the given reference image, the path of the given audio, and the text prompts respectively.
Prompts are also very important. It is recommended to `[Description of first frame]-[Description of human behavior]-[Description of background (optional)]`.
"--pretrained_model_name_or_path", "--pretrained_wav2vec_path", and "--transformer_path" in `inference.sh` are the paths of pretrained Wan2.1-1.3B weights, pretrained Wav2Vec2.0 weights, and pretrained StableAvatar weights, respectively.
"--sample_steps", "--overlap_window_length", and "--clip_sample_n_frames" refer to the total number of inference steps, the overlapping context length between two context windows, and the synthesized frame number in a batch/context window, respectively. 
Notably, the recommended `--sample_steps` range is [30-50], more steps bring higher quality. The recommended `--overlap_window_length` range is [5-15], as longer overlapping length results in higher quality and slower inference speed.
"--sample_text_guide_scale" and "--sample_audio_guide_scale" are Classify-Free-Guidance scale of text prompt and audio. The recommended range for prompt and audio cfg is `[3-6]`. You can increase the audio cfg to facilitate the lip synchronization with audio.

Additionally, you can also run the following command to launch a Gradio interface:
```bash
python app.py
```

We provide 6 cases in different resolution settings in `path/StableAvatar/examples` for validation. ❤️❤️Please feel free to try it out and enjoy the endless entertainment of infinite-length avatar video generation❤️❤️!

#### 💡Tips
- Wan2.1-1.3B-based StableAvatar weights have two versions: `transformer3d-square.pt` and `transformer3d-rec-vec.pt`, which are trained on two video datasets in two different resolution settings. Two versions both support generating audio-driven avatar video at three different resolution settings: 512x512, 480x832, and 832x480. You can modify `--transformer_path` in `inference.sh` to switch these two versions.

- If you have limited GPU resources, you can change the loading mode of StableAvatar by modifying "--GPU_memory_mode" in `inference.sh`. The options of "--GPU_memory_mode" are `model_full_load`, `sequential_cpu_offload`, `model_cpu_offload_and_qfloat8`, and `model_cpu_offload`. In particular, when you set `--GPU_memory_mode` to `sequential_cpu_offload`, the total GPU memory consumption is approximately 3G with slower inference speed.
Setting `--GPU_memory_mode` to `model_cpu_offload` can significantly cut GPU memory usage, reducing it by roughly half compared to `model_full_load` mode.

- If you have multiple Gpus, you can run Multi-GPU inference to speed up by modifying "--ulysses_degree" and "--ring_degree" in `inference.sh`. For example, if you have 8 GPUs, you can set `--ulysses_degree=4` and `--ring_degree=2`. Notably, you have to ensure ulysses_degree*ring_degree=total GPU number/world-size. Moreover, you can also add `--fsdp_dit` in `inference.sh` to activate FSDP in DiT to further reduce GPU memory consumption.
You can fun the following command:
```bash
bash multiple_gpu_inference.sh
```
In my setting, 4 GPUs are utilized for inference.

The video synthesized by StableAvatar is without audio. If you want to obtain the high quality MP4 file with audio, we recommend you to leverage ffmpeg on the <b>output_path</b> as follows:
```bash
ffmpeg -i video_without_audio.mp4 -i /path/audio.wav -c:v copy -c:a aac -shortest /path/output_with_audio.mp4
```

### 🧱 Model Training
<b>🔥🔥It’s worth noting that if you’re looking to train a conditioned Video Diffusion Transformer (DiT) model, such as Wan2.1, this training tutorial will also be helpful.🔥🔥</b>
For the training dataset, it has to be organized as follows:

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
```bash
ffmpeg -i raw_video_1.mp4 -q:v 1 -start_number 0 path/StableAvatar/talking_face_data/rec/speech/00001/images/frame_%d.png
```
The obtained frames are saved in `path/StableAvatar/talking_face_data/rec/speech/00001/images`.

For extracting the human face masks, please refer to [StableAnimator repo](https://github.com/Francis-Rings/StableAnimator). The Human Face Mask Extraction section in the tutorial provides off-the-shelf codes.

For extracting the human lip masks, you can run the following command:
```bash
pip install mediapipe
python lip_mask_extractor.py --folder_root="path/StableAvatar/talking_face_data/rec/singing" --start=1 --end=500
```
`--folder_root` refers to the root path of training datasets.
`--start` and `--end`  specify the starting and ending indices of the selected training dataset. For example, `--start=1 --end=500` indicates that the human lip extraction will start at `path/StableAvatar/talking_face_data/rec/singing/00001` and end at `path/StableAvatar/talking_face_data/rec/singing/00500`.

For extraction details of corresponding audio, please refer to the Audio Extraction section.
When your dataset is organized exactly as outlined above, you can easily train your Wan2.1-1.3B-based StableAvatar by running the following command:
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
<b>It is worth noting that training StableAvatar requires approximately 50GB of VRAM due to the mixed-resolution (480x832 and 832x480) training pipeline. 
However, if you train StableAvatar exclusively on 512x512 videos, the VRAM requirement is reduced to approximately 40GB.</b>
Additionally, The backgrounds of the selected training videos should remain static, as this helps the diffusion model calculate accurate reconstruction loss.
The audio should be clear and free from excessive background noise.

Regarding training Wan2.1-14B-based StableAvatar, you can run the following command:
```bash
# Training StableAvatar on a mixed resolution setting (480x832, 832x480, and 512x512) in multiple machines
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./checkpoints/Wan2.1-I2V-14B-480P
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-720P --local-dir ./checkpoints/Wan2.1-I2V-14B-720P # Optional
bash train_14B.sh
```
We utilize deepspeed stage-2 to train Wan2.1-14B-based StableAvatar. The GPU configuration can be modified in `path/StableAvatar/accelerate_config/accelerate_config_machine_14B_multiple.yaml`.
The deepspeed optimization configuration and deepspeed scheduler configuration are in `path/StableAvatar/deepspeed_config/zero_stage2_config.json`.
Notably, we observe that Wan2.1-1.3B-based StableAvatar is already capable of synthesizing infinite-length high quality avatar videos. The Wan2.1-14B backbone significantly increase the inference latency and GPU memory consumption during training, indicating limited efficiency in terms of performance-to-resource ratio.

You can also run the following commands to perform lora training:
```bash
# Training StableAvatar-1