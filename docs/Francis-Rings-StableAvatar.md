# StableAvatar: Generate Infinite-Length Avatar Videos from Audio

**Bring your audio to life with StableAvatar, the groundbreaking technology for generating infinite-length, high-fidelity avatar videos!**  

[Project Page](https://francis-rings.github.io/StableAvatar) | [Paper (Arxiv)](https://arxiv.org/abs/2508.08248) | [Hugging Face Model](https://huggingface.co/FrancisRing/StableAvatar/tree/main) | [Hugging Face Demo](https://huggingface.co/spaces/YinmingHuang/StableAvatar) | [YouTube](https://www.youtube.com/watch?v=6lhvmbzvv3Y) | [Bilibili](https://www.bilibili.com/video/BV1hUt9z4EoQ)

**Authors:** Shuyuan Tu, Yueming Pan, Yinming Huang, Xintong Han, Zhen Xing, Qi Dai, Chong Luo, Zuxuan Wu, Yu-Gang Jiang

StableAvatar is an innovative approach to generate captivating avatar videos directly from audio inputs. Experience the power of infinite-length video synthesis with unparalleled quality and identity preservation. Our model overcomes the limitations of existing methods, delivering seamless, high-fidelity avatar animations without the need for post-processing techniques like face swapping or restoration.

**Key Features:**

*   ✅ **Infinite-Length Video Generation:** Produce videos of any length without quality degradation.
*   ✅ **ID-Preserving:** Maintain consistent identity throughout the entire video.
*   ✅ **High-Fidelity:** Generate high-quality videos with natural audio synchronization.
*   ✅ **End-to-End Synthesis:** Directly synthesize videos without face-related post-processing.
*   ✅ **Time-step-aware Audio Adapter:** Prevents error accumulation via time-step-aware modulation
*   ✅ **Audio Native Guidance Mechanism:** Enhances audio synchronization.
*   ✅ **Dynamic Weighted Sliding-window Strategy:** Fuses latent over time for a smoother experience.

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
  <span>StableAvatar generates <b>infinite-length</b> and <b>ID-preserving videos</b>. All videos are <b>directly synthesized</b> without the use of any face-related post-processing tools.</span>
</p>

<p align="center">
  <video src="https://github.com/user-attachments/assets/90691318-311e-40b9-9bd9-62db83ab1492" width="768" autoplay loop muted playsinline></video>
  <br/>
  <span>Comparison results between StableAvatar and SOTA models highlight the superior performance of StableAvatar in delivering <b>infinite-length, high-fidelity, identity-preserving avatar animation</b>.</span>
</p>

## 🖼️ Overview

<p align="center">
  <img src="assets/figures/framework.jpg" alt="model architecture" width="1280"/>
  </br>
  <i>The overview of the framework of StableAvatar.</i>
</p>

Current diffusion models for audio-driven avatar video generation struggle to synthesize long videos. StableAvatar addresses this by introducing a novel Time-step-aware Audio Adapter and a Dynamic Weighted Sliding-window Strategy. This enables the generation of smooth, high-quality, and infinite-length videos.

## 📰 News

*   **\[2025-9-8]**: 🔥  Check out our brand new demo! Watch the generated videos on [YouTube](https://www.youtube.com/watch?v=GH4hrxIis3Q) and [Bilibili](https://www.bilibili.com/video/BV1jGYPzqEux).
*   **\[2025-8-29]**: 🔥 StableAvatar public demo is now live on [Hugging Face Spaces](https://huggingface.co/spaces/YinmingHuang/StableAvatar). (Note: Due to long video generation times, the demo is currently accessible to <b>Hugging Face Pro</b> users only.)
*   **\[2025-8-18]**: 🔥 StableAvatar can run on [ComfyUI](https://github.com/smthemex/ComfyUI_StableAvatar) in just 10 steps, making it 3x faster. Thanks @[smthemex](https://github.com/smthemex) for the contribution.
*   **\[2025-8-16]**: 🔥 Finetuning codes and lora training/finetuning codes are released! Other codes will be public soon.
*   **\[2025-8-15]**: 🔥 StableAvatar can run on Gradio Interface. Thanks @[gluttony-10](https://space.bilibili.com/893892) for the contribution!
*   **\[2025-8-15]**: 🔥 StableAvatar can run on [ComfyUI](https://github.com/smthemex/ComfyUI_StableAvatar). Thanks @[smthemex](https://github.com/smthemex) for the contribution.
*   **\[2025-8-13]**: 🔥 Added changes to run StableAvatar on the new Blackwell series Nvidia chips, including the RTX 6000 Pro.
*   **\[2025-8-11]**: 🔥 The project page, code, technical report and [a basic model checkpoint](https://huggingface.co/FrancisRing/StableAvatar/tree/main) are released. Further lora training codes, the evaluation dataset and StableAvatar-pro will be released very soon.

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

## 🚀 Quickstart

Get started with the basic version of the model, which supports generating **infinite-length videos at 480x832 or 832x480 or 512x512 resolution**. If you have limited resources, you can adjust the number of animated frames or output resolution.

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

Download the necessary model weights. If you experience connection issues with Hugging Face, you can utilize a mirror by setting the environment variable: `export HF_ENDPOINT=https://hf-mirror.com`.

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

### 🧱 Audio Extraction

Extract audio from a video file:

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

Run the inference script:

```bash
bash inference.sh
```

Modify the parameters in `inference.sh` to customize the output, including resolution, prompts, and paths to the necessary weights. Prompts should follow the format: `[Description of first frame]-[Description of human behavior]-[Description of background (optional)]`.

For a Gradio interface, run:

```bash
python app.py
```

Example videos are available in `path/StableAvatar/examples`.

#### 💡 Tips

*   StableAvatar weights (Wan2.1-1.3B-based) come in two versions: `transformer3d-square.pt` and `transformer3d-rec-vec.pt`. Modify `--transformer_path` in `inference.sh` to switch between them.
*   Use `--GPU_memory_mode` in `inference.sh` to manage GPU memory usage: `model_full_load`, `sequential_cpu_offload`, `model_cpu_offload_and_qfloat8`, or `model_cpu_offload`.
*   For multi-GPU inference, modify `--ulysses_degree` and `--ring_degree` in `inference.sh`.
*   To get an MP4 file with audio, use `ffmpeg`:

    ```bash
    ffmpeg -i video_without_audio.mp4 -i /path/audio.wav -c:v copy -c:a aac -shortest /path/output_with_audio.mp4
    ```

### 🧱 Model Training

**🔥🔥 This training tutorial is also helpful if you are looking to train a conditioned Video Diffusion Transformer (DiT) model, such as Wan2.1. 🔥🔥**

Dataset organization:

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

StableAvatar uses mixed-resolution videos.

*   `talking_face_data/square`: 512x512 videos
*   `talking_face_data/vec`: 480x832 videos
*   `talking_face_data/rec`: 832x480 videos

Each video folder contains: `images`, `face_masks`, `lip_masks`, `sub_clip.mp4`, `audio.wav`

*   Extract frames from raw videos (speech) and store them in the subfolder `images` using `ffmpeg`.
*   Extract human face masks using the [StableAnimator repo](https://github.com/Francis-Rings/StableAnimator).
*   Extract human lip masks using:

    ```bash
    pip install mediapipe
    python lip_mask_extractor.py --folder_root="path/StableAvatar/talking_face_data/rec/singing" --start=1 --end=500
    ```

*   Extract audio using the Audio Extraction section.

Train StableAvatar by running the corresponding command based on your needs.

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

*   GPU configuration is modified in `path/StableAvatar/accelerate_config/accelerate_config_machine_1B_multiple.yaml` for multi-machine training.
*   The backgrounds of the selected training videos should remain static.
*   The audio should be clear and free from excessive background noise.

For training Wan2.1-14B-based models:

```bash
# Training StableAvatar on a mixed resolution setting (480x832, 832x480, and 512x512) in multiple machines
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./checkpoints/Wan2.1-I2V-14B-480P
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-720P --local-dir ./checkpoints/Wan2.1-I2V-14B-720P # Optional
bash train_14B.sh
```

*   Deepspeed configuration in  `path/StableAvatar/deepspeed_config/zero_stage2_config.json`
*   GPU configuration is in `path/StableAvatar/accelerate_config/accelerate_config_machine_14B_multiple.yaml`.

You can also perform lora training with:

```bash
# Training StableAvatar-1.3B on a mixed resolution setting (480x832 and 832x480) in a single machine
bash train_1B_rec_vec_lora.sh
# Training StableAvatar-1.3B on a mixed resolution setting (480x832 and 832x480) in multiple machines
bash train_1B_rec_vec_lora_64.sh
# Lora-Training StableAvatar-14B on a mixed resolution setting (480x832, 832x480, and 512x512) in multiple machines
bash train_14B_lora.sh
```

### 🧱 Model Finetuning

Finetune StableAvatar using:

```bash
# Finetuning StableAvatar on a mixed resolution setting (480x832 and 832x480) in a single machine
bash train_1B_rec_vec.sh
# Finetuning StableAvatar on a mixed resolution setting (480x832 and 832x480) in multiple machines
bash train_1B_rec_vec_64.sh
```

For lora finetuning:

```bash
# Lora-Finetuning StableAvatar-1.3B on a mixed resolution setting (480x832 and 832x480) in a single machine
bash train_1B_rec_vec_lora.sh
```

### 🧱 VRAM Requirement and Runtime

The basic model (--GPU_memory_mode="model_full_load") requires approximately 18GB VRAM and finishes in 3 minutes on a 4090 GPU for a 5s video (480x832, fps=25).

**🔥 Theoretically, StableAvatar is capable of synthesizing hours of video without significant quality degradation; however, the 3D VAE decoder demands significant GPU memory, especially when decoding 10k+ frames. You have the option to run the VAE decoder on CPU. 🔥**

## 📧 Contact

For suggestions or if you find our work helpful, please contact:

*   Email: francisshuyuan@gmail.com

If you find our work useful, please consider giving a star ⭐ to this GitHub repository and citing it ❤️:
```bib
@article{tu2025stableavatar,
  title={StableAvatar: Infinite-Length Audio-Driven Avatar Video Generation},
  author={Tu, Shuyuan and Pan, Yueming and Huang, Yinming and Han, Xintong and Xing, Zhen and Dai, Qi and Luo, Chong and Wu, Zuxuan and Jiang Yu-Gang},
  journal={arXiv preprint arXiv:2508.08248},
  year={2025}
}
```