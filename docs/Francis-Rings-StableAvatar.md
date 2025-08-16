# StableAvatar: Generate Infinite-Length Audio-Driven Avatar Videos

StableAvatar revolutionizes audio-driven avatar video generation, enabling the creation of high-fidelity, identity-preserving videos of **unlimited length**.  Check out the original repo [here](https://github.com/Francis-Rings/StableAvatar)!

[Project Page](https://francis-rings.github.io/StableAvatar) | [Paper](https://arxiv.org/abs/2508.08248) | [Hugging Face Model](https://huggingface.co/FrancisRing/StableAvatar/tree/main) | [YouTube Demo](https://www.youtube.com/watch?v=6lhvmbzvv3Y) | [Bilibili Demo](https://www.bilibili.com/video/BV1hUt9z4EoQ)

*Shuyuan Tu<sup>1</sup>, Yueming Pan<sup>3</sup>, Yinming Huang<sup>1</sup>, Xintong Han<sup>4</sup>, Zhen Xing<sup>1</sup>, Qi Dai<sup>2</sup>, Chong Luo<sup>2</sup>, Zuxuan Wu<sup>1</sup>, Yu-Gang Jiang<sup>1</sup>
<br/>
[<sup>1</sup>Fudan University; <sup>2</sup>Microsoft Research Asia; <sup>3</sup>Xi'an Jiaotong University; <sup>4</sup>Tencent Inc]

---
**Key Features:**

*   **Infinite-Length Video Generation:** Create videos of any duration without compromising quality.
*   **Identity Preservation:** Maintains consistent facial identity throughout the entire video.
*   **End-to-End Synthesis:**  Generates videos directly, *without* reliance on post-processing tools like face-swapping or restoration.
*   **High-Fidelity Output:** Delivers high-quality avatar animations with natural audio synchronization.
*   **Time-step-aware Audio Adapter:** Prevents error accumulation via time-step-aware modulation.
*   **Audio Native Guidance Mechanism:** Enhance the audio synchronization by leveraging the diffusion’s own evolving joint audio-latent prediction as a dynamic guidance signal.
*   **Dynamic Weighted Sliding-window Strategy:** Fuses latent over time for smooth videos.

---

## Demo Videos

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

---
## Overview

<p align="center">
  <img src="assets/figures/framework.jpg" alt="model architecture" width="1280"/>
  </br>
  <i>The overview of the framework of StableAvatar.</i>
</p>
StableAvatar overcomes limitations in existing audio-driven avatar generation models by introducing a novel architecture for synthesizing high-quality, infinite-length videos.  The architecture employs several key features, including time-step-aware audio adapter,  Audio Native Guidance Mechanism, and Dynamic Weighted Sliding-window Strategy, to address error accumulation and enhance audio synchronization and smoothness of the infinite-length videos.

---

## What's New

*   **[2025-8-16]:** 🔥 Release of finetuning codes and lora training/finetuning codes! Other codes will be public as soon as possible. Stay tuned!
*   **[2025-8-15]:** 🔥 StableAvatar can run on Gradio Interface. Thanks @[gluttony-10](https://space.bilibili.com/893892) for the contribution!
*   **[2025-8-15]:** 🔥 StableAvatar can run on [ComfyUI](https://github.com/smthemex/ComfyUI_StableAvatar). Thanks @[smthemex](https://github.com/smthemex) for the contribution.
*   **[2025-8-13]:** 🔥 Added changes to run StableAvatar on the new Blackwell series Nvidia chips, including the RTX 6000 Pro.
*   **[2025-8-11]:** 🔥 The project page, code, technical report and [a basic model checkpoint](https://huggingface.co/FrancisRing/StableAvatar/tree/main) are released. Further lora training codes, the evaluation dataset and StableAvatar-pro will be released very soon. Stay tuned!

---

## Quickstart Guide

Get started with StableAvatar and generate your own infinite-length avatar videos!

### 🧱 Environment Setup

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
# Optional to install flash_attn to accelerate attention computation
pip install flash_attn
```

#### 🧱 Environment setup for Blackwell series chips

```bash
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
# Optional to install flash_attn to accelerate attention computation
pip install flash_attn
```

### 🧱 Download Weights

If you encounter connection issues with Hugging Face, you can utilize the mirror endpoint by setting the environment variable: `export HF_ENDPOINT=https://hf-mirror.com`.
Please download weights manually as follows:

```bash
pip install "huggingface_hub[cli]"
cd StableAvatar
mkdir checkpoints
huggingface-cli download FrancisRing/StableAvatar --local-dir ./checkpoints
```

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

Extract audio from your video files:

```bash
python audio_extractor.py --video_path="path/test/video.mp4" --saved_audio_path="path/test/audio.wav"
```

### 🧱 Vocal Separation (Optional)

Separate vocals from background music for improved lip-sync:

```bash
pip install audio-separator
python vocal_seperator.py --audio_separator_model_file="path/StableAvatar/checkpoints/Kim_Vocal_2.onnx" --audio_file_path="path/test/audio.wav" --saved_vocal_path="path/test/vocal.wav"
```

### 🧱 Base Model Inference

Run the inference script:

```bash
bash inference.sh
```
Wan2.1-1.3B-based StableAvatar supports audio-driven avatar video generation at three different resolution settings: 512x512, 480x832, and 832x480. You can modify "--width" and "--height" in `inference.sh` to set the resolution of the animation. "--output_dir" in `inference.sh` refers to the saved path of the generated animation. "--validation_reference_path", "--validation_driven_audio_path", and "--validation_prompts" in `inference.sh` refer to the path of the given reference image, the path of the given audio, and the text prompts respectively.
Prompts are also very important. It is recommended to `[Description of first frame]-[Description of human behavior]-[Description of background (optional)]`.
"--pretrained_model_name_or_path", "--pretrained_wav2vec_path", and "--transformer_path" in `inference.sh` are the paths of pretrained Wan2.1-1.3B weights, pretrained Wav2Vec2.0 weights, and pretrained StableAvatar weights, respectively.
"--sample_steps", "--overlap_window_length", and "--clip_sample_n_frames" refer to the total number of inference steps, the overlapping context length between two context windows, and the synthesized frame number in a batch/context window, respectively. 
Notably, the recommended `--sample_steps` range is [30-50], more steps bring higher quality. The recommended `--overlap_window_length` range is [5-15], as longer overlapping length results in higher quality and slower inference speed.
"--sample_text_guide_scale" and "--sample_audio_guide_scale" are Classify-Free-Guidance scale of text prompt and audio. The recommended range for prompt and audio cfg is `[3-6]`. You can increase the audio cfg to facilitate the lip synchronization with audio.
Refer to `examples` folder for sample configurations.

You can also launch a Gradio interface:

```bash
python app.py
```

#### 💡 Tips:

*   Use the `--transformer_path` in `inference.sh` to switch different model versions of  Wan2.1-1.3B-based StableAvatar.
*   Control GPU memory usage with the `--GPU_memory_mode` flag.
*   Use multi-GPU inference with `--ulysses_degree`, `--ring_degree`, and `--fsdp_dit`.
*   Combine the video output with the audio using FFmpeg:

    ```bash
    ffmpeg -i video_without_audio.mp4 -i /path/audio.wav -c:v copy -c:a aac -shortest /path/output_with_audio.mp4
    ```

### 🧱 Model Training

Training details and dataset structure can be found in the original README.  It is also worth noting that training StableAvatar requires approximately 50GB of VRAM due to the mixed-resolution (480x832 and 832x480) training pipeline. However, if you train StableAvatar exclusively on 512x512 videos, the VRAM requirement is reduced to approximately 40GB.
Also, The backgrounds of the selected training videos should remain static, as this helps the diffusion model calculate accurate reconstruction loss. The audio should be clear and free from excessive background noise.

*   **Training Commands:**

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

*   **Wan2.1-14B Training:**

    ```bash
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
*  **720P Training:**  Modify height and width in `train_1B_square.py`/`train_1B_vec_rec.py`/`train_14B.py` to train at 720p.

### 🧱 Model Finetuning

*   **Finetuning Commands:**
    ```bash
    # Finetuning StableAvatar on a mixed resolution setting (480x832 and 832x480) in a single machine
    bash finetune_1B_rec_vec.sh
    # Finetuning StableAvatar on a mixed resolution setting (480x832 and 832x480) in multiple machines
    bash finetune_1B_rec_vec_64.sh
    ```
    ```bash
    # Lora-Finetuning StableAvatar-1.3B on a mixed resolution setting (480x832 and 832x480) in a single machine
    bash finetune_1B_rec_vec_lora.sh
    ```
    You can modify `--rank` and `--network_alpha` to control the quality of your lora training/finetuning.

### 🧱 VRAM Requirement and Runtime

For the 5s video (480x832, fps=25), the basic model (--GPU_memory_mode="model_full_load") requires approximately 18GB VRAM and finishes in 3 minutes on a 4090 GPU.

**🔥🔥Theoretically, StableAvatar is capable of synthesizing hours of video without significant quality degradation; however, the 3D VAE decoder demands significant GPU memory, especially when decoding 10k+ frames. You have the option to run the VAE decoder on CPU.🔥🔥**

---

## Contact and Citation

For any questions or suggestions, please feel free to contact me:

*   Email: [francisshuyuan@gmail.com](mailto:francisshuyuan@gmail.com)

If you found this project helpful, please consider giving a star to the repository and citing our work:

```bib
@article{tu2025stableavatar,
  title={StableAvatar: Infinite-Length Audio-Driven Avatar Video Generation},
  author={Tu, Shuyuan and Pan, Yueming and Huang, Yinming and Han, Xintong and Xing, Zhen and Dai, Qi and Luo, Chong and Wu, Zuxuan and Jiang Yu-Gang},
  journal={arXiv preprint arXiv:2508.08248},
  year={2025}
}
```