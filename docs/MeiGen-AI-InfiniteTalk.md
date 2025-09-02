<div align="center">

<p align="center">
  <img src="assets/logo2.jpg" alt="InfinteTalk" width="440"/>
</p>

# InfiniteTalk: Generate Unlimited Talking Videos with Audio

InfiniteTalk revolutionizes video creation by generating high-quality talking videos from audio, enabling realistic lip synchronization and expression alignment; [Check out the original repo](https://github.com/MeiGen-AI/InfiniteTalk).

</div>

## Key Features

*   💬 **Sparse-Frame Video Dubbing:** Synchronizes not only lips but also head movements, body posture, and facial expressions with the audio.
*   ⏱️ **Infinite-Length Generation:** Supports the creation of videos with unlimited duration.
*   ✨ **Enhanced Stability:** Produces videos with reduced hand/body distortions compared to other models.
*   🚀 **Superior Lip Accuracy:** Achieves improved lip synchronization, resulting in more realistic video output.
*   🖼️ **Image-to-Video Support**: Generate talking videos from a single image and audio input.

## Latest News

*   **August 19, 2025:** Release of the [Technique-Report](https://arxiv.org/abs/2508.14033), weights, and code. Gradio and [ComfyUI](https://github.com/MeiGen-AI/InfiniteTalk/tree/comfyui) branches have been released.
*   **August 19, 2025:** Project page released: [Project Page](https://meigen-ai.github.io/InfiniteTalk/)

## Community Works

*   **Wan2GP:** Integration in [Wan2GP](https://github.com/deepbeepmeep/Wan2GP) by [deepbeepmeep](https://github.com/deepbeepmeep), offering optimization for low VRAM, and multiple video editing features.
*   **ComfyUI:** ComfyUI support from [kijai](https://github.com/kijai) via [ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper)

## Video Demos

### Video-to-video (HQ videos can be found on [Google Drive](https://drive.google.com/drive/folders/1BNrH6GJZ2Wt5gBuNLmfXZ6kpqb9xFPjU?usp=sharing))

| Video 1                                                          | Video 2                                                          | Video 3                                                          | Video 4                                                          |
| :-------------------------------------------------------------- | :-------------------------------------------------------------- | :-------------------------------------------------------------- | :-------------------------------------------------------------- |
| <video src="https://github.com/user-attachments/assets/04f15986-8de7-4bb4-8cde-7f7f38244f9f" width="320" controls loop></video> | <video src="https://github.com/user-attachments/assets/1500f72e-a096-42e5-8b44-f887fa8ae7cb" width="320" controls loop></video> | <video src="https://github.com/user-attachments/assets/28f484c2-87dc-4828-a9e7-cb963da92d14" width="320" controls loop></video> | <video src="https://github.com/user-attachments/assets/665fabe4-3e24-4008-a0a2-a66e2e57c38b" width="320" controls loop></video> |

### Image-to-video

| Video 1                                                          | Video 2                                                          | Video 3                                                          | Video 4                                                          |
| :-------------------------------------------------------------- | :-------------------------------------------------------------- | :-------------------------------------------------------------- | :-------------------------------------------------------------- |
| <video src="https://github.com/user-attachments/assets/7e4a4dad-9666-4896-8684-2acb36aead59" width="320" controls loop></video> | <video src="https://github.com/user-attachments/assets/bd6da665-f34d-4634-ae94-b4978f92ad3a" width="320" controls loop></video> | <video src="https://github.com/user-attachments/assets/510e2648-82db-4648-aaf3-6542303dbe22" width="320" controls loop></video> | <video src="https://github.com/user-attachments/assets/27bb087b-866a-4300-8a03-3bbb4ce3ddf9" width="320" controls loop></video> |
| <video src="https://github.com/user-attachments/assets/3263c5e1-9f98-4b9b-8688-b3e497460a76" width="320" controls loop></video> | <video src="https://github.com/user-attachments/assets/5ff3607f-90ec-4eee-b964-9d5ee3028005" width="320" controls loop></video> | <video src="https://github.com/user-attachments/assets/e504417b-c8c7-4cf0-9afa-da0f3cbf3726" width="320" controls loop></video> | <video src="https://github.com/user-attachments/assets/56aac91e-c51f-4d44-b80d-7d115e94ead7" width="320" controls loop></video> |

## Quick Start

### 🛠️ Installation

#### 1. Create a Conda Environment and Install Dependencies:

```bash
conda create -n infinitetalk python=3.10
conda activate infinitetalk
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install -U xformers==0.0.28 --index-url https://download.pytorch.org/whl/cu121
pip install misaki[en] ninja psutil packaging wheel flash_attn==2.7.4.post1
pip install -r requirements.txt
conda install -c conda-forge librosa
conda install -c conda-forge ffmpeg
```

#### 2. FFmpeg Installation (Alternative):

```bash
sudo yum install ffmpeg ffmpeg-devel
```

### 🧱 Model Preparation

#### 1. Download Models

| Model Name                      | Download Link                                                                                                | Notes                         |
| :------------------------------ | :----------------------------------------------------------------------------------------------------------- | :---------------------------- |
| Wan2.1-I2V-14B-480P             | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P)                                                | Base Model                    |
| chinese-wav2vec2-base         | 🤗 [Huggingface](https://huggingface.co/TencentGameMate/chinese-wav2vec2-base)                                       | Audio Encoder                 |
| MeiGen-InfiniteTalk             | 🤗 [Huggingface](https://huggingface.co/MeiGen-AI/InfiniteTalk)                                                    | Audio Condition Weights       |

#### 2. Download Models using huggingface-cli:

```bash
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./weights/Wan2.1-I2V-14B-480P
huggingface-cli download TencentGameMate/chinese-wav2vec2-base --local-dir ./weights/chinese-wav2vec2-base
huggingface-cli download TencentGameMate/chinese-wav2vec2-base model.safetensors --revision refs/pr/1 --local-dir ./weights/chinese-wav2vec2-base
huggingface-cli download MeiGen-AI/InfiniteTalk --local-dir ./weights/InfiniteTalk
```

### 🔑 Quick Inference

*   **Lip Synchronization Accuracy:** Adjust `Audio CFG` (3–5 optimal). Higher values improve lip sync.
*   **FusionX LoRA:** Faster inference, but can cause color shifts.
*   **V2V Generation:** For unlimited-length. Mimics original camera movement. Consider using SDEdit for improved camera accuracy (but it may introduce color shifts).
*   **I2V Generation:** Great results for up to 1 minute from a single image.  Consider using a script to convert the image to a video, especially for long clips: [Convert Image to Video Script](https://github.com/MeiGen-AI/InfiniteTalk/blob/main/tools/convert_img_to_video.py).
*   **Quantization Model:** For reduced memory usage, if experiencing out-of-memory issues.

#### 1. Run Inference

##### 1) Single GPU

```bash
python generate_infinitetalk.py \
    --ckpt_dir weights/Wan2.1-I2V-14B-480P \
    --wav2vec_dir 'weights/chinese-wav2vec2-base' \
    --infinitetalk_dir weights/InfiniteTalk/single/infinitetalk.safetensors \
    --input_json examples/single_example_image.json \
    --size infinitetalk-480 \
    --sample_steps 40 \
    --mode streaming \
    --motion_frame 9 \
    --save_file infinitetalk_res
```

##### 2) 720P

```bash
python generate_infinitetalk.py \
    --ckpt_dir weights/Wan2.1-I2V-14B-480P \
    --wav2vec_dir 'weights/chinese-wav2vec2-base' \
    --infinitetalk_dir weights/InfiniteTalk/single/infinitetalk.safetensors \
    --input_json examples/single_example_image.json \
    --size infinitetalk-720 \
    --sample_steps 40 \
    --mode streaming \
    --motion_frame 9 \
    --save_file infinitetalk_res_720p
```

##### 3) Low VRAM

```bash
python generate_infinitetalk.py \
    --ckpt_dir weights/Wan2.1-I2V-14B-480P \
    --wav2vec_dir 'weights/chinese-wav2vec2-base' \
    --infinitetalk_dir weights/InfiniteTalk/single/infinitetalk.safetensors \
    --input_json examples/single_example_image.json \
    --size infinitetalk-480 \
    --sample_steps 40 \
    --num_persistent_param_in_dit 0 \
    --mode streaming \
    --motion_frame 9 \
    --save_file infinitetalk_res_lowvram
```

##### 4) Multi-GPU

```bash
GPU_NUM=8
torchrun --nproc_per_node=$GPU_NUM --standalone generate_infinitetalk.py \
    --ckpt_dir weights/Wan2.1-I2V-14B-480P \
    --wav2vec_dir 'weights/chinese-wav2vec2-base' \
    --infinitetalk_dir weights/InfiniteTalk/single/infinitetalk.safetensors \
    --dit_fsdp --t5_fsdp \
    --ulysses_size=$GPU_NUM \
    --input_json examples/single_example_image.json \
    --size infinitetalk-480 \
    --sample_steps 40 \
    --mode streaming \
    --motion_frame 9 \
    --save_file infinitetalk_res_multigpu
```

##### 5) Multi-Person Animation

```bash
python generate_infinitetalk.py \
    --ckpt_dir weights/Wan2.1-I2V-14B-480P \
    --wav2vec_dir 'weights/chinese-wav2vec2-base' \
    --infinitetalk_dir weights/InfiniteTalk/multi/infinitetalk.safetensors \
    --input_json examples/multi_example_image.json \
    --size infinitetalk-480 \
    --sample_steps 40 \
    --num_persistent_param_in_dit 0 \
    --mode streaming \
    --motion_frame 9 \
    --save_file infinitetalk_res_multiperson
```

#### 2. FusioniX or Lightx2v (Requires only 4-8 Steps)

*   [FusioniX LoRA](https://huggingface.co/vrgamedevgirl84/Wan14BT2VFusioniX/blob/main/FusionX_LoRa/Wan2.1_I2V_14B_FusionX_LoRA.safetensors) (8 steps)
*   [lightx2v](https://huggingface.co/Kijai/WanVideo_comfy/blob/main/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors) (4 steps)

```bash
python generate_infinitetalk.py \
    --ckpt_dir weights/Wan2.1-I2V-14B-480P \
    --wav2vec_dir 'weights/chinese-wav2vec2-base' \
    --infinitetalk_dir weights/InfiniteTalk/single/infinitetalk.safetensors \
    --lora_dir weights/Wan2.1_I2V_14B_FusionX_LoRA.safetensors \
    --input_json examples/single_example_image.json \
    --lora_scale 1.0 \
    --size infinitetalk-480 \
    --sample_text_guide_scale 1.0 \
    --sample_audio_guide_scale 2.0 \
    --sample_steps 8 \
    --mode streaming \
    --motion_frame 9 \
    --sample_shift 2 \
    --num_persistent_param_in_dit 0 \
    --save_file infinitetalk_res_lora
```

#### 3. Quantization Model (Single GPU)

```bash
python generate_infinitetalk.py \
    --ckpt_dir weights/Wan2.1-I2V-14B-480P \
    --wav2vec_dir 'weights/chinese-wav2vec2-base' \
    --infinitetalk_dir weights/InfiniteTalk/single/infinitetalk.safetensors \
    --input_json examples/single_example_image.json \
    --size infinitetalk-480 \
    --sample_steps 40 \
    --mode streaming \
    --quant fp8 \
    --quant_dir weights/InfiniteTalk/quant_models/infinitetalk_single_fp8.safetensors \
    --motion_frame 9 \
    --num_persistent_param_in_dit 0 \
    --save_file infinitetalk_res_quant
```

#### 4. Gradio Demo

```bash
python app.py \
    --ckpt_dir weights/Wan2.1-I2V-14B-480P \
    --wav2vec_dir 'weights/chinese-wav2vec2-base' \
    --infinitetalk_dir weights/InfiniteTalk/single/infinitetalk.safetensors \
    --num_persistent_param_in_dit 0 \
    --motion_frame 9
```

or

```bash
python app.py \
    --ckpt_dir weights/Wan2.1-I2V-14B-480P \
    --wav2vec_dir 'weights/chinese-wav2vec2-base' \
    --infinitetalk_dir weights/InfiniteTalk/multi/infinitetalk.safetensors \
    --num_persistent_param_in_dit 0 \
    --motion_frame 9
```

## Citation

```bibtex
@misc{yang2025infinitetalkaudiodrivenvideogeneration,
      title={InfiniteTalk: Audio-driven Video Generation for Sparse-Frame Video Dubbing},
      author={Shaoshu Yang and Zhe Kong and Feng Gao and Meng Cheng and Xiangyu Liu and Yong Zhang and Zhuoliang Kang and Wenhan Luo and Xunliang Cai and Ran He and Xiaoming Wei},
      year={2025},
      eprint={2508.14033},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2508.14033},
}
```

## License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.