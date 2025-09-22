<div align="center">

<p align="center">
  <img src="assets/logo2.jpg" alt="InfiniteTalk" width="440"/>
</p>

# InfiniteTalk: Transform Audio into Stunning Talking Videos

[Original Repository](https://github.com/MeiGen-AI/InfiniteTalk)

</div>

> **InfiniteTalk** revolutionizes video creation, enabling users to generate unlimited-length talking videos driven by audio and images.

InfiniteTalk is a cutting-edge framework designed for **audio-driven video generation**, specifically tailored for sparse-frame video dubbing. This innovative model allows you to create realistic and engaging videos by synchronizing lip movements, head gestures, body posture, and facial expressions with an audio track.  It also supports image-to-video generation.

<p align="center">
  <img src="assets/pipeline.png">
</p>

## Key Features

*   **Audio-Driven Dubbing:**  Accurately synchronizes lip movements, head motions, body language, and facial expressions with the provided audio.
*   **Infinite-Length Video Generation:** Supports the creation of videos with virtually unlimited duration.
*   **Enhanced Stability:**  Minimizes hand and body distortions compared to previous methods, ensuring a more polished final result.
*   **Superior Lip Synchronization:**  Achieves state-of-the-art lip synchronization for a natural and convincing appearance.
*   **Image-to-Video Capabilities:** Transforms a single image and audio into a dynamic talking video.

## What's New

*   **August 19, 2025:** Release of the [Technique-Report](https://arxiv.org/abs/2508.14033) along with model weights and code, including Gradio and [ComfyUI](https://github.com/MeiGen-AI/InfiniteTalk/tree/comfyui) support.
*   **August 19, 2025:**  Launch of the dedicated [project page](https://meigen-ai.github.io/InfiniteTalk/).

## Community Contributions

*   **Wan2GP Integration:** Thanks to [deepbeepmeep](https://github.com/deepbeepmeep) for integrating InfiniteTalk into Wan2GP, offering low VRAM optimization and extensive video editing capabilities, including MMaudio support and Qwen Image Edit.
*   **ComfyUI Support:** Thanks to [kijai](https://github.com/kijai) for providing support through ComfyUI.

## To-Do List

*   [x] Release the technical report
*   [x] Inference implementation
*   [x] Checkpoint availability
*   [x] Multi-GPU Inference
*   [ ] Inference Acceleration
    *   [x] TeaCache
    *   [x] int8 quantization
    *   [ ] LCM distillation
    *   [ ] Sparse Attention
*   [x] Low VRAM operation
*   [x] Gradio demo
*   [x] ComfyUI integration

## Video Demonstrations

### Video-to-Video (HQ videos available on [Google Drive](https://drive.google.com/drive/folders/1BNrH6GJZ2Wt5gBuNLmfXZ6kpqb9xFPjU?usp=sharing))

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/04f15986-8de7-4bb4-8cde-7f7f38244f9f" width="320" controls loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/1500f72e-a096-42e5-8b44-f887fa8ae7cb" width="320" controls loop></video>
     </td>
     <td>
          <video src="https://github.com/user-attachments/assets/28f484c2-87dc-4828-a9e7-cb963da92d14" width="320" controls loop></video>
     </td>
     <td>
          <video src="https://github.com/user-attachments/assets/665fabe4-3e24-4008-a0a2-a66e2e57c38b" width="320" controls loop></video>
     </td>
  </tr>
</table>

### Image-to-Video

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/7e4a4dad-9666-4896-8684-2acb36aead59" width="320" controls loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/bd6da665-f34d-4634-ae94-b4978f92ad3a" width="320" controls loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/510e2648-82db-4648-aaf3-6542303dbe22" width="320" controls loop></video>
     </td>
     <td>
          <video src="https://github.com/user-attachments/assets/27bb087b-866a-4300-8a03-3bbb4ce3ddf9" width="320" controls loop></video>
     </td>
     
  </tr>
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/3263c5e2-9f98-4b9b-8688-b3e497460a76" width="320" controls loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/5ff3607f-90ec-4eee-b964-9d5ee3028005" width="320" controls loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/e504417b-c8c7-4cf0-9afa-da0f3cbf3726" width="320" controls loop></video>
     </td>
     <td>
          <video src="https://github.com/user-attachments/assets/56aac91e-c51f-4d44-b80d-7d115e94ead7" width="320" controls loop></video>
     </td>
     
  </tr>
</table>

## Quick Start Guide

### Installation

1.  **Create Conda Environment and Install Dependencies:**

```bash
conda create -n multitalk python=3.10
conda activate multitalk
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install -U xformers==0.0.28 --index-url https://download.pytorch.org/whl/cu121
pip install misaki[en]
pip install ninja 
pip install psutil 
pip install packaging
pip install wheel
pip install flash_attn==2.7.4.post1
pip install -r requirements.txt
conda install -c conda-forge librosa
conda install -c conda-forge ffmpeg
```
   or
```bash
sudo yum install ffmpeg ffmpeg-devel
```
2.  **Model Preparation**

    *   **Download Models:** Use `huggingface-cli` to download the necessary models:

    ```bash
    huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./weights/Wan2.1-I2V-14B-480P
    huggingface-cli download TencentGameMate/chinese-wav2vec2-base --local-dir ./weights/chinese-wav2vec2-base
    huggingface-cli download TencentGameMate/chinese-wav2vec2-base model.safetensors --revision refs/pr/1 --local-dir ./weights/chinese-wav2vec2-base
    huggingface-cli download MeiGen-AI/InfiniteTalk --local-dir ./weights/InfiniteTalk
    ```

### Quick Inference Instructions

*   **Key Parameters:**
    *   `--mode`: `"streaming"` for long videos; `"clip"` for short videos.
    *   `--use_teacache`: Enables TeaCache for acceleration.
    *   `--size`: `"infinitetalk-480"` (default) or `"infinitetalk-720"` for resolution.
    *   `--use_apg`:  Activates APG for certain optimizations.
    *   `--teacache_thresh`: TeaCache coefficient.
    *   `--sample_text_guide_scale`: Text guidance scale (5 without LoRA, 1 with LoRA).
    *   `--sample_audio_guide_scale`: Audio guidance scale (4 without LoRA, 2 with LoRA).
    *   `--max_frame_num`: Maximum video frame length (default 40 seconds/1000 frames).

#### 1. Single GPU Inference

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

#### 2. 720P Inference

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

#### 3. Low VRAM Inference

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

#### 4. Multi-GPU Inference

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

#### 5. Multi-Person Animation

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

#### 6. Run with FusioniX or Lightx2v (LoRA)

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

#### 7. Quantization Model Inference

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

#### 8. Run Gradio Demo

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

If you use InfiniteTalk in your research, please cite it:

```
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

The models are licensed under the Apache 2.0 License.  You are responsible for your use of the models and must adhere to the license terms, especially regarding content that violates laws or harms individuals/groups.