<div align="center">

<p align="center">
  <img src="assets/logo2.jpg" alt="InfinteTalk" width="440"/>
</p>

# InfiniteTalk: Generate Stunning Talking Head Videos from Audio - Introducing InfiniteTalk!

[Original Repository](https://github.com/MeiGen-AI/InfiniteTalk)

</div>

> **InfiniteTalk** revolutionizes video creation by generating high-quality talking head videos from audio, enabling both video-to-video and image-to-video generation.

<p align="center">
  <img src="assets/pipeline.png">
</p>

## Key Features

*   💬 **Sparse-Frame Video Dubbing:** Synchronizes lips, head movements, body posture, and facial expressions with the audio.
*   ⏱️ **Infinite-Length Generation:** Supports the creation of videos with unlimited duration.
*   ✨ **Enhanced Stability:** Reduces hand and body distortions compared to previous methods.
*   🚀 **Superior Lip Synchronization:** Achieves highly accurate lip movements matching the audio.

## What's New

*   **August 19, 2025:** Release of the [Technique-Report](https://arxiv.org/abs/2508.14033), model weights, and code, including Gradio and [ComfyUI](https://github.com/MeiGen-AI/InfiniteTalk/tree/comfyui) support.
*   **August 19, 2025:** Launch of the project page: [https://meigen-ai.github.io/InfiniteTalk/](https://meigen-ai.github.io/InfiniteTalk/)

## Community Works

*   [Wan2GP](https://github.com/deepbeepmeep/Wan2GP/): Integration by [deepbeepmeep](https://github.com/deepbeepmeep) for low VRAM optimization, video editing options, and integration of other models (MMaudio support, Qwen Image Edit, etc.).
*   [ComfyUI](https://github.com/kijai/ComfyUI-WanVideoWrapper): Thanks to [kijai](https://github.com/kijai) for ComfyUI support.

## Todo List

*   [x] Release the technical report
*   [x] Inference
*   [x] Checkpoints
*   [x] Multi-GPU Inference
*   [ ] Inference acceleration
    *   [x] TeaCache
    *   [x] int8 quantization
    *   [ ] LCM distillation
    *   [ ] Sparse Attention
*   [x] Run with very low VRAM
*   [x] Gradio demo
*   [x] ComfyUI

## Video Demos

### Video-to-video (HQ videos can be found on [Google Drive](https://drive.google.com/drive/folders/1BNrH6GJZ2Wt5gBuNLmfXZ6kpqb9xFPjU?usp=sharing) )

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

### Image-to-video

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
          <video src="https://github.com/user-attachments/assets/3263c5e1-9f98-4b9b-8688-b3e497460a76" width="320" controls loop></video>
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

## Quick Start

### Installation

1.  **Create Conda Environment and Install Dependencies:**
    ```bash
    conda create -n multitalk python=3.10
    conda activate multitalk
    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
    pip install -U xformers==0.0.28 --index-url https://download.pytorch.org/whl/cu121
    ```

2.  **Install Flash-Attention:**
    ```bash
    pip install misaki[en]
    pip install ninja 
    pip install psutil 
    pip install packaging
    pip install wheel
    pip install flash_attn==2.7.4.post1
    ```

3.  **Install Additional Dependencies:**
    ```bash
    pip install -r requirements.txt
    conda install -c conda-forge librosa
    ```

4.  **Install FFmpeg:**
    ```bash
    conda install -c conda-forge ffmpeg
    ```
    or
    ```bash
    sudo yum install ffmpeg ffmpeg-devel
    ```

### Model Preparation

1.  **Download Models:**

    | Model                       | Download Link                                                                 | Notes               |
    | --------------------------- | ----------------------------------------------------------------------------- | ------------------- |
    | Wan2.1-I2V-14B-480P         | [🤗 Hugging Face](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P)         | Base Model          |
    | chinese-wav2vec2-base     | [🤗 Hugging Face](https://huggingface.co/TencentGameMate/chinese-wav2vec2-base) | Audio Encoder       |
    | MeiGen-InfiniteTalk       | [🤗 Hugging Face](https://huggingface.co/MeiGen-AI/InfiniteTalk)             | Audio Condition Weights |

    Download models using huggingface-cli:
    ```bash
    huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./weights/Wan2.1-I2V-14B-480P
    huggingface-cli download TencentGameMate/chinese-wav2vec2-base --local-dir ./weights/chinese-wav2vec2-base
    huggingface-cli download TencentGameMate/chinese-wav2vec2-base model.safetensors --revision refs/pr/1 --local-dir ./weights/chinese-wav2vec2-base
    huggingface-cli download MeiGen-AI/InfiniteTalk --local-dir ./weights/InfiniteTalk
    ```

### Quick Inference

*   **Resolution:** Compatible with both 480P and 720P.
*   **Lip Synchronization:** Audio CFG works optimally between 3–5. Increase for better sync.
*   **FusionX LoRA:** Enables faster inference and higher quality but may increase color shift over time and reduce ID preservation.
*   **V2V Generation:** Supports unlimited-length generation. Model mimics original camera movements. Use SDEdit for improved accuracy but with color shift.
*   **I2V Generation:** Generates quality results from a single image for up to 1 minute. Color shifts may occur beyond 1 minute. For high-quality generation beyond 1 min you can copy the image to a video by translating or zooming in the image. See the [convert image to video script](https://github.com/MeiGen-AI/InfiniteTalk/blob/main/tools/convert_img_to_video.py).
*   **Quantization Model:** Use the quantization model to reduce memory usage if inference is killed due to insufficient memory.

#### Usage of InfiniteTalk
```
--mode streaming: long video generation.
--mode clip: generate short video with one chunk. 
--use_teacache: run with TeaCache.
--size infinitetalk-480: generate 480P video.
--size infinitetalk-720: generate 720P video.
--use_apg: run with APG.
--teacache_thresh: A coefficient used for TeaCache acceleration
—-sample_text_guide_scale： When not using LoRA, the optimal value is 5. After applying LoRA, the recommended value is 1.
—-sample_audio_guide_scale： When not using LoRA, the optimal value is 4. After applying LoRA, the recommended value is 2.
—-sample_audio_guide_scale： When not using LoRA, the optimal value is 4. After applying LoRA, the recommended value is 2.
--max_frame_num: The max frame length of the generated video, the default is 40 seconds(1000 frames).
```

#### 1. Inference

##### 1) Run with single GPU

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

##### 2) Run with 720P

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

##### 3) Run with very low VRAM

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

##### 4) Multi-GPU inference

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

##### 5) Multi-Person animation

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

#### 2. Run with FusioniX or Lightx2v (Require only 4~8 steps)

[FusioniX](https://huggingface.co/vrgamedevgirl84/Wan14BT2VFusioniX/blob/main/FusionX_LoRa/Wan2.1_I2V_14B_FusionX_LoRA.safetensors) and [lightx2v](https://huggingface.co/Kijai/WanVideo_comfy/blob/main/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors)

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

#### 3. Run with the quantization model (Only support run with single gpu)

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

#### 4. Run with Gradio

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

If you use InfiniteTalk in your research, please cite the following:

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

This project is licensed under the [Apache 2.0 License](LICENSE).