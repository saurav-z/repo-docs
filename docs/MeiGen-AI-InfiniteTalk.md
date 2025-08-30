<div align="center">
<p align="center">
  <img src="assets/logo2.jpg" alt="InfiniteTalk" width="440"/>
</p>

<h1>InfiniteTalk: Transform Static Images and Videos into Dynamic, Talking Visuals with Audio</h1>

<p>
  <a href="https://meigen-ai.github.io/InfiniteTalk/"><img src="https://img.shields.io/badge/Project-Page-green"></a>
  <a href="https://arxiv.org/abs/2508.14033"><img src="https://img.shields.io/badge/Technique-Report-red"></a>
  <a href="https://huggingface.co/MeiGen-AI/InfiniteTalk"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue"></a>
</p>
</div>

> **InfiniteTalk enables the creation of unlimited-length talking videos from audio, offering both video-to-video dubbing and image-to-video generation capabilities.**

<p align="center">
  <img src="assets/pipeline.png" alt="InfiniteTalk Pipeline">
</p>

## Key Features

InfiniteTalk is a cutting-edge framework for audio-driven video generation, offering a range of features:

*   💬 **Sparse-Frame Video Dubbing:** Synchronizes lip movements, head movements, body posture, and facial expressions with the provided audio track.
*   ⏱️ **Infinite-Length Generation:**  Generates videos of unlimited duration, going beyond the limitations of traditional methods.
*   ✨ **Enhanced Stability:** Reduces distortions in hands and body compared to existing methods like MultiTalk, resulting in more realistic outputs.
*   🚀 **Superior Lip Accuracy:** Achieves higher fidelity in lip synchronization, leading to more natural-looking results.
*   🖼️ **Image-to-Video Generation:** Transforms a single image and audio into a dynamic video.

## What's New

*   **August 19, 2025:** Release of the [Technical Report](https://arxiv.org/abs/2508.14033), model weights, and code, including Gradio and ComfyUI ( [ComfyUI branch](https://github.com/MeiGen-AI/InfiniteTalk/tree/comfyui) ) support.
*   **August 19, 2025:** Launch of the [Project Page](https://meigen-ai.github.io/InfiniteTalk/).

## Community Contributions

*   **Wan2GP:** Thanks to [deepbeepmeep](https://github.com/deepbeepmeep) for integrating InfiniteTalk into [Wan2GP](https://github.com/deepbeepmeep/Wan2GP/), which is optimized for low VRAM and includes numerous video editing options and support for other models (MMaudio, Qwen Image Edit, etc.).
*   **ComfyUI:** Thanks to [kijai](https://github.com/kijai) for providing ComfyUI support ([ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper)).

## Example Videos

### Video-to-Video

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

## Quick Start Guide

### Installation

1.  **Create Conda Environment & Install Dependencies:**

    ```bash
    conda create -n infinitetalk python=3.10
    conda activate infinitetalk
    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
    pip install -U xformers==0.0.28 --index-url https://download.pytorch.org/whl/cu121
    ```

2.  **Flash-Attention Installation:**

    ```bash
    pip install misaki[en]
    pip install ninja
    pip install psutil
    pip install packaging
    pip install wheel
    pip install flash_attn==2.7.4.post1
    ```

3.  **Additional Dependencies:**

    ```bash
    pip install -r requirements.txt
    conda install -c conda-forge librosa
    ```

4.  **FFmpeg Installation:**

    ```bash
    conda install -c conda-forge ffmpeg
    ```
    or
    ```bash
    sudo yum install ffmpeg ffmpeg-devel
    ```

### Model Preparation

1.  **Download Models:**

    | Model                     | Download Link                                                                        | Notes                          |
    | ------------------------- | ------------------------------------------------------------------------------------ | ------------------------------ |
    | Wan2.1-I2V-14B-480P       | [Hugging Face](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P)                  | Base Model                     |
    | chinese-wav2vec2-base     | [Hugging Face](https://huggingface.co/TencentGameMate/chinese-wav2vec2-base)          | Audio Encoder                  |
    | MeiGen-InfiniteTalk       | [Hugging Face](https://huggingface.co/MeiGen-AI/InfiniteTalk)                       | Audio Condition Weights        |

    Download the models using `huggingface-cli`:

    ```bash
    huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./weights/Wan2.1-I2V-14B-480P
    huggingface-cli download TencentGameMate/chinese-wav2vec2-base --local-dir ./weights/chinese-wav2vec2-base
    huggingface-cli download TencentGameMate/chinese-wav2vec2-base model.safetensors --revision refs/pr/1 --local-dir ./weights/chinese-wav2vec2-base
    huggingface-cli download MeiGen-AI/InfiniteTalk --local-dir ./weights/InfiniteTalk
    ```

### Quick Inference

Our model supports both 480P and 720P resolutions.

*   **Lip Synchronization:**  Adjust `Audio CFG` (between 3-5) for better accuracy.
*   **FusionX LoRA:** Faster inference and quality, but may cause color shift over longer durations and reduce ID preservation.
*   **V2V Generation:** Mimics original video's camera movement; SDEdit can improve accuracy, but introduces color shift.
*   **I2V Generation:** Good results from a single image for up to 1 minute; Copying the image to a video using translation or zooming is recommended for longer durations. See the script to [convert image to video](https://github.com/MeiGen-AI/InfiniteTalk/blob/main/tools/convert_img_to_video.py).
*   **Quantization Model:** Use the quantization model to reduce memory usage if inference is killed due to insufficient memory.

#### Inference Examples

1.  **Single GPU Inference:**

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

2.  **720P Inference:**

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

3.  **Low VRAM Inference:**

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

4.  **Multi-GPU Inference:**

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

5.  **Multi-Person Animation:**

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

6.  **Inference with FusioniX or Lightx2v (requires only 4-8 steps):**

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

7.  **Quantization Model Inference (Single GPU only):**

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

8.  **Gradio Demo:**

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

If you find our work useful, please cite:

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

The models in this repository are licensed under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0). You are responsible for your use of the models and are fully accountable for ensuring that your use complies with applicable laws.