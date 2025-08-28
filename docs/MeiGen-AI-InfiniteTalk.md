<div align="center">

<p align="center">
  <img src="assets/logo2.jpg" alt="InfiniteTalk" width="440"/>
</p>

<h1>InfiniteTalk: Generate Unlimited Talking Videos from Audio</h1>

[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://meigen-ai.github.io/InfiniteTalk/)
[![Technique Report](https://img.shields.io/badge/Technique-Report-red)](https://arxiv.org/abs/2508.14033)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/MeiGen-AI/InfiniteTalk)

</div>

> **InfiniteTalk: Unleash the power of AI to create stunning, long-form talking videos, effortlessly syncing audio with visuals for unparalleled realism.**

InfiniteTalk is a groundbreaking audio-driven video generation framework, enabling the creation of high-quality talking videos from both existing videos and still images.  It excels in generating lip-synced videos with consistent facial expressions, head movements, and body posture alignment.

[**View the original repository for detailed information and code.**](https://github.com/MeiGen-AI/InfiniteTalk)

## Key Features

*   💬 **Sparse-Frame Video Dubbing:**  Achieves accurate lip synchronization, head movement, body posture, and facial expression alignment with the audio.
*   ⏱️ **Infinite-Length Generation:** Generate videos of virtually any length, overcoming limitations of traditional methods.
*   ✨ **Enhanced Stability:**  Improved stability and reduced hand/body distortions compared to prior models.
*   🚀 **Superior Lip Accuracy:**  Delivers state-of-the-art lip synchronization, surpassing competing solutions.
*   🖼️ **Image-to-Video Generation:** Transform static images into dynamic, talking videos.

##  Latest Updates

*   **August 19, 2025:**  Release of the [Technique Report](https://arxiv.org/abs/2508.14033), model weights, and source code.  Gradio and [ComfyUI](https://github.com/MeiGen-AI/InfiniteTalk/tree/comfyui) support are also available.
*   **August 19, 2025:** Launch of the official [project page](https://meigen-ai.github.io/InfiniteTalk/).

## Community Contributions

*   **Wan2GP:**  Thanks to [deepbeepmeep](https://github.com/deepbeepmeep) for integrating InfiniteTalk into Wan2GP, which is optimized for low VRAM and offers various video editing features and model support.
*   **ComfyUI:** Special thanks to [kijai](https://github.com/kijai) for ComfyUI support.

## Video Demos

### Video-to-video

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


## Quick Start Guide

### Installation

1.  **Create a Conda environment and install PyTorch and XFormers:**

    ```bash
    conda create -n infinitetalk python=3.10
    conda activate infinitetalk
    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
    pip install -U xformers==0.0.28 --index-url https://download.pytorch.org/whl/cu121
    ```

2.  **Install flash-attn:**

    ```bash
    pip install misaki[en]
    pip install ninja
    pip install psutil
    pip install packaging
    pip install wheel
    pip install flash_attn==2.7.4.post1
    ```

3.  **Install other dependencies:**

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

1.  **Download the Necessary Models:**

    ```bash
    huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./weights/Wan2.1-I2V-14B-480P
    huggingface-cli download TencentGameMate/chinese-wav2vec2-base --local-dir ./weights/chinese-wav2vec2-base
    huggingface-cli download TencentGameMate/chinese-wav2vec2-base model.safetensors --revision refs/pr/1 --local-dir ./weights/chinese-wav2vec2-base
    huggingface-cli download MeiGen-AI/InfiniteTalk --local-dir ./weights/InfiniteTalk
    ```

### Quick Inference

*  **Lip Synchronization Accuracy:** Adjust the `audio CFG` to achieve the best lip sync, the optimal range is between 3–5.  Increase it for better accuracy.
*  **FusionX:** This feature enables faster inference and improves image quality.
*  **V2V generation:** Allows you to generate video of unlimited length. The model will mimic the original video camera movements. Using SDEdit improves camera accuracy but can also introduce colour shifts.
*  **I2V generation:** High-quality results for a single image are generated for about 1 minute. When using this method, try creating a video by zooming into the image or panning across it. 
*  **Quantization model:** To reduce memory, use the quantization model.

1.  **Single GPU Inference**

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

2.  **Run with 720P**

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

3.  **Run with very low VRAM**

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

4.  **Multi-GPU inference**

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

5.  **Multi-Person animation**

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

6.  **Run with FusioniX or Lightx2v (requires 4-8 steps)**

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

7.  **Run with the quantization model (Single GPU only)**

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

8.  **Run with Gradio**

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

If you utilize InfiniteTalk in your research, please cite it using the following BibTeX entry:

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

This project is licensed under the Apache 2.0 License. Please refer to the license file for full details.