<div align="center">
  <img src="assets/logo2.jpg" alt="InfinteTalk" width="440"/>
</div>

# InfiniteTalk: Generate Realistic Talking Videos from Audio

**InfiniteTalk empowers users to create high-quality, talking-head videos from audio input, offering a novel approach to video dubbing and generation.**  This repository provides the code, models, and instructions for generating videos where the subject's lip movements, head pose, and facial expressions are synchronized with the provided audio.  [Visit the original repository on GitHub](https://github.com/MeiGen-AI/InfiniteTalk) to learn more and contribute.

**Key Features:**

*   💬 **Sparse-Frame Video Dubbing:** Accurately synchronizes lips, head movements, body posture, and facial expressions with the audio input.
*   ⏱️ **Infinite-Length Generation:**  Supports generating videos of unlimited duration.
*   ✨ **Enhanced Stability:**  Reduces visual distortions, providing more stable and natural-looking results compared to existing methods.
*   🚀 **Superior Lip Synchronization:** Achieves improved lip-sync accuracy.
*   🖼️ **Image-to-Video Generation:** Generate talking videos from a static image and audio input.

## Quick Start

### 🛠️ Installation

1.  **Create a Conda environment and install dependencies:**

    ```bash
    conda create -n multitalk python=3.10
    conda activate multitalk
    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
    pip install -U xformers==0.0.28 --index-url https://download.pytorch.org/whl/cu121
    pip install -r requirements.txt
    conda install -c conda-forge librosa
    conda install -c conda-forge ffmpeg
    ```

2.  **Additional Installations:**

    ```bash
    pip install misaki[en]
    pip install ninja
    pip install psutil
    pip install packaging
    pip install wheel
    pip install flash_attn==2.7.4.post1
    ```

### 🧱 Model Preparation

1.  **Download Pre-trained Models:**  Use `huggingface-cli` to download the necessary weights.

    ```bash
    huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./weights/Wan2.1-I2V-14B-480P
    huggingface-cli download TencentGameMate/chinese-wav2vec2-base --local-dir ./weights/chinese-wav2vec2-base
    huggingface-cli download TencentGameMate/chinese-wav2vec2-base model.safetensors --revision refs/pr/1 --local-dir ./weights/chinese-wav2vec2-base
    huggingface-cli download MeiGen-AI/InfiniteTalk --local-dir ./weights/InfiniteTalk
    ```

### 🔑 Quick Inference

1.  **Run the base inference script:**

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

    *   Refer to the original README for details on running with 720P, low VRAM, multi-GPU, and multi-person animation.
    *   The README also details how to use FusioniX and quantization models.

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

## Community Works

*   **Wan2GP:** Integration by [deepbeepmeep](https://github.com/deepbeepmeep) for low VRAM optimization and video editing features.
*   **ComfyUI:** Support by [kijai](https://github.com/kijai).

## To-Do List

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

## 📚 Citation

If you find our work useful in your research, please consider citing:

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

## 📜 License

This project is licensed under the Apache 2.0 License.  You are responsible for your use of the generated content and must comply with the license terms.