# StableAvatar: Generate Infinite-Length Avatar Videos from Audio

**Create stunning, infinite-length avatar videos driven by audio with StableAvatar, the cutting-edge technology for realistic and seamless video generation. ([Original Repo](https://github.com/Francis-Rings/StableAvatar))**

[![Project Page](https://img.shields.io/badge/Project-Page-Green)](https://francis-rings.github.io/StableAvatar)
[![Arxiv](https://img.shields.io/badge/Paper-Arxiv-red)](https://arxiv.org/abs/2508.08248)
[![HuggingFace Model](https://img.shields.io/badge/HuggingFace-Model-orange)](https://huggingface.co/FrancisRing/StableAvatar/tree/main)
[![YouTube Demo](https://img.shields.io/badge/YouTube-Watch-red?style=flat-square&logo=youtube)](https://www.youtube.com/watch?v=6lhvmbzvv3Y)
[![Bilibili Demo](https://img.shields.io/badge/Bilibili-Watch-blue?style=flat-square&logo=bilibili)](https://www.bilibili.com/video/BV1hUt9z4EoQ)

StableAvatar is an innovative approach to generating high-fidelity, identity-preserving avatar videos directly from audio, capable of producing videos of virtually unlimited length. This breakthrough eliminates the need for face-swapping or restoration tools, providing a streamlined and efficient workflow.

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

**Key Features:**

*   **Infinite-Length Video Generation:** Create avatar videos that can run for extended durations without significant quality degradation.
*   **ID Preservation:** Maintain the identity of the avatar throughout the entire video.
*   **End-to-End Synthesis:** Generate videos directly, bypassing the need for post-processing techniques.
*   **High Fidelity:** Produce videos with superior quality, natural audio synchronization, and consistent identity.

<p style="text-align: justify;">
  <span>StableAvatar synthesizes audio-driven avatar videos, demonstrating the capability to produce <b>infinite-length</b> and <b>ID-preserving videos</b>. The videos are generated <b>directly by StableAvatar without the use of face-related post-processing tools</b>, such as the face-swapping tool FaceFusion or face restoration models like GFP-GAN and CodeFormer.</span>
</p>

<p align="center">
  <video src="https://github.com/user-attachments/assets/90691318-311e-40b9-9bd9-62db83ab1492" width="768" autoplay loop muted playsinline></video>
  <br/>
  <span>Comparison results between StableAvatar and state-of-the-art (SOTA) audio-driven avatar video generation models highlight the superior performance of StableAvatar in delivering <b>infinite-length, high-fidelity, identity-preserving avatar animation</b>.</span>
</p>


## 🔑 Quickstart

The basic model checkpoint (Wan2.1-1.3B-based) allows for generating **infinite-length videos at resolutions of 480x832, 832x480, or 512x512**. To optimize memory usage, adjust the frame count and resolution as needed.

### 🧱 Environment Setup
Detailed environment setup instructions are available in the original repository, including options for various PyTorch versions and hardware configurations (including Blackwell series chips).

```bash
# Example setup (refer to original repo for full details)
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install flash_attn # Optional for accelerated attention
```

### 🧱 Download Weights

Download the necessary model weights using the Hugging Face CLI:

```bash
pip install "huggingface_hub[cli]"
cd StableAvatar
mkdir checkpoints
huggingface-cli download FrancisRing/StableAvatar --local-dir ./checkpoints
```

Ensure the file structure matches the described organization.

### 🧱 Audio Extraction

Extract audio from video files:

```bash
python audio_extractor.py --video_path="path/test/video.mp4" --saved_audio_path="path/test/audio.wav"
```

### 🧱 Vocal Separation (Optional)

Separate vocals from audio for improved lip-sync:

```bash
pip install audio-separator[gpu]
python vocal_seperator.py --audio_separator_model_file="path/StableAvatar/checkpoints/Kim_Vocal_2.onnx" --audio_file_path="path/test/audio.wav" --saved_vocal_path="path/test/vocal.wav"
```

### 🧱 Base Model Inference

Run the inference script using the provided `inference.sh` as a starting point:

```bash
bash inference.sh
```

Customize the script with your desired settings, including resolution, prompts, audio/video paths, and model paths.  Pay special attention to prompt formatting: `[Description of first frame]-[Description of human behavior]-[Description of background (optional)]`.

*   Explore the example videos in `/path/StableAvatar/examples` for inspiration.

#### 💡 Tips

*   Experiment with different StableAvatar weight versions for optimal results.
*   Manage GPU memory usage with various `--GPU_memory_mode` settings in the inference script.
*   Utilize multi-GPU inference for faster processing by modifying `--ulysses_degree` and `--ring_degree`.
*   Combine the output video with audio using `ffmpeg`.

```bash
ffmpeg -i video_without_audio.mp4 -i /path/audio.wav -c:v copy -c:a aac -shortest /path/output_with_audio.mp4
```

### 🧱 Model Training & Fine-tuning
Detailed training and fine-tuning procedures are described in the original repository, including:
-   Dataset organization guidance.
-   Training commands with example parameters.
-   Lora training setup and customization.

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

## Contact

For inquiries and collaboration, reach out to:

Email: francisshuyuan@gmail.com

**If you find this project useful, please consider giving it a star ⭐ and citing the work ❤️:**

```bib
@article{tu2025stableavatar,
  title={StableAvatar: Infinite-Length Audio-Driven Avatar Video Generation},
  author={Tu, Shuyuan and Pan, Yueming and Huang, Yinming and Han, Xintong and Xing, Zhen and Dai, Qi and Luo, Chong and Wu, Zuxuan and Jiang Yu-Gang},
  journal={arXiv preprint arXiv:2508.08248},
  year={2025}
}
```