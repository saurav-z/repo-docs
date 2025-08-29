# StableAvatar: Generate Infinite-Length Avatar Videos from Audio

**Revolutionize avatar video creation with StableAvatar, an innovative model that synthesizes limitless, high-quality videos directly from audio, without post-processing.** ([Original Repo](https://github.com/Francis-Rings/StableAvatar))

<p align="center">
  <a href='https://francis-rings.github.io/StableAvatar'><img src='https://img.shields.io/badge/Project-Page-Green'></a> 
  <a href='https://arxiv.org/abs/2508.08248'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a> 
  <a href='https://huggingface.co/FrancisRing/StableAvatar/tree/main'><img src='https://img.shields.io/badge/HuggingFace-Model-orange'></a> 
  <a href='https://www.youtube.com/watch?v=6lhvmbzvv3Y'><img src='https://img.shields.io/badge/YouTube-Watch-red?style=flat-square&logo=youtube'></a> 
  <a href='https://www.bilibili.com/video/BV1hUt9z4EoQ'><img src='https://img.shields.io/badge/Bilibili-Watch-blue?style=flat-square&logo=bilibili'></a>
</p>

## Key Features

*   **Infinite-Length Video Generation:**  Create videos of any length without quality degradation.
*   **Direct Synthesis:**  Generates videos directly, eliminating the need for face-swapping or restoration tools.
*   **Identity Preservation:**  Maintains the identity of the avatar throughout the video.
*   **High Fidelity:** Produces high-quality, realistic avatar animations.
*   **Audio Synchronization:**  Provides superior audio-visual synchronization.
*   **End-to-End Solution:** End-to-end video diffusion transformer.

## Showcase

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

<p style="text-align: justify;">
  <span>StableAvatar generates audio-driven avatar videos, demonstrating its ability to synthesize <b>infinite-length</b> and <b>ID-preserving videos</b>. These videos are generated <b>without any face-related post-processing tools</b>, such as the face-swapping tool FaceFusion or face restoration models like GFP-GAN and CodeFormer.</span>
</p>

<p align="center">
  <video src="https://github.com/user-attachments/assets/90691318-311e-40b9-9bd9-62db83ab1492" width="768" autoplay loop muted playsinline></video>
  <br/>
  <span>Comparison results between StableAvatar and state-of-the-art (SOTA) audio-driven avatar video generation models highlight the superior performance of StableAvatar in delivering <b>infinite-length, high-fidelity, identity-preserving avatar animation</b>.</span>
</p>

## Overview

<p align="center">
  <img src="assets/figures/framework.jpg" alt="model architecture" width="1280"/>
  </br>
  <i>The overview of the framework of StableAvatar.</i>
</p>

StableAvatar, a video diffusion transformer, addresses the limitations of existing models by synthesizing high-quality, infinite-length videos without post-processing. Key innovations include a Time-step-aware Audio Adapter to prevent error accumulation and an Audio Native Guidance Mechanism for improved audio synchronization. A Dynamic Weighted Sliding-window Strategy is introduced for smoother infinite-length videos. Experiments demonstrate StableAvatar's superior performance qualitatively and quantitatively.

## News & Updates

*   **[2025-8-18]:** 🔥 StableAvatar now runs on [ComfyUI](https://github.com/smthemex/ComfyUI_StableAvatar), making it 3x faster! Thanks @[smthemex](https://github.com/smthemex)
*   **[2025-8-16]:** 🔥 Finetuning and LoRA training/finetuning codes are released!
*   **[2025-8-15]:** 🔥 StableAvatar runs on Gradio Interface. Thanks @[gluttony-10](https://space.bilibili.com/893892)!
*   **[2025-8-15]:** 🔥 StableAvatar runs on [ComfyUI](https://github.com/smthemex/ComfyUI_StableAvatar). Thanks @[smthemex](https://github.com/smthemex)
*   **[2025-8-13]:** 🔥 Updates for new Blackwell series Nvidia chips, including RTX 6000 Pro.
*   **[2025-8-11]:** 🔥 Project page, code, technical report, and a basic model checkpoint ([HuggingFace](https://huggingface.co/FrancisRing/StableAvatar/tree/main)) released!

## To-Do List

-   [x] StableAvatar-1.3B-basic
-   [x] Inference Code
-   [x] Data Pre-Processing Code (Audio Extraction)
-   [x] Data Pre-Processing Code (Vocal Separation)
-   [x] Training Code
-   [x] Full Finetuning Code
-   [x] Lora Training Code
-   [x] Lora Finetuning Code
-   [ ] Inference Code with Audio Native Guidance
-   [ ] StableAvatar-pro

## Quickstart

The basic model checkpoint (Wan2.1-1.3B-based) supports generating infinite-length videos at resolutions of 480x832, 832x480, or 512x512. Reduce the number of animated frames or resolution if you encounter memory issues.

### Environment Setup

**PyTorch Installation:**

*   **For CUDA 12.4:**
    ```bash
    pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu124
    pip install -r requirements.txt
    # Optional to install flash_attn to accelerate attention computation
    pip install flash_attn
    ```
*   **For CUDA 12.8 (Blackwell):**
    ```bash
    pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
    pip install -r requirements.txt
    # Optional to install flash_attn to accelerate attention computation
    pip install flash_attn
    ```

### Download Weights

Download model weights using Hugging Face CLI:

```bash
pip install "huggingface_hub[cli]"
cd StableAvatar
mkdir checkpoints
huggingface-cli download FrancisRing/StableAvatar --local-dir ./checkpoints
```

**Organize files:**

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

### Audio Extraction

Extract audio from a video file (.mp4):

```bash
python audio_extractor.py --video_path="path/test/video.mp4" --saved_audio_path="path/test/audio.wav"
```

### Vocal Separation (Optional)

Separate vocal from audio (.wav) for improved lip-sync:

```bash
pip install audio-separator[gpu]
python vocal_seperator.py --audio_separator_model_file="path/StableAvatar/checkpoints/Kim_Vocal_2.onnx" --audio_file_path="path/test/audio.wav" --saved_vocal_path="path/test/vocal.wav"
```

### Base Model Inference

Run `inference.sh` to generate videos.  Modify the following parameters:

*   `--width` and `--height`: Set output resolution (e.g., 512x512, 480x832, 832x480).
*   `--output_dir`:  Set the output directory.
*   `--validation_reference_path`, `--validation_driven_audio_path`, `--validation_prompts`: Provide the paths to the reference image, audio, and text prompts, respectively.
    Prompts should follow this format: `[Description of first frame]-[Description of human behavior]-[Description of background (optional)]`.
*   `--pretrained_model_name_or_path`, `--pretrained_wav2vec_path`, `--transformer_path`: Set paths to pretrained models.
*   `--sample_steps`: Set inference steps (recommended range: 30-50).
*   `--overlap_window_length`: Set overlapping context length (recommended range: 5-15).
*   `--clip_sample_n_frames`: Set the number of synthesized frames per batch.
*   `--sample_text_guide_scale` and `--sample_audio_guide_scale`: Set CFG scales (recommended range: 3-6). Increase audio CFG for better lip-sync.

You can also launch a Gradio interface:
```bash
python app.py
```

**Example files:**

*   Example configurations are in `path/StableAvatar/examples`.

#### Tips:
*   `transformer3d-square.pt` and `transformer3d-rec-vec.pt`: Two model weights, each supporting different resolutions.  Modify `--transformer_path` to switch.
*   `--GPU_memory_mode`:  Optimize GPU memory usage (`model_full_load`, `sequential_cpu_offload`, `model_cpu_offload_and_qfloat8`, or `model_cpu_offload`).
*   Multi-GPU inference: Use `--ulysses_degree` and `--ring_degree` in `inference.sh` (ensure the product equals the total GPU number/world-size).  Add `--fsdp_dit` for FSDP in DiT.

Run multi GPU:
```bash
bash multiple_gpu_inference.sh
```

**Audio with Video:**
Use ffmpeg to combine video and audio:

```bash
ffmpeg -i video_without_audio.mp4 -i /path/audio.wav -c:v copy -c:a aac -shortest /path/output_with_audio.mp4
```

### Model Training

*   **Dataset Organization:** Ensure your training dataset is structured as specified in the documentation.
*   **Training Commands:** Training scripts: `train_1B_square.sh`, `train_1B_square_64.sh`, `train_1B_rec_vec.sh`, and `train_1B_rec_vec_64.sh`.
*   Parameters: `CUDA_VISIBLE_DEVICES`, `--pretrained_model_name_or_path`, `--pretrained_wav2vec_path`, `--output_dir`, `--train_data_*_dir`, `--validation_reference_path`, `--validation_driven_audio_path`, `--video_sample_n_frames`, `--num_train_epochs`.
*   The backgrounds of the selected training videos should remain static, as this helps the diffusion model calculate accurate reconstruction loss.
*   The audio should be clear and free from excessive background noise.

*Training Wan2.1-14B-based StableAvatar*
```bash
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./checkpoints/Wan2.1-I2V-14B-480P
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-720P --local-dir ./checkpoints/Wan2.1-I2V-14B-720P # Optional
bash train_14B.sh
```
*Training Wan2.1-1.3B-based StableAvatar with LoRA*
```bash
bash train_1B_rec_vec_lora.sh
```

### Model Finetuning and Lora Finetuning
```bash
# Fully finetuning
bash train_1B_rec_vec.sh

# LoRA finetuning
bash train_1B_rec_vec_lora.sh
```
### VRAM and Runtime

*   The basic model (480x832, fps=25) requires ~18GB VRAM and takes ~3 minutes on a 4090 GPU for a 5-second video.
*   Consider running the VAE decoder on CPU if you need to generate very long videos (10k+ frames).

## Contact

For questions, suggestions, or if you find our work helpful:

*   Email: francisshuyuan@gmail.com

**If you find our work useful, please consider giving a star ⭐ to this github repository and citing it ❤️:**

```bib
@article{tu2025stableavatar,
  title={StableAvatar: Infinite-Length Audio-Driven Avatar Video Generation},
  author={Tu, Shuyuan and Pan, Yueming and Huang, Yinming and Han, Xintong and Xing, Zhen and Dai, Qi and Luo, Chong and Wu, Zuxuan and Jiang Yu-Gang},
  journal={arXiv preprint arXiv:2508.08248},
  year={2025}
}