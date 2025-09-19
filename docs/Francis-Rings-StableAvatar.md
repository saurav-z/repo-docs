# StableAvatar: Generate Infinite-Length Avatar Videos with Audio

**Experience the future of video creation with StableAvatar, an innovative approach to generating unlimited-length, audio-driven avatar videos.**

[Project Page](https://francis-rings.github.io/StableAvatar) | [Arxiv Paper](https://arxiv.org/abs/2508.08248) | [Hugging Face Model](https://huggingface.co/FrancisRing/StableAvatar/tree/main) | [Hugging Face Demo](https://huggingface.co/spaces/YinmingHuang/StableAvatar) | [YouTube Demo](https://www.youtube.com/watch?v=6lhvmbzvv3Y) | [Bilibili Demo](https://www.bilibili.com/video/BV1hUt9z4EoQ)

StableAvatar revolutionizes audio-driven avatar video generation, enabling the synthesis of infinite-length videos that preserve identity and maintain high fidelity.  Developed by Shuyuan Tu and a team of researchers, StableAvatar eliminates the need for post-processing tools like face-swapping or restoration, producing seamless results directly.

[Include a row of example videos (3 per row) like the original README. Keep the controls and loop enabled.]

## Key Features

*   **Infinite-Length Video Generation:** Create avatar videos of any length without compromising quality.
*   **Identity Preservation:**  Ensure consistent identity throughout the entire video.
*   **High-Fidelity Results:**  Achieve superior audio synchronization and natural movements.
*   **End-to-End Synthesis:**  Generate videos directly without the need for face-related post-processing.
*   **Time-step-aware Audio Adapter:** Introduced to prevent latent distribution error accumulation.
*   **Audio Native Guidance Mechanism:** Enhance audio synchronization by leveraging the diffusion’s own evolving joint audio-latent prediction as a dynamic guidance signal.
*   **Dynamic Weighted Sliding-window Strategy:** Fuse latent over time to enhance the smoothness of the infinite-length videos.

## Overview

[Include the model architecture image from the original README]

StableAvatar employs a novel end-to-end video diffusion transformer, trained with tailored modules to overcome limitations in existing models. It utilizes a Time-step-aware Audio Adapter and an Audio Native Guidance Mechanism to ensure accurate audio synchronization and prevent latent distribution errors, resulting in high-quality, continuous avatar videos.

## News & Updates

*   **[2025-9-8]:** 🔥  New demo released on [YouTube](https://www.youtube.com/watch?v=GH4hrxIis3Q) and [Bilibili](https://www.bilibili.com/video/BV1jGYPzqEux).
*   **[2025-8-29]:** 🔥 StableAvatar public demo is now live on [Hugging Face Spaces](https://huggingface.co/spaces/YinmingHuang/StableAvatar). (Note: demo accessible to **Hugging Face Pro** users only.)
*   **[2025-8-18]:** 🔥 StableAvatar runs on [ComfyUI](https://github.com/smthemex/ComfyUI_StableAvatar) in 10 steps.
*   **[2025-8-16]:** 🔥 Finetuning and LoRA codes released.
*   **[2025-8-15]:** 🔥 Gradio Interface and ComfyUI integration.
*   **[2025-8-13]:** 🔥 Added changes to run StableAvatar on the new Blackwell series Nvidia chips.
*   **[2025-8-11]:** 🔥 Project page, code, technical report, and basic model checkpoint released.

## To-Do List

*   [x] StableAvatar-1.3B-basic
*   [x] Inference Code
*   [x] Data Pre-Processing Code (Audio Extraction)
*   [x] Data Pre-Processing Code (Vocal Separation)
*   [x] Training Code
*   [x] Full Finetuning Code
*   [x] Lora Training Code
*   [x] Lora Finetuning Code
*   [ ] Inference Code with Audio Native Guidance
*   [ ] StableAvatar-pro

## Quickstart

Get started with StableAvatar and generate high-quality, infinite-length avatar videos using the following steps.

### Environment Setup

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
# Optional to install flash_attn to accelerate attention computation
pip install flash_attn
```

**For Blackwell series chips:**

```bash
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
# Optional to install flash_attn to accelerate attention computation
pip install flash_attn
```

### Download Weights

**If you encounter connection issues, use the mirror endpoint:** `export HF_ENDPOINT=https://hf-mirror.com`.

```bash
pip install "huggingface_hub[cli]"
cd StableAvatar
mkdir checkpoints
huggingface-cli download FrancisRing/StableAvatar --local-dir ./checkpoints
```

[**Important:**  Ensure the file structure matches the original's for proper operation. Include the file structure from the original.]

### Audio Extraction

Extract audio from your video:

```bash
python audio_extractor.py --video_path="path/test/video.mp4" --saved_audio_path="path/test/audio.wav"
```

### Vocal Separation (Optional)

Separate vocals from audio for better lip sync:

```bash
pip install audio-separator[gpu]
python vocal_seperator.py --audio_separator_model_file="path/StableAvatar/checkpoints/Kim_Vocal_2.onnx" --audio_file_path="path/test/audio.wav" --saved_vocal_path="path/test/vocal.wav"
```

### Base Model Inference

Run the inference script:

```bash
bash inference.sh
```

[**Important:**  Modify the parameters in `inference.sh` as needed, including resolution, paths, and prompts.  Provide detailed instructions on what each parameter does, and recommended ranges.]

You can also launch a Gradio interface with:

```bash
python app.py
```

[Provide an example command. The examples mentioned in the original readme are good.]

**Tip:**  For better results, use the Wan2.1-1.3B-based StableAvatar weights with the `--transformer_path` flag, especially if dealing with limited GPU resources. Also note the guidance on reducing GPU memory usage with  `--GPU_memory_mode` and using multi-GPU inference. For the finished product combine the video and audio using ffmpeg.

```bash
ffmpeg -i video_without_audio.mp4 -i /path/audio.wav -c:v copy -c:a aac -shortest /path/output_with_audio.mp4
```

### Model Training

[Provide a summarized version of the model training section. Focus on the key information and steps, not every single detail.]

[Include a snippet of the dataset folder structure. Also highlight the training scripts and their uses with brief explanations.]

### Model Finetuning

[Summarize the model finetuning process, including the key commands and parameters.]

### VRAM Requirement and Runtime

For a 5s video (480x832, fps=25), the basic model requires approximately 18GB VRAM and finishes in 3 minutes on a 4090 GPU.

**Access the comprehensive documentation and example resources on the [original GitHub repository](https://github.com/Francis-Rings/StableAvatar).**

## Contact

For suggestions or assistance, contact:

Email: francisshuyuan@gmail.com

**If you find this project useful, please consider giving it a star on GitHub and citing our paper:**

[Include BibTex citation from original README]