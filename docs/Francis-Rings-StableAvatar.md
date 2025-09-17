# StableAvatar: Generate Infinite-Length Audio-Driven Avatar Videos

**Create stunning, endless avatar videos directly from audio with StableAvatar!** ([Original Repo](https://github.com/Francis-Rings/StableAvatar))

[![Project Page](https://img.shields.io/badge/Project-Page-Green)](https://francis-rings.github.io/StableAvatar)
[![Arxiv](https://img.shields.io/badge/Paper-Arxiv-red)](https://arxiv.org/abs/2508.08248)
[![HuggingFace Model](https://img.shields.io/badge/HuggingFace-Model-orange)](https://huggingface.co/FrancisRing/StableAvatar/tree/main)
[![Hugging Face Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue)](https://huggingface.co/spaces/YinmingHuang/StableAvatar)
[![YouTube](https://img.shields.io/badge/YouTube-Watch-red?style=flat-square&logo=youtube)](https://www.youtube.com/watch?v=6lhvmbzvv3Y)
[![Bilibili](https://img.shields.io/badge/Bilibili-Watch-blue?style=flat-square&logo=bilibili)](https://www.bilibili.com/video/BV1hUt9z4EoQ)

StableAvatar is an innovative approach to generating long-form, high-quality avatar videos driven by audio, eliminating the need for post-processing tools.

**Key Features:**

*   **Infinite-Length Video Generation:** Produce avatar videos of virtually unlimited duration.
*   **Identity Preservation:** Ensures consistent and accurate representation of the avatar throughout the video.
*   **High Fidelity:** Generates videos with impressive visual quality.
*   **End-to-End Solution:** Directly synthesizes videos without requiring face-swapping or restoration tools.
*   **Novel Architecture:** Utilizes a Time-step-aware Audio Adapter and Audio Native Guidance Mechanism.

**Video Examples:**

**(Include the 9 video examples from the original README, properly formatted and captioned. The captions should describe what the video is showing and highlight the key aspects like "ID-preserving", "infinite-length", or "high-fidelity")**

**[Video 1 Description - e.g., Demonstrating ID-preserving and high-fidelity]**

<video src="https://github.com/user-attachments/assets/d7eca208-6a14-46af-b337-fb4d2b66ba8d" width="320" controls loop></video>

**[Video 2 Description - e.g., Showcasing infinite-length generation]**

<video src="https://github.com/user-attachments/assets/b15784b1-c013-4126-a764-10c844341a4e" width="320" controls loop></video>

**[Video 3 Description]**

<video src="https://github.com/user-attachments/assets/87faa5c1-a118-4a03-a071-45f18e87e6a0" width="320" controls loop></video>

**[Video 4 Description]**

<video src="https://github.com/user-attachments/assets/531eb413-8993-4f8f-9804-e3c5ec5794d4" width="320" controls loop></video>

**[Video 5 Description]**

<video src="https://github.com/user-attachments/assets/cdc603e-df46-4cf8-a14e-1575053f996f" width="320" controls loop></video>

**[Video 6 Description]**

<video src="https://github.com/user-attachments/assets/7022dc93-f705-46e5-b8fc-3a3fb755795c" width="320" controls loop></video>

**[Video 7 Description]**

<video src="https://github.com/user-attachments/assets/0ba059eb-ff6f-4d94-80e6-f758c613b737" width="320" controls loop></video>

**[Video 8 Description]**

<video src="https://github.com/user-attachments/assets/03e6c1df-85c6-448d-b40d-aacb8add4e45" width="320" controls loop></video>

**[Video 9 Description]**

<video src="https://github.com/user-attachments/assets/90b78154-dda0-4eaa-91fd-b5485b718a7f" width="320" controls loop></video>

**[Comparison Demo: Highlight Superior Performance]**

<video src="https://github.com/user-attachments/assets/90691318-311e-40b9-9bd9-62db83ab1492" width="768" autoplay loop muted playsinline></video>

**Overview**

**(Include the framework image, properly formatted and with a caption.)**

<p align="center">
  <img src="assets/figures/framework.jpg" alt="model architecture" width="1280"/>
  </br>
  <i>The overview of the framework of StableAvatar.</i>
</p>

StableAvatar overcomes the limitations of existing audio-driven avatar video generation models, offering a solution for creating long, high-quality videos. The framework includes a Time-step-aware Audio Adapter and Audio Native Guidance Mechanism, leading to significant improvements in audio synchronization and identity consistency.

**News**

*   `[2025-9-8]`: 🔥  New demo released! [YouTube](https://www.youtube.com/watch?v=GH4hrxIis3Q) and [Bilibili](https://www.bilibili.com/video/BV1jGYPzqEux).
*   `[2025-8-29]`: 🔥 Public demo is now live on [Hugging Face Spaces](https://huggingface.co/spaces/YinmingHuang/StableAvatar). (Note: Hugging Face Pro users only).
*   `[2025-8-18]`: 🔥 Runs on [ComfyUI](https://github.com/smthemex/ComfyUI_StableAvatar) in 10 steps. Thanks @[smthemex](https://github.com/smthemex).
*   `[2025-8-16]`: 🔥 Finetuning and LoRA codes released!
*   `[2025-8-15]`: 🔥 Runs on Gradio Interface. Thanks @[gluttony-10](https://space.bilibili.com/893892).
*   `[2025-8-15]`: 🔥 Runs on [ComfyUI](https://github.com/smthemex/ComfyUI_StableAvatar). Thanks @[smthemex](https://github.com/smthemex).
*   `[2025-8-13]`: 🔥 Added support for new Blackwell series Nvidia chips.
*   `[2025-8-11]`: 🔥 Project page, code, technical report, and basic model checkpoint released.

**Quickstart Guide**

For the basic model version, generate infinite-length videos at resolutions like 480x832, 832x480 or 512x512.

*   **Environment Setup:**

    ```bash
    pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu124
    pip install -r requirements.txt
    # Optional to install flash_attn to accelerate attention computation
    pip install flash_attn
    ```

    **For Blackwell series:**

    ```bash
    pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
    pip install -r requirements.txt
    # Optional to install flash_attn to accelerate attention computation
    pip install flash_attn
    ```

*   **Download Weights:**
    If you encounter connection issues with Hugging Face, use the mirror endpoint by setting the environment variable: `export HF_ENDPOINT=https://hf-mirror.com`.

    ```bash
    pip install "huggingface_hub[cli]"
    cd StableAvatar
    mkdir checkpoints
    huggingface-cli download FrancisRing/StableAvatar --local-dir ./checkpoints
    ```
    The file structure should be organized as shown in the original README.

*   **Audio Extraction:**

    ```bash
    python audio_extractor.py --video_path="path/test/video.mp4" --saved_audio_path="path/test/audio.wav"
    ```

*   **Vocal Separation (Optional):**

    ```bash
    pip install audio-separator[gpu]
    python vocal_seperator.py --audio_separator_model_file="path/StableAvatar/checkpoints/Kim_Vocal_2.onnx" --audio_file_path="path/test/audio.wav" --saved_vocal_path="path/test/vocal.wav"
    ```

*   **Base Model Inference:**
    Modify `inference.sh` for your desired settings.

    ```bash
    bash inference.sh
    ```
    Use 512x512, 480x832, or 832x480 resolutions. Adjust parameters like `--width`, `--height`, `--output_dir`, `--validation_reference_path`, `--validation_driven_audio_path`, `--validation_prompts`, `--pretrained_model_name_or_path`, `--pretrained_wav2vec_path`, `--transformer_path`, `--sample_steps`, `--overlap_window_length`, `--clip_sample_n_frames`, `--sample_text_guide_scale`, and `--sample_audio_guide_scale`.
    Run Gradio interface:

    ```bash
    python app.py
    ```

    Check the `path/StableAvatar/examples` directory for validation examples.

*   **Tips:**
    *   Use the `--transformer_path` setting to switch between transformer3d models for different resolutions.
    *   Use `--GPU_memory_mode` to manage GPU memory usage.
    *   Run Multi-GPU inference using `--ulysses_degree` and `--ring_degree`.
    *   Use ffmpeg to add audio: `ffmpeg -i video_without_audio.mp4 -i /path/audio.wav -c:v copy -c:a aac -shortest /path/output_with_audio.mp4`

*   **Model Training:**
    Training data should be structured as shown in the README. The structure is: `talking_face_data/`.
    Training commands and file structure is also described in the README.

*   **Model Finetuning**
    Follow the instructions in the README.

*   **VRAM and Runtime**
    The 5s video (480x832, fps=25), the basic model (--GPU_memory_mode="model_full_load") requires approximately 18GB VRAM and finishes in 3 minutes on a 4090 GPU.

**Contact**

Email: francisshuyuan@gmail.com

**Citation**

```bib
@article{tu2025stableavatar,
  title={StableAvatar: Infinite-Length Audio-Driven Avatar Video Generation},
  author={Tu, Shuyuan and Pan, Yueming and Huang, Yinming and Han, Xintong and Xing, Zhen and Dai, Qi and Luo, Chong and Wu, Zuxuan and Jiang Yu-Gang},
  journal={arXiv preprint arXiv:2508.08248},
  year={2025}
}
```