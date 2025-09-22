<p align="center" width="100%">
<img src="assets/long-rl-logo.png" alt="Long-RL Logo" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

# Long-RL: Revolutionizing Reinforcement Learning for Long Video Understanding

**Tackle long video reasoning challenges and unlock new possibilities in vision-language models with Long-RL, a cutting-edge framework that leverages reinforcement learning for superior performance!** Explore the original repo [here](https://github.com/NVlabs/Long-RL).

[![Paper](https://img.shields.io/badge/ArXiv-Paper-brown)](https://arxiv.org/abs/2507.07966)
[![Code](https://img.shields.io/badge/GitHub-Long%20RL-blue)](https://github.com/NVlabs/Long-RL)
[![Model](https://img.shields.io/badge/HuggingFace-Model-yellow)](https://huggingface.co/Efficient-Large-Model/LongVILA-R1-7B)
[![Video](https://img.shields.io/badge/YouTube-Video-red)](https://www.youtube.com/watch?v=ykbblK2jiEg)
[![Demo](https://img.shields.io/badge/Gradio-Demo-bron)](https://long-rl.hanlab.ai)

<div align="center">

[![Watch the video](assets/demo_video_first_frame.png)](https://www.youtube.com/watch?v=ykbblK2jiEg)

</div>

**Key Features:**

*   **Breakthrough Performance:** Achieve state-of-the-art results on video understanding benchmarks with the LongVILA-R1-7B model.
*   **Scalable Architecture:** Train on hour-long videos with sequence parallelism and a vLLM-based engine, optimized for efficiency.
*   **Comprehensive Framework:** Supports RL training on various modalities (video, text, audio) and models (VILA, Qwen series, and image/video generation models).
*   **Large-Scale Dataset:** Benefit from the LongVideo-Reason dataset, comprising 104K long video QA pairs with high-quality reasoning annotations.
*   **Flexible Configuration:** Process up to 8,192 video frames per video, with adjustable FPS settings for optimal performance.
*   **Open-Ended Reward Support:** Utilize the open-ended reward setting for training open-ended QAs by setting `--worker.rollout.open_ended_reward=True`.
*   **Cached Video Embeddings:** Accelerate training with support for cached video embeddings by setting `--data.cache_dir` and `--worker.actor.cached_embeds_dir`.
*   **Chunked Gathering:** Improve memory efficiency with chunked gathering by setting `--worker.rollout.num_chunk_seq` in the training script.

**Key Improvements and News:**

*   **NeurIPS 2025 Acceptance:** The Long-RL framework has been accepted by NeurIPS 2025.
*   **Extended Frame Support:** LongVILA-R1-7B now supports up to 8,192 video frames per video.
*   **Gradio Demo Release:** Explore the capabilities of LongVILA-R1-7B with the interactive Gradio demo.
*   **Model Weights Availability:** Access the LongVILA-R1-7B model weights on Hugging Face, achieving 65.1% / 71.1% on VideoMME.
*   **Data Generation Instructions:** Detailed instructions and scripts for generating the LongVideo-Reason dataset are available.
*   **New Feature Releases:** Open-ended reward, cached video embeddings, and chunked gathering are now supported.

**Highlights:**

*   **Single-Node Hour-Level Training:** Enables RL training on hour-long videos (3,600 frames) on a single A100 node (8 GPUs).
*   **Omni-Model RL Support:** Supports RL training on models that take text, video, and audio as input.
*   **Image/Video Generation RL:** Offers RL training capabilities for image and video generation models (e.g., Stable Diffusion, Wan series).

**Sections:**

1.  [News](#news)
2.  [Highlights](#highlights)
3.  [Introduction](#introduction)
4.  [LongVILA-R1 Model Usage](#longvila-r1-model-usage)
5.  [Supported Features](#supported-features)
6.  [Installation](#installation)
7.  [Training](#training)
8.  [LongVideo-Reason](#longvideo-reason)
9.  [Examples](#examples)
10. [How to contribute](#how-to-contribute)
11. [Core Contributors](#core-contributors)
12. [Citation](#citation)
13. [Acknowledgement](#acknowledgement)

**(Remaining content follows as per the original README, properly formatted)**