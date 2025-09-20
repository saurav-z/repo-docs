<p align="center" width="100%">
<img src="assets/long-rl-logo.png" alt="Long-RL Logo" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

# Long-RL: Revolutionizing Reinforcement Learning for Long Video Understanding

Tackle long video reasoning challenges head-on with **Long-RL**, a full-stack framework leveraging reinforcement learning to scale vision-language models, with state-of-the-art results and efficient training capabilities.  ([See the original repo](https://github.com/NVlabs/Long-RL))

**Key Features:**

*   **Enhanced Long Video Reasoning:**  Achieve superior performance on video benchmarks by effectively handling long video sequences.
*   **LongVideo-Reason Dataset:** Benefit from a large-scale dataset of 104K long video QA pairs with high-quality reasoning annotations, covering sports, games, and vlogs.
*   **Two-Stage Training Pipeline:** Leverage a two-stage training process combining chain-of-thought supervised fine-tuning (CoT-SFT) and reinforcement learning (RL) for optimal results.
*   **Multi-modal Reinforcement Sequence Parallelism (MR-SP):**  Train efficiently with a custom training infrastructure designed for long video RL, including sequence parallelism and a vLLM-based engine.
*   **Multi-Modality Support:** Supports RL training on models that accept video, text, and audio inputs (e.g., Qwen series models).
*   **Model Compatibility:** Compatible with various models, including VILA and Qwen series models, as well as image and video generation models.
*   **Open-Ended Reward Support**: Fine-tune on open-ended QA (non-multi-choice) benchmarks.
*   **Cached video embeddings**: Train using cached video embeddings for fast training.
*   **Chunked Gathering**: Enable chunked gathering for resource efficient training.

**Key Performance Highlights:**

*   **LongVILA-R1-7B** achieves impressive accuracy on video benchmarks, e.g., **71.1%** on VideoMME (with subtitles).
*   **Up to 8,192 video frames per video** and adjustable FPS.
*   **MR-SP system accelerates long video RL training by up to 2.1x.**
*   **Hour-level long video RL training on a single node** is supported (3,600 frames - 256k tokens) on a single A100 node (8 GPUs).
*   Support for RL training on image/video generation models, e.g., Stable Diffusion and Wan series models.

## Sections

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
```
Key improvements:

*   **SEO Optimization:** Title and descriptions include relevant keywords like "Reinforcement Learning", "Long Video Understanding", and model names.
*   **Concise and Engaging Hook:** A one-sentence summary to draw readers in.
*   **Clear Headings and Structure:**  Uses Markdown headings for readability and organization.
*   **Bulleted Key Features:** Highlights essential capabilities for quick understanding.
*   **Direct Link Back to Original Repo:**  Placed at the beginning.
*   **Contextual Summary:** Summarizes the main points of the original README, focusing on what's important to users.
*   **Emphasis on Performance:** Highlights the model's performance gains with key metrics.
*   **Improved Flow:** Reorganized the information for better readability.
*   **More Active Voice:** Used action-oriented language.
*   **Clear Calls to Action:** Encourages the reader to explore the project.