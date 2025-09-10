<p align="center" width="100%">
<img src="assets/long-rl-logo.png" alt="Long-RL Logo" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

# Long-RL: Revolutionizing Long Video Reasoning with Reinforcement Learning

**Unlock the power of long video understanding with Long-RL, a full-stack framework that leverages reinforcement learning to scale vision-language models for in-depth video analysis.**  [[Paper](https://arxiv.org/abs/2507.07966)] | [[GitHub Repo](https://github.com/NVlabs/Long-RL)] | [[Model](https://huggingface.co/Efficient-Large-Model/LongVILA-R1-7B)] | [[Video Demo](https://www.youtube.com/watch?v=ykbblK2jiEg)] | [[Gradio Demo](https://long-rl.hanlab.ai)]

**Key Features:**

*   **Scalable Long Video Processing:**  Process up to 8,192 video frames, enabling detailed analysis of extended video content.
*   **Advanced Reasoning:**  Achieve state-of-the-art performance on video benchmarks with LongVILA-R1-7B, including 65.1% / 71.1% accuracy on VideoMME (without/with subtitles).
*   **Multi-Modal Support:**  Train on diverse modalities including video, text, and audio, and support various models such as VILA and Qwen series.
*   **Efficient Training:**  Utilize Multi-modal Reinforcement Sequence Parallelism (MR-SP) for optimized long video RL training, achieving up to 2.1x speedup.
*   **Comprehensive Dataset:**  Leverage LongVideo-Reason, a large-scale dataset (104K video QA pairs) with high-quality reasoning annotations.
*   **Flexible Deployment:**  Use with vLLM engine.

**[Watch the Video](https://www.youtube.com/watch?v=ykbblK2jiEg)**

## Key Improvements & Updates

*   **[July 30, 2025]** LongVILA-R1-7B supports up to 8,192 video frames per video, with configurable FPS settings.
*   **[July 24, 2025]** Gradio demo released.
*   **[July 24, 2025]** LongVILA-R1-7B model weights released on Hugging Face.
*   **[July 19, 2025]** Detailed data generation instructions for the LongVideo-Reason dataset released.
*   **[July 18, 2025]** New Features: Open-ended reward, Cached video embeddings, and Chunked gathering.
*   **[July 10, 2025]** Paper and GitHub repository released.

## Model Performance

| Models             | VideoMME (w/o sub) | VideoMME (w sub) | ActivityNet-QA (test) | LongVideoBench (val) | PerceptionTest (val) | NExT-QA (mc) | VNBench (val) |
|:-------------------|:------------------:|:----------------:|:---------------------:|:--------------------:|:--------------------:|:--------:|:-------------:|
| **LongVILA-7B**    |      **60.1**      |     **65.1**     |       **59.5**        |       **57.1**       |       **58.1**       | **80.7** |   **63.0**    |
| **LongVILA-R1-7B** |      **65.1**      |     **71.1**     |       **64.8**        |       **58.0**       |       **68.9**       | **81.5** |   **75.5**    |

## Quick Links

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
11. [Core Contributors](#core-Contributors)
12. [Citation](#citation)
13. [Acknowledgement](#acknowledgement)

## Get Started

*   [Installation](#installation)
*   [Training](#training)
*   [Model Usage](#longvila-r1-model-usage)
*   [Examples](#examples)

**For more details and the latest updates, please visit the [Long-RL GitHub repository](https://github.com/NVlabs/Long-RL).**