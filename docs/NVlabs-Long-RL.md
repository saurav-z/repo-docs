<p align="center" width="100%">
<img src="assets/long-rl-logo.png" alt="Long-RL Logo" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

# Long-RL: Revolutionizing Long Video Reasoning with Reinforcement Learning

**Tackle the challenges of long video reasoning with Long-RL, a cutting-edge framework that scales vision-language models using reinforcement learning!**

[![Paper](https://img.shields.io/badge/ArXiv-Paper-brown)](https://arxiv.org/abs/2507.07966)
[![Code](https://img.shields.io/badge/GitHub-Long%20RL-blue)](https://github.com/NVlabs/Long-RL)
[![Model](https://img.shields.io/badge/HuggingFace-Model-yellow)](https://huggingface.co/Efficient-Large-Model/LongVILA-R1-7B)
[![Video](https://img.shields.io/badge/YouTube-Video-red)](https://www.youtube.com/watch?v=ykbblK2jiEg)
[![Demo](https://img.shields.io/badge/Gradio-Demo-bron)](https://long-rl.hanlab.ai)

<div align="center">

[![Watch the video](assets/demo_video_first_frame.png)](https://www.youtube.com/watch?v=ykbblK2jiEg)

</div>

**[Explore the Long-RL Framework on GitHub](https://github.com/NVlabs/Long-RL)**

## Key Features

*   **Scalable Long Video Reasoning:** Train on hour-long videos (3,600 frames, 256k tokens) on a single A100 node (8 GPUs) using sequence parallelism.
*   **Multi-Modal Support:** Train with text, video, and audio inputs using Omni-model RL.
*   **Flexible Training:** Support reinforcement learning on image/video generation models, including Stable Diffusion and Wan series.
*   **High-Performance Models:** Utilize LongVILA-R1-7B, achieving state-of-the-art results on video benchmarks.
*   **Efficient Training Infrastructure:** Leverage Multi-modal Reinforcement Sequence Parallelism (MR-SP) for up to 2.1x speedup on long video RL training.
*   **Large-Scale Dataset:** Benefit from the LongVideo-Reason dataset, comprising 104K long video QA pairs with high-quality reasoning annotations across diverse domains.
*   **Open-Ended Reward Support:** Train for open-ended QAs using an OpenAI API.
*   **Cached Video Embeddings:** Accelerate training with support for cached video embeddings.
*   **Chunked Gathering Support:** Optimize memory usage for training with large batches.

## Introduction

Long-RL is a comprehensive framework designed to scale vision-language models (VLMs) for reasoning with long videos.  This is achieved through a full-stack approach incorporating a large-scale dataset (LongVideo-Reason), a two-stage training pipeline (CoT-SFT + RL), and a specialized training infrastructure (MR-SP). Long-RL unlocks the potential of reinforcement learning to handle complex, long-form video content.

### Supported Models
*   VILA series models
*   Qwen-VL series models
*   Image and video diffusion model RL

### Supported Algorithms
*   GRPO
*   DAPO
*   Reinforce

## LongVILA-R1 Model Usage

Detailed usage instructions, including code examples for general inference and integration with the vLLM engine, can be found in the original README.

## Installation

```bash
git clone https://github.com/NVlabs/Long-RL.git
cd Long-RL
pip install -e .
```

If you plan to train with Qwen-Omni models:
```bash
bash vllm_replace.sh
```

## Training

Training scripts and instructions are available in the `examples` directory. Detailed instructions on single and multi-node training are provided.

## LongVideo-Reason

Information on the LongVideo-Reason dataset, including data generation and evaluation, is available in the [`longvideo-reason`](longvideo-reason/) directory.

## Examples

Explore the capabilities of Long-RL with example videos showcasing the framework's ability to understand and answer questions about various video types.

## How to Contribute

We welcome all contributions. Please see the original README for contribution instructions.

## Core Contributors

[Yukang Chen](https://yukangchen.com/), [Wei Huang](https://aaron-weihuang.com/), [Shuai Yang](https://andysonys.github.io), [Qinghao Hu](https://tonyhao.xyz/), [Baifeng Shi](https://bfshi.github.io/), [Hanrong Ye](https://sites.google.com/site/yhrspace/home), [Ligeng Zhu](https://lzhu.me/).

## Citation

Please cite our work if it helps in your research. See the original README for citation details.

## Acknowledgement

We acknowledge the contributions of EasyR1, verl, vllm, and Flow-GRPO, whose work greatly influenced this project.