<p align="center" width="100%">
<img src="assets/long-rl-logo.png" alt="Long-RL Logo" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

# Long-RL: Revolutionizing Long Video Reasoning with Reinforcement Learning

**Long-RL empowers vision-language models to understand and reason about long videos, achieving state-of-the-art results.** [[Paper](https://arxiv.org/abs/2507.07966)] | [[GitHub](https://github.com/NVlabs/Long-RL)] | [[Model](https://huggingface.co/Efficient-Large-Model/LongVILA-R1-7B)] | [[Video](https://www.youtube.com/watch?v=ykbblK2jiEg)] | [[Demo](https://long-rl.hanlab.ai)]

<div align="center">

[![Watch the video](assets/demo_video_first_frame.png)](https://www.youtube.com/watch?v=ykbblK2jiEg)

</div>

Long-RL introduces a novel full-stack framework to scale reasoning in vision-language models (VLMs) for long videos using reinforcement learning (RL).  This project introduces a new framework, including a state-of-the-art model, LongVILA-R1-7B, a new dataset, LongVideo-Reason, and the custom training infrastructure, MR-SP, to provide improvements in video-based RL training.

## Key Features

*   **LongVILA-R1-7B**:  Achieves leading performance on video benchmarks, demonstrating improved accuracy and steady performance gains with longer video inputs.

    *   **Up to 8,192 Frames**: Supports processing of up to 8,192 video frames per video with configurable FPS settings.

*   **LongVideo-Reason Dataset**:  A large-scale dataset with 104K high-quality video QA pairs.
*   **Multi-modal Reinforcement Sequence Parallelism (MR-SP)**:  A custom training infrastructure that speeds up long video RL training by up to 2.1x.
*   **Omni-Model RL**: Support for RL training on models that take video, text, and audio for input.
*   **Open-Ended Reward Training**: Enables training on open-ended QA tasks via OpenAI integration.
*   **Cached Embeddings**: Offers accelerated RL training using cached video embeddings.

## Key Highlights

*   **Hour-Level Video Training:** Train RL models on videos up to an hour long (3,600 frames/256K tokens) on a single A100 node (8 GPUs) using sequence parallelism.
*   **Flexible Model Support**:  Supports RL training for models from the VILA, Qwen, and image/video generation model families.

## Key Updates

*   **[2025.7.30]** **LongVILA-R1-7B** now supports videos up to **8,192** frames.
*   **[2025.7.24]**  Interactive Gradio Demo is available: [https://long-rl.hanlab.ai](https://long-rl.hanlab.ai)
*   **[2025.7.24]** Model weights released on Hugging Face:  [https://huggingface.co/Efficient-Large-Model/LongVILA-R1-7B](https://huggingface.co/Efficient-Large-Model/LongVILA-R1-7B)
*   **[2025.7.19]**  Data generation instructions for LongVideo-Reason are released.
*   **[2025.7.18]**  Open-ended reward, cached embeddings, and chunked gathering features are added.
*   **[2025.7.10]**  Paper and codebase released!

## Quick Start

**Installation:**

```bash
git clone https://github.com/NVlabs/Long-RL.git
cd Long-RL
pip install -e .
```

**Model Usage:**  Refer to the detailed instructions and example code within the README and  [LongVILA-R1 Model Usage](#longvila-r1-model-usage) section.

## Further Information

*   **[Introduction](#introduction)** - Supported models and algorithms.
*   **[LongVILA-R1 Model Usage](#longvila-r1-model-usage)** - Instructions for inference and vLLM integration.
*   **[Supported Features](#supported-features)** - Details on key features.
*   **[Installation](#installation)** -  Detailed installation instructions.
*   **[Training](#training)** -  Training examples for single and multi-node setups.
*   **[LongVideo-Reason](#longvideo-reason)** -  Information on the dataset.
*   **[Examples](#examples)** -  Visual examples of model performance.
*   **[How to contribute](#how-to-contribute)** - Guide for contributing to the project.
*   **[Core Contributors](#core-Contributors)** - List of core contributors.
*   **[Citation](#citation)** -  How to cite the project.
*   **[Acknowledgement](#acknowledgement)** -  Acknowledgements to related projects.

**For more details, please refer to the full documentation in this README and the accompanying research paper.**