<p align="center" width="100%">
<img src="assets/long-rl-logo.png" alt="Long-RL Logo" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

# Long-RL: Revolutionizing Long Video Reasoning with Reinforcement Learning

**Long-RL introduces a full-stack framework that leverages reinforcement learning to scale vision-language models (VLMs) to long videos, achieving state-of-the-art performance on video benchmarks and supporting hour-long video processing.** Explore the original repository: [Long-RL on GitHub](https://github.com/NVlabs/Long-RL).

[![Paper](https://img.shields.io/badge/ArXiv-Paper-brown)](https://arxiv.org/abs/2507.07966)
[![Code](https://img.shields.io/badge/GitHub-Long%20RL-blue)](https://github.com/NVlabs/Long-RL)
[![Model](https://img.shields.io/badge/HuggingFace-Model-yellow)](https://huggingface.co/Efficient-Large-Model/LongVILA-R1-7B)
[![Video](https://img.shields.io/badge/YouTube-Video-red)](https://www.youtube.com/watch?v=ykbblK2jiEg)
[![Demo](https://img.shields.io/badge/Gradio-Demo-bron)](https://long-rl.hanlab.ai)

<div align="center">
  <a href="https://www.youtube.com/watch?v=ykbblK2jiEg">
    <img src="assets/demo_video_first_frame.png" alt="Demo Video" style="width: 100%; min-width: 300px; display: block; margin: auto;">
  </a>
</div>

## Key Features

*   **Long Video Processing:** Supports processing up to **8,192 video frames** per video with configurable FPS settings.
*   **High-Performance Model:**  LongVILA-R1-7B achieves **65.1% / 71.1%** on VideoMME (without/with subtitles) and outperforms previous models on multiple benchmarks.
*   **Multi-Modal Support:** Enables RL training on models with video, text, and audio inputs.
*   **Flexible Training:** Supports RL training on image/video generation models (e.g., Stable Diffusion, Wan series).
*   **Efficient Training Infrastructure:**  Features Multi-modal Reinforcement Sequence Parallelism (MR-SP) for faster long video RL training (up to 2.1x speedup).
*   **Open-ended Reward Support**: Supports training on open-ended QA tasks.
*   **Cached Video Embeddings:** Supports faster video encoding for large batch and long video frames during training.
*   **Chunked Gathering:** Supports chunked gathering to avoid CPU OOM problems.

## Table of Contents

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

## News

*   **[2025.7.30]** **LongVILA-R1-7B** now supports processing up to **8,192** video frames per video, with configurable FPS settings.
*   **[2025.7.24]** Gradio demo released: [https://long-rl.hanlab.ai](https://long-rl.hanlab.ai).
*   **[2025.7.24]** Model weights of **LongVILA-R1-7B** released on HuggingFace: [https://huggingface.co/Efficient-Large-Model/LongVILA-R1-7B](https://huggingface.co/Efficient-Large-Model/LongVILA-R1-7B).
*   **[2025.7.19]** Detailed instructions and scripts for the LongVideo-Reason dataset generation process released in the [`longvideo-reason`](longvideo-reason/) directory.
*   **[2025.7.18]** New supported features: *Open-ended reward*, *Cached video embeddings*, and *Chunked gathering*.
*   **[2025.7.10]** Paper released: [https://arxiv.org/abs/2507.07966](https://arxiv.org/abs/2507.07966) and GitHub repository [Long-RL](https://github.com/NVlabs/Long-RL).

## Highlights

1.  **Efficient Training:** Hour-level long video RL training on a single A100 node (8 GPUs). See `examples/new_supports/qwen2_5_vl_3b_video_1h.sh`.
2.  **Omni-Model Support:** RL training with text, video, and audio inputs. See `examples/new_supports/qwen2_5_omni_3b_grpo.sh`.
3.  **Image/Video Generation RL:** RL training on image/video generation models. See `examples/new_supports/sd3_image_grpo.sh` and `examples/new_supports/wan_video_grpo.sh`.

## Introduction

Long-RL is a full-stack framework for scaling vision-language models to handle long videos. It incorporates: (1) a large-scale dataset, LongVideo-Reason, (2) a two-stage training pipeline combining Chain-of-Thought supervised fine-tuning (CoT-SFT) and reinforcement learning (RL), and (3) a training infrastructure optimized for long video RL, named Multi-modal Reinforcement Sequence Parallelism (MR-SP).

**Supported Models:**

*   VILA series models (image/video) with SP support
*   Qwen-VL series models (text/image/video/audio) with SP support
*   Image and video diffusion model RL

**Supported Algorithms:**

*   GRPO, DAPO, and Reinforce with SP support

## LongVILA-R1 Model Usage

```python
from transformers import AutoModel

model_path = "Efficient-Large-Model/LongVILA-R1-7B"
model = AutoModel.from_pretrained(model_path, trust_remote_code=True, device_map="auto")

# You can adjust the FPS value as needed. 
# To disable FPS control, set it to 0 and manually specify the number of processed video frames via `num_video_frames`.
# Example:
# model.config.fps = 8.0
# model.config.num_video_frames, model.config.fps = 512, 0


use_thinking = True # Switching between thinking and non-thinking modes
system_prompt_thinking = "You are a helpful assistant. The user asks a question, and then you solves it.\n\nPlease first think deeply about the question based on the given video, and then provide the final answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\n\n Question: {question}"

prompt = "What is the main purpose of the video?"
video_path = "video.mp4"

if use_thinking:
  prompt = system_prompt_thinking.format(question=prompt)

response = model.generate_content([prompt, {"path": video_path}])
print("Response: ", response)
```

### with vLLM engine
Tested on `vllm==0.9.1`. We need to get the remote code first.
```bash
mkdir remote_code
cp path_to/Efficient-Large-Model/LongVILA-R1-7B/*.py remote_code
```
Then, you can use the following code for model generation.
```python
import os
from transformers import AutoModel
from vllm import LLM, SamplingParams
from remote_code.media import extract_media
from remote_code.mm_utils import process_images
from remote_code.tokenizer_utils import tokenize_conversation

model_path = "path_to/Efficient-Large-Model/LongVILA-R1-7B"

model_encoder = AutoModel.from_pretrained(model_path, trust_remote_code=True, device_map="auto", llm_only_need_embed=True)
# you can change gpu_memory_utilization according to GPU memory
llm = LLM(model=os.path.join(model_path, "llm"), enable_prompt_embeds=True, gpu_memory_utilization=0.5)

use_thinking = True # Switching between thinking and non-thinking modes
system_prompt_thinking = "You are a helpful assistant. The user asks a question, and then you solves it.\n\nPlease first think deeply about the question based on the given video, and then provide the final answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\n\n Question: {question}"

prompt = "What is the main purpose of the video?"
video_path = "video.mp4"

if use_thinking:
  prompt = system_prompt_thinking.format(question=prompt)

conversation = [{"from": "human", "value": [prompt, {"path": video_path}]}]
media = extract_media(conversation, model_encoder.config)
input_ids = tokenize_conversation(conversation, model_encoder.tokenizer, add_generation_prompt=True).unsqueeze(0).cuda()
media["video"] = [
    process_images(images, model_encoder.vision_tower.image_processor, model_encoder.config).half()
    for images in media["video"]
]

inputs_embeds, _, _ = model_encoder._embed(input_ids, media, {"video": {}}, None, None)

completions = llm.generate(prompts=[{"prompt_embeds": inputs_embeds.squeeze(0)}], sampling_params=SamplingParams(max_tokens=1024))
response = completions[0].outputs[0].text
print("Response: ", response)
```

## Supported Features

*   **Open-ended reward:** Training support for open-ended QAs.
*   **Cached video embeddings:** Supports using cached video embeddings.
*   **Chunked gathering:** Support for chunked gathering for `all_gather_data_proto`.

## Installation

```bash
git clone https://github.com/NVlabs/Long-RL.git
cd Long-RL
pip install -e .
```
If you want to train Qwen-Omni models, please
```bash
bash vllm_replace.sh
```

## Training

### Single Node

Refer to training scripts in the `examples` directory. For example:

```bash
bash examples/new_supports/qwen2_5_vl_3b_video_grpo.sh $VIDEO_PATH
```

### Multi-Nodes

Follow the instructions from the EasyR1 repo, [here](https://github.com/hiyouga/EasyR1/tree/main?tab=readme-ov-file#how-to-run-70b-model-in-multi-node-environment).

Example `sbatch` script usage:

```bash
bash scripts/srun_multi_nodes.sh $TRAIN_SCRIPT $NNODES
```

### Merge Checkpoint in Hugging Face Format

```bash
python3 scripts/model_merger.py --local_dir checkpoints/easy_r1/exp_name/global_step_1/actor
```

## LongVideo-Reason

Detailed instructions for data generation and model evaluation on the `LongVideo-Reason` benchmark are available in the [`longvideo-reason`](longvideo-reason/) directory.

## Examples

\[Image examples with links to Google Drive]

## How to Contribute

*   Install git.
*   Create a fork.
*   Clone your fork locally.
*   Read the installation sections.
*   Commit and push your changes.
*   Submit a pull request.

## Core Contributors

[Yukang Chen](https://yukangchen.com/), [Wei Huang](https://aaron-weihuang.com/), [Shuai Yang](https://andysonys.github.io), [Qinghao Hu](https://tonyhao.xyz/), [Baifeng Shi](https://bfshi.github.io/), [Hanrong Ye](https://sites.google.com/site/yhrspace/home), [Ligeng Zhu](https://lzhu.me/).

## Citation

```bibtex
@misc{long-rl,
  title = {Long-RL: Scaling RL to Long Sequences},
  author = {Yukang Chen, Wei Huang, Shuai Yang, Qinghao Hu, Baifeng Shi, Hanrong Ye, Ligeng Zhu, Zhijian Liu, Pavlo Molchanov, Jan Kautz, Xiaojuan Qi, Sifei Liu,Hongxu Yin, Yao Lu, Song Han},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/NVlabs/Long-RL}},
}
```

```bibtex
@article{chen2025longvila-r1,
      title={Scaling RL to Long Videos},
      author={Yukang Chen and Wei Huang and Baifeng Shi and Qinghao Hu and Hanrong Ye and Ligeng Zhu and Zhijian Liu and Pavlo Molchanov and Jan Kautz and Xiaojuan Qi and Sifei Liu and Hongxu Yin and Yao Lu and Song Han},
      year={2025},
      eprint={2507.07966},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

```bibtex
@inproceedings{chen2024longvila,
      title={LongVILA: Scaling Long-Context Visual Language Models for Long Videos},
      author={Yukang Chen and Fuzhao Xue and Dacheng Li and Qinghao Hu and Ligeng Zhu and Xiuyu Li and Yunhao Fang and Haotian Tang and Shang Yang and Zhijian Liu and Ethan He and Hongxu Yin and Pavlo Molchanov and Jan Kautz and Linxi Fan and Yuke Zhu and Yao Lu and Song Han},
      booktitle={The International Conference on Learning Representations (ICLR)},
      year={2025},
}
```

## Acknowledgement

*   [EasyR1](https://github.com/hiyouga/EasyR1): The codebase we built upon.
*   [verl](https://github.com/volcengine/verl): The RL training framework.
*   [vllm](https://github.com/vllm-project/vllm): Used for the rollout engine.
*   [Flow-GRPO](https://github.com/yifan123/flow_grpo): Referenced for image/video generation RL.