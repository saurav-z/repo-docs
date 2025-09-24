<p align="center" width="100%">
<img src="assets/long-rl-logo.png" alt="Long-RL Logo" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

# Long-RL: Revolutionizing Long Video Reasoning with Reinforcement Learning

**Tackle long video reasoning challenges head-on with Long-RL, a cutting-edge framework that scales Reinforcement Learning (RL) to long video sequences.**  [See the original repo](https://github.com/NVlabs/Long-RL)

[![Paper](https://img.shields.io/badge/ArXiv-Paper-brown)](https://arxiv.org/abs/2507.07966)
[![Code](https://img.shields.io/badge/GitHub-Long%20RL-blue)](https://github.com/NVlabs/Long-RL)
[![Model](https://img.shields.io/badge/HuggingFace-Model-yellow)](https://huggingface.co/Efficient-Large-Model/LongVILA-R1-7B)
[![Video](https://img.shields.io/badge/YouTube-Video-red)](https://www.youtube.com/watch?v=ykbblK2jiEg)
[![Demo](https://img.shields.io/badge/Gradio-Demo-bron)](https://long-rl.hanlab.ai)

<div align="center">

[![Watch the video](assets/demo_video_first_frame.png)](https://www.youtube.com/watch?v=ykbblK2jiEg)

</div>

## Key Features

*   **Breakthrough Performance:** Achieve state-of-the-art results on video benchmarks, including up to **71.1% accuracy on VideoMME** with subtitles using LongVILA-R1-7B.
*   **Long Video Support:** Process up to **8,192 video frames** per video with configurable FPS settings.
*   **Efficient Training:** Experience up to **2.1x speedup** in long video RL training with the Multi-modal Reinforcement Sequence Parallelism (MR-SP) system.
*   **Multi-Modal Support:** Train on various modalities including video, text, and audio.
*   **Flexible Model Support:** Train with VILA and Qwen series models.
*   **Versatile Algorithm Support:** Supports GRPO, DAPO & Reinforce algorithms with SP support.
*   **Image/Video Generation RL Support:** Includes support for image/video generation models, like Stable Diffusion and Wan series.
*   **Publicly Available Training System:** Leverage a readily available training system for RL on long videos, including hour-long videos (e.g., 3,600 frames) on a single A100 node (8 GPUs).
*   **Open-Ended Reward:** Support for training with open-ended questions.

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
10. [How to Contribute](#how-to-contribute)
11. [Core Contributors](#core-contributors)
12. [Citation](#citation)
13. [Acknowledgement](#acknowledgement)

## News

*   [x] \[2025.9.18] **Long-RL** has been accepted by NeurIPS 2025.
*   [x] \[2025.7.30] **LongVILA-R1-7B** supports processing up to **8,192** video frames per video, with configurable FPS settings. Please refer to its [usage instructions](#longvila-r1-model-usage).
*   [x] \[2025.7.24] We release a gradio demo (https://long-rl.hanlab.ai) with our LongVILA-R1-7B model deployed.
*   [x] \[2025.7.24] We release the model weights of **LongVILA-R1-7B** on HuggingFace (https://huggingface.co/Efficient-Large-Model/LongVILA-R1-7B). LongVILA-R1-7B achieves **65.1% / 71.1%** on VideoMME. It supports reasoning on both **multiple-choice** and **open-ended** questions, and can also switch to non-thinking mode.
*   [x] \[2025.7.19] We release a detailed instruction and scripts for the data generation process of our LongVideo-Reason dataset in the [`longvideo-reason`](longvideo-reason/) directory.
*   [x] \[2025.7.18] We release new supported features, including *Open-ended reward*, *Cached video embeddings*, and *Chunked gathering* as introduced in [Supported Features](#supported-features).
*   [x] \[2025.7.10] We release [Paper](https://arxiv.org/abs/2507.07966) and this GitHub repo [Long-RL](https://github.com/NVlabs/Long-RL).

## Highlights

1.  **Hour-level long video RL training on a single node**:  We supports RL training on hour-level videos (3,600 frames - 256k tokens) with sequence parallel, on a single A100 node (8 GPUs). `examples/new_supports/qwen2_5_vl_3b_video_1h.sh`
2.  **Omni-model RL**: We supports RL training on omni models, that take text, video, and audio for inputs. `examples/new_supports/qwen2_5_omni_3b_grpo.sh`
3.  **Image/video generation RL**:  We supports RL training on image/video generation models, like [Stable Diffusion](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium) and [Wan](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers) series models. `examples/new_supports/sd3_image_grpo.sh` and `examples/new_supports/wan_video_grpo.sh`.

## Introduction

**Key Capabilities:**

*   Scales vision-language models (VLMs) to long videos using reinforcement learning.
*   Addresses the unique challenges of long video reasoning.
*   Includes a large-scale dataset, LongVideo-Reason, with 104K QA pairs.
*   Employs a two-stage training pipeline: Chain-of-thought supervised fine-tuning (CoT-SFT) and RL.
*   Features the Multi-modal Reinforcement Sequence Parallelism (MR-SP) system for efficient long video RL training.

**Supported Models:**

*   VILA series models
*   Qwen-VL series models
*   Image and video diffusion models

**Supported Algorithms:**

*   GRPO
*   DAPO
*   Reinforce

## LongVILA-R1 Model Usage

### General Inference

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

*   **Open-ended reward**: Support training for open-ended QAs. Requires setting `--worker.rollout.open_ended_reward=True` and exporting your OpenAI API key.
*   **Cached video embeddings**: Accelerates training by caching video embeddings. Requires caching embeddings using `verl/utils/cache_video_embeds_vila.py` and setting `--data.cache_dir` and `--worker.actor.cached_embeds_dir`.
*   **Chunked gathering**: Mitigates CPU OOM issues with large batches or long video frames.  Use `--worker.rollout.num_chunk_seq` in the training script.

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

### Single node

Refer to the training scripts in the `examples` directory (e.g., `examples/new_supports/qwen2_5_vl_3b_video_grpo.sh $VIDEO_PATH`).

### Multi-nodes

Use the provided `srun_multi_nodes.sh` script, where `TRAIN_SCRIPT` is the single-node training script and `NNODES` is the number of nodes required.

```bash
bash scripts/srun_multi_nodes.sh $TRAIN_SCRIPT $NNODES
```

Example:

```bash
bash scripts/srun_multi_nodes.sh examples/new_supports/qwen2_5_vl_3b_video_grpo.sh 2
```

### Merge Checkpoint in Hugging Face Format

```bash
python3 scripts/model_merger.py --local_dir checkpoints/easy_r1/exp_name/global_step_1/actor
```

## LongVideo-Reason

Detailed instructions and evaluation details for the `LongVideo-Reason` benchmark are in the [`longvideo-reason`](longvideo-reason/) directory.

## Examples

[Provide example video links with image previews]

## How to Contribute

*   Ensure Git is installed.
*   Create a fork of the project.
*   Clone your fork locally.
*   Follow the [Installation](#installation) instructions.
*   Commit and push your changes.
*   Submit a pull request.

## Core Contributors

[Yukang Chen](https://yukangchen.com/), [Wei Huang](https://aaron-weihuang.com/), [Shuai Yang](https://andysonys.github.io), [Qinghao Hu](https://tonyhao.xyz/), [Baifeng Shi](https://bfshi.github.io/), [Hanrong Ye](https://sites.google.com/site/yhrspace/home), [Ligeng Zhu](https://lzhu.me/).

We welcome all contributions and will acknowledge them.

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

*   [EasyR1](https://github.com/hiyouga/EasyR1) - the codebase we built upon.
*   [verl](https://github.com/volcengine/verl) - the RL training framework we built upon.
*   [vllm](https://github.com/vllm-project/vllm) - for the rollout engine.
*   [Flow-GRPO](https://github.com/yifan123/flow_grpo) - for the image/video generation RL.