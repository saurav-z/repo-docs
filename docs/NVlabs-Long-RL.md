<p align="center" width="100%">
<img src="assets/long-rl-logo.png" alt="Long-RL Logo" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

# Long-RL: Revolutionizing Long Video Reasoning with Reinforcement Learning

**Tackle the challenges of long video reasoning with Long-RL, a groundbreaking framework leveraging reinforcement learning to scale vision-language models (VLMs).**  Explore our research, models, and training infrastructure on the [original Long-RL repository](https://github.com/NVlabs/Long-RL).

[![Paper](https://img.shields.io/badge/ArXiv-Paper-brown)](https://arxiv.org/abs/2507.07966)
[![Code](https://img.shields.io/badge/GitHub-Long%20RL-blue)](https://github.com/NVlabs/Long-RL)
[![Model](https://img.shields.io/badge/HuggingFace-Model-yellow)](https://huggingface.co/Efficient-Large-Model/LongVILA-R1-7B)
[![Video](https://img.shields.io/badge/YouTube-Video-red)](https://www.youtube.com/watch?v=ykbblK2jiEg)
[![Demo](https://img.shields.io/badge/Gradio-Demo-bron)](https://long-rl.hanlab.ai)

<div align="center">

[![Watch the video](assets/demo_video_first_frame.png)](https://www.youtube.com/watch?v=ykbblK2jiEg)

</div>

## Key Features

*   **LongVILA-R1-7B**: Achieve state-of-the-art performance on video benchmarks, supporting up to 8,192 video frames and configurable FPS.
*   **Multi-modal Reinforcement Sequence Parallelism (MR-SP)**: Train on hour-long videos with sequence parallelism for efficient RL training.
*   **Comprehensive Support**: Train on text, video, and audio inputs with various models (VILA, Qwen) and algorithms (GRPO, DAPO, Reinforce).
*   **Open-Source Framework**: Explore the training framework for various modalities and models.
*   **LongVideo-Reason Dataset**: Leverage a large-scale dataset with 104K long video QA pairs for advanced reasoning.

## Highlights

*   **Hour-level Long Video RL Training**: Train on videos up to an hour long (3,600 frames) on a single A100 node (8 GPUs) using sequence parallel.
*   **Omni-Model RL Support**: Train with models that take text, video, and audio inputs.
*   **Image/Video Generation RL**: Train on image and video generation models like Stable Diffusion and Wan.

##  Model Performance (Key Metrics)

| Models             | VideoMME (w/o sub) | VideoMME (w sub) | ActivityNet-QA (test) | LongVideoBench (val) | PerceptionTest (val) | NExT-QA (mc) | VNBench (val) |
|:-------------------|:------------------:|:----------------:|:---------------------:|:--------------------:|:--------------------:|:--------:|:-------------:|
| **LongVILA-7B**    |      **60.1**      |     **65.1**     |       **59.5**        |       **57.1**       |       **58.1**       | **80.7** |   **63.0**    |
| **LongVILA-R1-7B** |      **65.1**      |     **71.1**     |       **64.8**        |       **58.0**       |       **68.9**       | **81.5** |   **75.5**    |

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
11. [Core Contributors](#core-Contributors)
12. [Citation](#citation)
13. [Acknowledgement](#acknowledgement)

## News

*   **[2025.7.30]**:  **LongVILA-R1-7B** supports up to **8,192** video frames per video and configurable FPS.
*   **[2025.7.24]**: Gradio demo released (https://long-rl.hanlab.ai) and model weights for **LongVILA-R1-7B** on HuggingFace (https://huggingface.co/Efficient-Large-Model/LongVILA-R1-7B).
*   **[2025.7.19]**: Data generation instructions and scripts released for the `LongVideo-Reason` dataset.
*   **[2025.7.18]**: New features: Open-ended reward, Cached video embeddings, and Chunked gathering.
*   **[2025.7.10]**: Paper release on arXiv and GitHub repository launch.

## Introduction

**Supported Models**:

*   VILA series (image and video)
*   Qwen-VL series (text, image, video, and audio)
*   Image and video diffusion models

**Supported Algorithms**:

*   GRPO, DAPO & Reinforce

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

*   **Open-ended reward**:  Support for open-ended QA with the specified steps in the readme.
*   **Cached video embeddings**:  Utilize cached embeddings for faster RL training.
*   **Chunked gathering**: Supports chunked gathering for efficient memory usage.

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

Refer to the training scripts in the `examples` directory.

### Multi-nodes

Refer to the EasyR1 repo for multi-node training setup.

```bash
bash scripts/srun_multi_nodes.sh $TRAIN_SCRIPT $NNODES
```

### Merge Checkpoint in Hugging Face Format

```bash
python3 scripts/model_merger.py --local_dir checkpoints/easy_r1/exp_name/global_step_1/actor
```

## LongVideo-Reason

Access detailed instructions and evaluation methods for the `LongVideo-Reason` benchmark.

## Examples

[Football Video](https://drive.google.com/file/d/1QJ-ZsDrmYS8v1XU4eWfYu5oHuXeyGSdK/view?usp=share_link)

<p align="center" width="100%">
<img src="assets/example-football.png" alt="Football Example" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

[Texas Hold’em Video](https://drive.google.com/file/d/1U0N563a2s24o_NDie1VfWauxFuSu31wC/view?usp=share_link)

<p align="center" width="100%">
<img src="assets/example-TexasHold.png" alt="Texas Hold'em Example" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

[Starcraft II Video](https://drive.google.com/file/d/1rnF4I6-EBpqhzA0SnwyajpxbAhMezDCn/view?usp=share_link)

<p align="center" width="100%">
<img src="assets/example-starcraft2.png" alt="Starcraft II Example" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

[Moving Cup Video](https://drive.google.com/file/d/1lo1E_bXXnMmWnFRudaSUgxMNxetEDHP9/view?usp=share_link)

<p align="center" width="100%">
<img src="assets/example-movingcup.png" alt="Moving Cup Example" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

## How to Contribute

*   Install git.
*   Fork the project.
*   Clone the repository.
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

*   [EasyR1](https://github.com/hiyouga/EasyR1): Codebase we built upon.
*   [verl](https://github.com/volcengine/verl): The RL training framework we built upon.
*   [vllm](https://github.com/vllm-project/vllm):  Used for the rollout engine.
*   [Flow-GRPO](https://github.com/yifan123/flow_grpo): Referenced for the image/video generation RL component.