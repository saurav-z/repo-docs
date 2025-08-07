<p align="center" width="100%">
<img src="assets/long-rl-logo.png" alt="Long-RL Logo" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

# Long-RL: Revolutionizing Long Video Reasoning with Reinforcement Learning

**Tackle the challenge of long video understanding with Long-RL, a cutting-edge framework that scales Reinforcement Learning (RL) to process and reason over lengthy video sequences.** [Explore the Long-RL Repository](https://github.com/NVlabs/Long-RL)

[![Paper](https://img.shields.io/badge/ArXiv-Paper-brown)](https://arxiv.org/abs/2507.07966)
[![Code](https://img.shields.io/badge/GitHub-Long%20RL-blue)](https://github.com/NVlabs/Long-RL)
[![Model](https://img.shields.io/badge/HuggingFace-Model-yellow)](https://huggingface.co/Efficient-Large-Model/LongVILA-R1-7B)
[![Video](https://img.shields.io/badge/YouTube-Video-red)](https://www.youtube.com/watch?v=ykbblK2jiEg)
[![Demo](https://img.shields.io/badge/Gradio-Demo-bron)](https://long-rl.hanlab.ai)

<div align="center">
    <a href="https://www.youtube.com/watch?v=ykbblK2jiEg">
        <img src="assets/demo_video_first_frame.png" alt="Demo Video First Frame" style="width: 100%; min-width: 300px; display: block; margin: auto;">
    </a>
</div>

## Key Features

*   **Scalable RL for Long Videos:**  Train on hour-long videos (up to 3,600 frames) with sequence parallelism on a single A100 node (8 GPUs).
*   **Multi-Modal Support:** Train on models with diverse inputs, including video, text, and audio.
*   **Model Agnostic:** Supports training on various models, including VILA, Qwen series, and image/video generation models (Stable Diffusion, Wan).
*   **Optimized Training Infrastructure:** Utilizes Multi-modal Reinforcement Sequence Parallelism (MR-SP) for efficient long video RL training, achieving up to 2.1x speedup.
*   **LongVILA-R1-7B Model:**  Achieves strong performance on video benchmarks, reaching 65.1% and 71.1% accuracy on VideoMME (without and with subtitles), supports up to 8,192 video frames.
*   **Open-ended Reward Support:** Supports training on open-ended QA tasks.
*   **Efficiency Enhancements:**  Includes features like cached video embeddings and chunked gathering for reduced memory usage and faster training.

## What's New

*   **LongVILA-R1-7B Enhanced:** Supports up to **8,192** video frames per video, configurable FPS settings.
*   **Gradio Demo Released:**  Interact with LongVILA-R1-7B via a public Gradio demo: [https://long-rl.hanlab.ai](https://long-rl.hanlab.ai)
*   **Hugging Face Model Release:** Access the LongVILA-R1-7B model weights on Hugging Face: [https://huggingface.co/Efficient-Large-Model/LongVILA-R1-7B](https://huggingface.co/Efficient-Large-Model/LongVILA-R1-7B)
*   **LongVideo-Reason Dataset Instructions:** Detailed data generation instructions and scripts are available in the [`longvideo-reason/`](longvideo-reason/) directory.
*   **New Features:** Introduced open-ended reward, cached video embeddings, and chunked gathering.

## Table of Contents

1.  [Introduction](#introduction)
2.  [LongVILA-R1 Model Usage](#longvila-r1-model-usage)
3.  [Supported Features](#supported-features)
4.  [Installation](#installation)
5.  [Training](#training)
6.  [LongVideo-Reason](#longvideo-reason)
7.  [Examples](#examples)
8.  [How to Contribute](#how-to-contribute)
9.  [Core Contributors](#core-contributors)
10. [Citation](#citation)
11. [Acknowledgement](#acknowledgement)

## Introduction

Long-RL introduces a novel full-stack framework that leverages reinforcement learning to scale reasoning in vision-language models (VLMs) for long videos.  It addresses challenges with:

*   **LongVideo-Reason Dataset:** A large-scale dataset with 104K long video QA pairs.
*   **Two-Stage Training Pipeline:** Integrates chain-of-thought supervised fine-tuning (CoT-SFT) and RL.
*   **Multi-modal Reinforcement Sequence Parallelism (MR-SP):**  A training infrastructure optimized for long video RL.

**Supported Models:**

*   VILA series
*   Qwen-VL series
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

*   **Open-ended reward**: Training for open-ended QA (non-multi-choices). Configure with `--worker.rollout.open_ended_reward=True` and `export OPENAI_API_KEY=xxx`.
*   **Cached video embeddings**: Use cached embeddings for faster training by setting `--data.cache_dir` and `--worker.actor.cached_embeds_dir`.
*   **Chunked gathering**: Supports chunked gathering for `all_gather_data_proto` to avoid CPU OOM with `--worker.rollout.num_chunk_seq`.

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

Refer to the training scripts in the `examples` directory. For example:

```bash
bash examples/new_supports/qwen2_5_vl_3b_video_grpo.sh $VIDEO_PATH
```

### Multi-nodes

Refer to the EasyR1 repo for multi-node training instructions. Use `scripts/srun_multi_nodes.sh` with the training script and the number of nodes:

```bash
bash scripts/srun_multi_nodes.sh examples/new_supports/qwen2_5_vl_3b_video_grpo.sh 2
```

### Merge Checkpoint in Hugging Face Format

Use the `scripts/model_merger.py` script:

```bash
python3 scripts/model_merger.py --local_dir checkpoints/easy_r1/exp_name/global_step_1/actor
```

## LongVideo-Reason

Detailed instructions on the dataset and evaluation are available in the [`longvideo-reason`](longvideo-reason/) directory.

## Examples

<div align="center">
    <a href="https://drive.google.com/file/d/1QJ-ZsDrmYS8v1XU4eWfYu5oHuXeyGSdK/view?usp=share_link">
        Football Video
    </a>
</div>
<p align="center" width="100%">
    <img src="assets/example-football.png" alt="Football Video Example" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

<div align="center">
    <a href="https://drive.google.com/file/d/1U0N563a2s24o_NDie1VfWauxFuSu31wC/view?usp=share_link">
        Texas Hold’em Video
    </a>
</div>
<p align="center" width="100%">
    <img src="assets/example-TexasHold.png" alt="Texas Hold’em Video Example" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

<div align="center">
    <a href="https://drive.google.com/file/d/1rnF4I6-EBpqhzA0SnwyajpxbAhMezDCn/view?usp=share_link">
        Starcraft II Video
    </a>
</div>
<p align="center" width="100%">
    <img src="assets/example-starcraft2.png" alt="Starcraft II Video Example" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

<div align="center">
    <a href="https://drive.google.com/file/d/1lo1E_bXXnMmWnFRudaSUgxMNxetEDHP9/view?usp=share_link">
        Moving Cup Video
    </a>
</div>
<p align="center" width="100%">
    <img src="assets/example-movingcup.png" alt="Moving Cup Video Example" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

## How to Contribute

*   Install git.
*   Fork the project:  [https://github.com/NVlabs/Long-RL/fork](https://github.com/NVlabs/Long-RL/fork)
*   Clone your fork.
*   Install dependencies (see Installation).
*   Make changes, commit, and push.
*   Submit a pull request.

## Core Contributors

[Yukang Chen](https://yukangchen.com/), [Wei Huang](https://aaron-weihuang.com/), [Shuai Yang](https://andysonys.github.io), [Qinghao Hu](https://tonyhao.xyz/), [Baifeng Shi](https://bfshi.github.io/), [Hanrong Ye](https://sites.google.com/site/yhrspace/home), [Ligeng Zhu](https://lzhu.me/).

Contributions are welcome and will be acknowledged.

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

*   [EasyR1](https://github.com/hiyouga/EasyR1):  Base codebase.
*   [verl](https://github.com/volcengine/verl):  RL training framework.
*   [vllm](https://github.com/vllm-project/vllm):  Rollout engine.
*   [Flow-GRPO](https://github.com/yifan123/flow_grpo):  Image/video generation RL reference.