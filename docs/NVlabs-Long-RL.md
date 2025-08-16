<p align="center" width="100%">
<img src="assets/long-rl-logo.png" alt="Long-RL Logo" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

# Long-RL: Revolutionizing Long Video Reasoning with Reinforcement Learning

**Tackle the challenge of long video reasoning with Long-RL, a full-stack framework that leverages reinforcement learning to scale reasoning capabilities in vision-language models (VLMs).**  Explore the cutting-edge advancements in long-context video understanding, and experience the power of Long-RL for your projects. ([Original Repo](https://github.com/NVlabs/Long-RL))

[![Paper](https://img.shields.io/badge/ArXiv-Paper-brown)](https://arxiv.org/abs/2507.07966)
[![Code](https://img.shields.io/badge/GitHub-Long%20RL-blue)](https://github.com/NVlabs/Long-RL)
[![Model](https://img.shields.io/badge/HuggingFace-Model-yellow)](https://huggingface.co/Efficient-Large-Model/LongVILA-R1-7B)
[![Video](https://img.shields.io/badge/YouTube-Video-red)](https://www.youtube.com/watch?v=ykbblK2jiEg)
[![Demo](https://img.shields.io/badge/Gradio-Demo-bron)](https://long-rl.hanlab.ai)

<div align="center">
[![Watch the video](assets/demo_video_first_frame.png)](https://www.youtube.com/watch?v=ykbblK2jiEg)
</div>

## Key Features

*   **LongVILA-R1-7B Performance:**  Achieves strong performance on video benchmarks, including 65.1% and 71.1% accuracy on VideoMME (without and with subtitles respectively), and consistently outperforms LongVILA-7B across multiple benchmarks.
*   **Supports Long Video Sequences:**  Process up to **8,192 video frames** per video with configurable FPS settings, ideal for extensive video content analysis.
*   **Multi-Modal RL Training:** Supports RL training across various modalities including video, text, and audio, compatible with VILA, Qwen series models, and image/video generation models (Stable Diffusion, Wan series).
*   **Accelerated Training with MR-SP:**  Our Multi-modal Reinforcement Sequence Parallelism (MR-SP) system achieves up to **2.1x speedup** on long video RL training, optimizing for efficiency.
*   **Comprehensive Dataset & Training Pipeline:** Utilizes a large-scale dataset, *LongVideo-Reason*, comprising 104K long video QA pairs and a two-stage training pipeline integrating Chain-of-Thought Supervised Fine-tuning (CoT-SFT) and Reinforcement Learning (RL).
*   **Single-Node Hour-Level Training:** Supports RL training on hour-long videos (e.g., 3,600 frames) on a single A100 node (8 GPUs).
*   **Flexible Support:** Includes features such as Open-ended reward, Cached video embeddings, and Chunked gathering for improved efficiency.

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
*   **[2025.7.30]** LongVILA-R1-7B now supports processing up to 8,192 video frames per video, with configurable FPS settings. Refer to the [usage instructions](#longvila-r1-model-usage).
*   **[2025.7.24]** Released a Gradio demo (https://long-rl.hanlab.ai) with the LongVILA-R1-7B model deployed.
*   **[2025.7.24]** Released the model weights of LongVILA-R1-7B on HuggingFace (https://huggingface.co/Efficient-Large-Model/LongVILA-R1-7B). LongVILA-R1-7B achieves 65.1% / 71.1% on VideoMME and supports reasoning on both multiple-choice and open-ended questions.
*   **[2025.7.19]** Released detailed instructions and scripts for generating the LongVideo-Reason dataset in the [`longvideo-reason`](longvideo-reason/) directory.
*   **[2025.7.18]** Released new supported features, including Open-ended reward, Cached video embeddings, and Chunked gathering ([Supported Features](#supported-features)).
*   **[2025.7.10]** Released the [Paper](https://arxiv.org/abs/2507.07966) and the GitHub repository [Long-RL](https://github.com/NVlabs/Long-RL).

## Highlights

1.  **Efficient Training:** Supports RL training on hour-level videos (3,600 frames - 256k tokens) with sequence parallelism, on a single A100 node (8 GPUs). (e.g., `examples/new_supports/qwen2_5_vl_3b_video_1h.sh`)
2.  **Omni-Model RL Support:**  Supports RL training on omni models that take text, video, and audio as inputs. (e.g., `examples/new_supports/qwen2_5_omni_3b_grpo.sh`)
3.  **Image/Video Generation RL:** Supports RL training on image/video generation models such as Stable Diffusion and Wan series models. (e.g., `examples/new_supports/sd3_image_grpo.sh` and `examples/new_supports/wan_video_grpo.sh`).

## Introduction

Long-RL introduces a full-stack framework for scaling reasoning in vision-language models (VLMs) to long videos. Our approach addresses the unique challenges of long video reasoning by integrating:

*   A large-scale dataset, LongVideo-Reason.
*   A two-stage training pipeline that extends VLMs with chain-of-thought supervised fine-tuning (CoT-SFT) and reinforcement learning (RL).
*   A training infrastructure for long video RL, named Multi-modal Reinforcement Sequence Parallelism (MR-SP).

**Support Models:**

*   VILA series models on image and video, with SP support
*   Qwen-VL series models on text, image, video, and audio, with SP support
*   Image and video diffusion model RL

**Support Algorithms:**

*   GRPO, DAPO & Reinforce supported, with SP support

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

*   **Open-ended reward**: Support for training with open-ended QA, requiring setting `--worker.rollout.open_ended_reward=True` in training and exporting your OpenAI API key.
*   **Cached video embeddings**:  Supports using cached video embeddings to speed up video RL training.  Follow instructions to cache embeddings and configure `--data.cache_dir` and `--worker.actor.cached_embeds_dir`.
*   **Chunked gathering**: Supports chunked gathering to avoid CPU OOM issues. Configure with `--worker.rollout.num_chunk_seq`.

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

Use the scripts in the `examples` directory for single-node training (up to 8 GPUs).

```bash
bash examples/new_supports/qwen2_5_vl_3b_video_grpo.sh $VIDEO_PATH
```

### Multi-Nodes

For multi-node jobs, refer to the EasyR1 repo for guidance on multi-node setup.

Example `sbatch` script for multi-node execution (where `TRAIN_SCRIPT` is your single-node training script, and `NNODES` is the number of nodes):

```bash
bash scripts/srun_multi_nodes.sh $TRAIN_SCRIPT $NNODES
```

For example,

```bash
bash scripts/srun_multi_nodes.sh examples/new_supports/qwen2_5_vl_3b_video_grpo.sh 2
```

### Merge Checkpoint in Hugging Face Format

```bash
python3 scripts/model_merger.py --local_dir checkpoints/easy_r1/exp_name/global_step_1/actor
```

## LongVideo-Reason

Detailed instructions on the data generation process and model evaluation for the *LongVideo-Reason* benchmark are available in the [`longvideo-reason`](longvideo-reason/) directory.

## Examples

<div align="center">
<a href="https://drive.google.com/file/d/1QJ-ZsDrmYS8v1XU4eWfYu5oHuXeyGSdK/view?usp=share_link">Football Video</a>
</div>
<p align="center" width="100%">
<img src="assets/example-football.png" alt="Football Video Example" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

<div align="center">
<a href="https://drive.google.com/file/d/1U0N563a2s24o_NDie1VfWauxFuSu31wC/view?usp=share_link">Texas Hold’em Video</a>
</div>
<p align="center" width="100%">
<img src="assets/example-TexasHold.png" alt="Texas Hold’em Video Example" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

<div align="center">
<a href="https://drive.google.com/file/d/1rnF4I6-EBpqhzA0SnwyajpxbAhMezDCn/view?usp=share_link">Starcraft II Video</a>
</div>
<p align="center" width="100%">
<img src="assets/example-starcraft2.png" alt="Starcraft II Video Example" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

<div align="center">
<a href="https://drive.google.com/file/d/1lo1E_bXXnMmWnFRudaSUgxMNxetEDHP9/view?usp=share_link">Moving Cup Video</a>
</div>
<p align="center" width="100%">
<img src="assets/example-movingcup.png" alt="Moving Cup Video Example" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

## How to contribute

1.  Ensure you have git installed.
2.  Create a fork of the project on GitHub.
3.  Clone your forked repository locally.
4.  Follow the [Installation](#installation) instructions.
5.  Make your changes, commit, and push them to your fork.
6.  Submit a pull request to the main repository.

## Core Contributors

[Yukang Chen](https://yukangchen.com/), [Wei Huang](https://aaron-weihuang.com/), [Shuai Yang](https://andysonys.github.io), [Qinghao Hu](https://tonyhao.xyz/), [Baifeng Shi](https://bfshi.github.io/), [Hanrong Ye](https://sites.google.com/site/yhrspace/home), [Ligeng Zhu](https://lzhu.me/).

Contributions of all kinds are welcome and will be acknowledged.

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

*   [EasyR1](https://github.com/hiyouga/EasyR1):  The foundational codebase. Thanks for their contributions.
*   [verl](https://github.com/volcengine/verl): The RL training framework used.
*   [vllm](https://github.com/vllm-project/vllm): Used for the rollout engine.
*   [Flow-GRPO](https://github.com/yifan123/flow_grpo): Referenced for the image/video generation RL component.