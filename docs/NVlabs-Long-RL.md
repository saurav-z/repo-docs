<p align="center" width="100%">
<img src="assets/long-rl-logo.png" alt="Long-RL Logo" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

# Long-RL: Revolutionizing Long Video Reasoning with Reinforcement Learning

**Tackle the challenges of long video reasoning with Long-RL, a full-stack framework that scales vision-language models (VLMs) using reinforcement learning for improved understanding and analysis of lengthy video content. ([Original Repo](https://github.com/NVlabs/Long-RL))**

[![Paper](https://img.shields.io/badge/ArXiv-Paper-brown)](https://arxiv.org/abs/2507.07966)
[![Code](https://img.shields.io/badge/GitHub-Long%20RL-blue)](https://github.com/NVlabs/Long-RL)
[![Model](https://img.shields.io/badge/HuggingFace-Model-yellow)](https://huggingface.co/Efficient-Large-Model/LongVILA-R1-7B)
[![Video](https://img.shields.io/badge/YouTube-Video-red)](https://www.youtube.com/watch?v=ykbblK2jiEg)
[![Demo](https://img.shields.io/badge/Gradio-Demo-bron)](https://long-rl.hanlab.ai)

<div align="center">

[![Watch the video](assets/demo_video_first_frame.png)](https://www.youtube.com/watch?v=ykbblK2jiEg)

</div>

## Key Features

*   **Advanced Performance:** LongVILA-R1-7B achieves strong performance on video benchmarks, achieving **65.1% / 71.1%** on VideoMME.
*   **Enhanced Long Video Understanding**:  Models handle up to **8,192 video frames**, with configurable FPS settings.
*   **Comprehensive Dataset:** Leverages LongVideo-Reason, a large-scale dataset with 104K long video QA pairs.
*   **Innovative Training Pipeline:** Employs a two-stage training pipeline integrating chain-of-thought supervised fine-tuning (CoT-SFT) and reinforcement learning (RL).
*   **Efficient Training Infrastructure**: Utilizes Multi-modal Reinforcement Sequence Parallelism (MR-SP) for up to **2.1x speedup** in long video RL training.
*   **Multi-Modal Support:**  Supports RL training on various modalities (video, text, and audio), models (VILA and Qwen series), and image/video generation models.
*   **Flexible Deployment**:  Includes a Gradio demo for interactive exploration of LongVILA-R1-7B capabilities.
*   **Single Node Training**: Supports RL training on hour-long videos (e.g., 3,600 frames / around 256k tokens) on a single A100 node (8 GPUs).

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

*   [x] \[2025.7.30] **LongVILA-R1-7B** now processes up to **8,192** video frames, with customizable FPS.
*   [x] \[2025.7.24] Gradio demo (https://long-rl.hanlab.ai) released.
*   [x] \[2025.7.24] **LongVILA-R1-7B** model weights released on HuggingFace (https://huggingface.co/Efficient-Large-Model/LongVILA-R1-7B).
*   [x] \[2025.7.19] Data generation instructions for LongVideo-Reason are available.
*   [x] \[2025.7.18] New features: Open-ended reward, Cached video embeddings, and Chunked gathering.
*   [x] \[2025.7.10] Paper and GitHub repository released.

## Highlights

1.  **Hour-Level Long Video RL Training:** Train on hour-long videos (3,600 frames / 256k tokens) using sequence parallelism on a single A100 node (8 GPUs).
    `examples/new_supports/qwen2_5_vl_3b_video_1h.sh`
2.  **Omni-Model RL:** Train on models with text, video, and audio inputs.
    `examples/new_supports/qwen2_5_omni_3b_grpo.sh`
3.  **Image/Video Generation RL:** Train on image/video generation models like Stable Diffusion and Wan series models.
    `examples/new_supports/sd3_image_grpo.sh` and `examples/new_supports/wan_video_grpo.sh`.

## Introduction

Long-RL is a full-stack framework that scales reasoning in vision-language models (VLMs) to long videos, using reinforcement learning. This is achieved through a comprehensive approach:

*   **Large-Scale Dataset:** LongVideo-Reason dataset with 104K long video QA pairs.
*   **Two-Stage Training:**  CoT-SFT followed by RL.
*   **Efficient Training Infrastructure:** Multi-modal Reinforcement Sequence Parallelism (MR-SP) for fast long video RL training.

**Supported Models:**

*   VILA series (image and video)
*   Qwen-VL series (text, image, video, and audio)
*   Image and video diffusion model RL

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

```bash
mkdir remote_code
cp path_to/Efficient-Large-Model/LongVILA-R1-7B/*.py remote_code
```

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

*   **Open-ended reward:** Support for open-ended QA. Configure `--worker.rollout.open_ended_reward=True` in training and set `OPENAI_API_KEY`.
*   **Cached video embeddings:** Use cached video embeddings to speed up video RL training. Cache embeddings and configure `--data.cache_dir` and `--worker.actor.cached_embeds_dir`.
*   **Chunked gathering:** Reduce CPU OOM issues and memory usage using chunked gathering via `--worker.rollout.num_chunk_seq`.

## Installation

```bash
git clone https://github.com/NVlabs/Long-RL.git
cd Long-RL
pip install -e .
```

If you plan to train Qwen-Omni models:

```bash
bash vllm_replace.sh
```

## Training

### Single Node

Refer to example scripts in the `examples` directory. For example:

```bash
bash examples/new_supports/qwen2_5_vl_3b_video_grpo.sh $VIDEO_PATH
```

### Multi-Nodes

Follow the multi-node instructions from EasyR1 [here](https://github.com/hiyouga/EasyR1/tree/main?tab=readme-ov-file#how-to-run-70b-model-in-multi-node-environment).

Example `sbatch` script:

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

Detailed instructions on data generation and model evaluation are in the [`longvideo-reason`](longvideo-reason/) directory.

## Examples

<div align="center">
<a href="https://drive.google.com/file/d/1QJ-ZsDrmYS8v1XU4eWfYu5oHuXeyGSdK/view?usp=share_link">Football Video</a>
</div>
<p align="center" width="100%">
<img src="assets/example-football.png" alt="Stanford-Alpaca" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

<div align="center">
<a href="https://drive.google.com/file/d/1U0N563a2s24o_NDie1VfWauxFuSu31wC/view?usp=share_link">Texas Hold’em Video</a>
</div>
<p align="center" width="100%">
<img src="assets/example-TexasHold.png" alt="Stanford-Alpaca" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

<div align="center">
<a href="https://drive.google.com/file/d/1rnF4I6-EBpqhzA0SnwyajpxbAhMezDCn/view?usp=share_link">Starcraft II Video</a>
</div>
<p align="center" width="100%">
<img src="assets/example-starcraft2.png" alt="Stanford-Alpaca" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

<div align="center">
<a href="https://drive.google.com/file/d/1lo1E_bXXnMmWnFRudaSUgxMNxetEDHP9/view?usp=share_link">Moving Cup Video</a>
</div>
<p align="center" width="100%">
<img src="assets/example-movingcup.png" alt="Stanford-Alpaca" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

## How to Contribute

*   Install git.
*   Fork the project.
*   Clone your fork.
*   Follow installation instructions.
*   Commit and push your changes.
*   Submit a pull request.

## Core Contributors

[Yukang Chen](https://yukangchen.com/), [Wei Huang](https://aaron-weihuang.com/), [Shuai Yang](https://andysonys.github.io), [Qinghao Hu](https://tonyhao.xyz/), [Baifeng Shi](https://bfshi.github.io/), [Hanrong Ye](https://sites.google.com/site/yhrspace/home), [Ligeng Zhu](https://lzhu.me/).

All contributions are welcome and acknowledged.

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

*   [EasyR1](https://github.com/hiyouga/EasyR1): codebase.
*   [verl](https://github.com/volcengine/verl): RL training framework.
*   [vllm](https://github.com/vllm-project/vllm): rollout engine.
*   [Flow-GRPO](https://github.com/yifan123/flow_grpo): for image/video generation RL part.