<p align="center" width="100%">
<img src="assets/long-rl-logo.png" alt="Long-RL Logo" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

# Long-RL: Revolutionizing Long Video Reasoning with Reinforcement Learning

**Tackle the challenge of long video reasoning by leveraging reinforcement learning with Long-RL, unlocking unprecedented capabilities for vision-language models.**  For the latest updates, visit the [original Long-RL repository](https://github.com/NVlabs/Long-RL).

[![Paper](https://img.shields.io/badge/ArXiv-Paper-brown)](https://arxiv.org/abs/2507.07966)
[![Code](https://img.shields.io/badge/GitHub-Long%20RL-blue)](https://github.com/NVlabs/Long-RL)
[![Model](https://img.shields.io/badge/HuggingFace-Model-yellow)](https://huggingface.co/Efficient-Large-Model/LongVILA-R1-7B)
[![Video](https://img.shields.io/badge/YouTube-Video-red)](https://www.youtube.com/watch?v=ykbblK2jiEg)
[![Demo](https://img.shields.io/badge/Gradio-Demo-bron)](https://long-rl.hanlab.ai)

<div align="center">
  <a href="https://www.youtube.com/watch?v=ykbblK2jiEg">
    <img src="assets/demo_video_first_frame.png" alt="Demo Video" width="400">
  </a>
</div>

## Key Features

*   **Scalable Architecture:** Designed to handle long video sequences, supporting up to 8,192 video frames.
*   **High-Quality Dataset:** Utilizes LongVideo-Reason, a large-scale dataset with 104K long video QA pairs.
*   **Two-Stage Training Pipeline:** Employs chain-of-thought supervised fine-tuning (CoT-SFT) and reinforcement learning (RL) for enhanced performance.
*   **Efficient Training Infrastructure:**  Features Multi-modal Reinforcement Sequence Parallelism (MR-SP) for optimized RL training on long videos, including vLLM-based engine, cached video embeddings, and chunked gathering.
*   **Strong Performance:** Achieves state-of-the-art results on video benchmarks with LongVILA-R1-7B, outperforming LongVILA-7B.
*   **Multi-Modal Support:** Supports training on various modalities like video, text, and audio, as well as image and video generation models.
*   **Faster Training:** The MR-SP system achieves up to a 2.1x speedup on long video RL training.
*   **Flexible FPS:** Supports configurable FPS settings for diverse video processing needs.
*   **Open-ended reward:** Support training for open-ended QA (non-multi-choices QA).
*   **Cached video embeddings:** Support using cached video embeddings for faster video RL training.

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

*   \[2025.07.30] **LongVILA-R1-7B** now supports up to **8,192** video frames per video with configurable FPS settings. See the [usage instructions](#longvila-r1-model-usage).
*   \[2025.07.24]  A Gradio demo is available featuring the LongVILA-R1-7B model: [Gradio Demo](https://long-rl.hanlab.ai).
*   \[2025.07.24] Model weights for **LongVILA-R1-7B** are released on HuggingFace: [Hugging Face](https://huggingface.co/Efficient-Large-Model/LongVILA-R1-7B).  LongVILA-R1-7B achieves **65.1% / 71.1%** on VideoMME, supporting reasoning for both multiple-choice and open-ended questions, and non-thinking mode.
*   \[2025.07.19] Detailed instructions and scripts for the LongVideo-Reason dataset generation are available in the [`longvideo-reason`](longvideo-reason/) directory.
*   \[2025.07.18] New supported features include *Open-ended reward*, *Cached video embeddings*, and *Chunked gathering* [Supported Features](#supported-features).
*   \[2025.07.10] The [paper](https://arxiv.org/abs/2507.07966) and this GitHub repository [Long-RL](https://github.com/NVlabs/Long-RL) were released.

## Highlights

1.  **Hour-Level Long Video RL Training:** Supports RL training on hour-level videos (3,600 frames - 256k tokens) with sequence parallelism on a single A100 node (8 GPUs). See `examples/new_supports/qwen2_5_vl_3b_video_1h.sh`.
2.  **Omni-Model RL:** Supports RL training on omni-models that take text, video, and audio as inputs.  See `examples/new_supports/qwen2_5_omni_3b_grpo.sh`.
3.  **Image/Video Generation RL:** Supports RL training on image/video generation models like [Stable Diffusion](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium) and [Wan](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers) series models.  See `examples/new_supports/sd3_image_grpo.sh` and `examples/new_supports/wan_video_grpo.sh`.

## Introduction

Long-RL introduces a comprehensive framework for scaling reasoning in Vision-Language Models (VLMs) to process and understand long videos using reinforcement learning. The project addresses the unique challenges of long video reasoning by integrating a large-scale dataset (LongVideo-Reason), a two-stage training pipeline (CoT-SFT and RL), and a novel training infrastructure (Multi-modal Reinforcement Sequence Parallelism - MR-SP). This approach enables robust and efficient long video processing, leading to significant performance improvements on video understanding benchmarks.

**Supported Models:**

*   \[x]  VILA series models on image and video, with SP support.
    *   `examples/new_supports/nvila_2b_clevr_grpo.sh`
    *   `examples/new_supports/nvila_2b_video_grpo.sh`
    *   `examples/new_supports/longvila_7b_video_grpo.sh`
*   \[x]  Qwen-VL series models on text, image, video, and audio, with SP support.
    *   `examples/new_supports/qwen2_5_3b_math_grpo.sh`
    *   `examples/new_supports/qwen2_5_vl_3b_video_grpo.sh`
    *   `examples/new_supports/qwen2_5_omni_3b_grpo.sh`
*   \[x]  Image and video diffusion model RL
    *   `examples/new_supports/sd3_image_grpo.sh`
    *   `examples/new_supports/wan_video_grpo.sh`

**Supported Algorithms:**

*   \[x]  GRPO, DAPO & Reinforce supported, with SP support.
    *   `examples/new_supports/qwen2_5_vl_3b_video_dapo.sh`
    *   `examples/new_supports/qwen2_5_vl_3b_video_grpo.sh`
    *   `examples/new_supports/qwen2_5_vl_3b_video_reinforce.sh`

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


use_thinking = True  # Switching between thinking and non-thinking modes
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

model_encoder = AutoModel.from_pretrained(
    model_path, trust_remote_code=True, device_map="auto", llm_only_need_embed=True
)
# you can change gpu_memory_utilization according to GPU memory
llm = LLM(model=os.path.join(model_path, "llm"), enable_prompt_embeds=True, gpu_memory_utilization=0.5)

use_thinking = True  # Switching between thinking and non-thinking modes
system_prompt_thinking = "You are a helpful assistant. The user asks a question, and then you solves it.\n\nPlease first think deeply about the question based on the given video, and then provide the final answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\n\n Question: {question}"

prompt = "What is the main purpose of the video?"
video_path = "video.mp4"

if use_thinking:
    prompt = system_prompt_thinking.format(question=prompt)

conversation = [{"from": "human", "value": [prompt, {"path": video_path}]}]
media = extract_media(conversation, model_encoder.config)
input_ids = tokenize_conversation(
    conversation, model_encoder.tokenizer, add_generation_prompt=True
).unsqueeze(0).cuda()
media["video"] = [
    process_images(
        images,
        model_encoder.vision_tower.image_processor,
        model_encoder.config,
    ).half()
    for images in media["video"]
]

inputs_embeds, _, _ = model_encoder._embed(input_ids, media, {"video": {}}, None, None)

completions = llm.generate(
    prompts=[{"prompt_embeds": inputs_embeds.squeeze(0)}],
    sampling_params=SamplingParams(max_tokens=1024),
)
response = completions[0].outputs[0].text
print("Response: ", response)
```

## Supported Features

*   \[x] **Open-ended reward**: Supports training for open-ended QA (non-multiple-choice QAs).
    *   Set `--worker.rollout.open_ended_reward=True` in the training script.
    *   Export your OpenAI API key: `export OPENAI_API_KEY=xxx`.
*   \[x] **Cached video embeddings**: Supports using cached video embeddings for faster video RL training.
    *   Follow `verl/utils/cache_video_embeds_vila.py` to cache video embeddings.
    *   Set `--data.cache_dir` and `--worker.actor.cached_embeds_dir` in the training script.
*   \[x] **Chunked gathering**: Supports chunked gathering for `all_gather_data_proto` to address potential CPU OOM issues.
    *   Set `--worker.rollout.num_chunk_seq` in the training script (e.g., 8/16/32).

## Installation

```bash
git clone https://github.com/NVlabs/Long-RL.git
cd Long-RL
pip install -e .
```

If training Qwen-Omni models, run:

```bash
bash vllm_replace.sh
```

## Training

### Single Node

Refer to the training scripts in the `examples` directory (e.g., `examples/new_supports/qwen2_5_vl_3b_video_grpo.sh $VIDEO_PATH`).

### Multi-Nodes

Refer to the multi-node setup described in the EasyR1 repo.

Example `sbatch` script:

```bash
bash scripts/srun_multi_nodes.sh $TRAIN_SCRIPT $NNODES
```

Example:

```bash
bash scripts/srun_multi_nodes.sh examples/new_supports/qwen2_5_vl_3b_video_grpo.sh 2
```

### Merge Checkpoint in Hugging Face Format

Follow the EasyR1 repo's instructions:

```bash
python3 scripts/model_merger.py --local_dir checkpoints/easy_r1/exp_name/global_step_1/actor
```

## LongVideo-Reason

Detailed instructions for data generation and model evaluation on the `LongVideo-Reason` benchmark are available in the [`longvideo-reason`](longvideo-reason/) directory.

## Examples

<div align="center">
  <a href="https://drive.google.com/file/d/1QJ-ZsDrmYS8v1XU4eWfYu5oHuXeyGSdK/view?usp=share_link">
    <img src="assets/example-football.png" alt="Football Video Example" width="400">
  </a>
</div>

<div align="center">
  <a href="https://drive.google.com/file/d/1U0N563a2s24o_NDie1VfWauxFuSu31wC/view?usp=share_link">
    <img src="assets/example-TexasHold.png" alt="Texas Hold’em Video Example" width="400">
  </a>
</div>

<div align="center">
  <a href="https://drive.google.com/file/d/1rnF4I6-EBpqhzA0SnwyajpxbAhMezDCn/view?usp=share_link">
    <img src="assets/example-starcraft2.png" alt="Starcraft II Video Example" width="400">
  </a>
</div>

<div align="center">
  <a href="https://drive.google.com/file/d/1lo1E_bXXnMmWnFRudaSUgxMNxetEDHP9/view?usp=share_link">
    <img src="assets/example-movingcup.png" alt="Moving Cup Video Example" width="400">
  </a>
</div>

## How to Contribute

*   Install git.
*   Create a [fork](https://github.com/NVlabs/Long-RL/fork) of the project.
*   Clone your fork to your local machine.
*   Follow the [Installation](#installation) steps.
*   Commit and push your changes.
*   Create a pull request.

## Core Contributors

[Yukang Chen](https://yukangchen.com/), [Wei Huang](https://aaron-weihuang.com/), [Shuai Yang](https://andysonys.github.io), [Qinghao Hu](https://tonyhao.xyz/), [Baifeng Shi](https://bfshi.github.io/), [Hanrong Ye](https://sites.google.com/site/yhrspace/home), [Ligeng Zhu](https://lzhu.me/).

Contributions are welcomed and will be acknowledged.

## Citation

Please cite our paper and the framework if they are helpful for your research:

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

*   [EasyR1](https://github.com/hiyouga/EasyR1): Core codebase.
*   [verl](https://github.com/volcengine/verl): RL training framework.
*   [vllm](https://github.com/vllm-project/vllm): Rollout engine.
*   [Flow-GRPO](https://github.com/yifan123/flow_grpo): Inspiration for the image/video generation RL part.