<p align="center" width="100%">
<img src="assets/long-rl-logo.png" alt="Long-RL" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

# Long-RL: Revolutionizing Long Video Reasoning with Reinforcement Learning

**Tackle the challenge of reasoning in long videos with Long-RL, a cutting-edge framework leveraging reinforcement learning to scale Vision-Language Models (VLMs).**  ([Original Repo](https://github.com/NVlabs/Long-RL))

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

## Key Features:

*   **State-of-the-Art Performance:** Achieves impressive results on video benchmarks, including **65.1% / 71.1%** on VideoMME (w/o/with subtitles, respectively) with LongVILA-R1-7B.
*   **Extended Context Length:** Supports processing up to **8,192 video frames** per video, with configurable FPS settings, enabling in-depth analysis of lengthy content.
*   **Multi-Modal Support:**  Provides RL training support for video, text, and audio inputs.
*   **Model Versatility:** Works with a variety of models, including VILA, Qwen series, and image/video generation models (Stable Diffusion, Wan).
*   **Accelerated Training:** Offers up to **2.1x speedup** on long video RL training with Multi-modal Reinforcement Sequence Parallelism (MR-SP).
*   **Comprehensive Framework:** Includes a large-scale dataset, LongVideo-Reason, and a two-stage training pipeline (CoT-SFT & RL) for superior reasoning.
*   **Easy Deployment:**  A Gradio demo is available (https://long-rl.hanlab.ai) to experience LongVILA-R1-7B's capabilities.
*   **Flexible Training Infrastructure:** Supports RL training on various modalities (video, text, audio), various models (VILA and Qwen series), and even image and video generation models.

## Key Advantages of Long-RL

*   **Scalability:** Designed for hour-level long video RL training, even on a single node (8 GPUs).
*   **Efficiency:**  Utilizes MR-SP for optimized sequence parallelism and a vLLM-based engine for efficient rollouts and prefilling.
*   **Flexibility:** Supports various RL algorithms like GRPO, DAPO, and Reinforce.
*   **Open-Ended Reward**: Now provides a fully supported open-ended QA setting with the help of the OpenAI API.
*   **Advanced Features:** Cached video embeddings and chunked gathering are provided, greatly saving VRAM.

## Key Updates & Resources
*   **[2025.7.30] LongVILA-R1-7B** supports processing up to **8,192** video frames per video, with configurable FPS settings.
*   **[2025.7.24] Gradio Demo Release:** Experience LongVILA-R1-7B via a user-friendly demo: [https://long-rl.hanlab.ai](https://long-rl.hanlab.ai)
*   **[2025.7.24] Model Weights Released:** Download the model weights of **LongVILA-R1-7B** on HuggingFace: [https://huggingface.co/Efficient-Large-Model/LongVILA-R1-7B](https://huggingface.co/Efficient-Large-Model/LongVILA-R1-7B)
*   **[2025.7.19] Data Generation Scripts:** Detailed instructions and scripts for the data generation process of the LongVideo-Reason dataset available in the [`longvideo-reason`](longvideo-reason/) directory.
*   **[2025.7.18] New Feature Support:**  Open-ended reward, cached video embeddings, and chunked gathering introduced.
*   **[2025.7.10] Paper & Repo Release**: Published the paper [here](https://arxiv.org/abs/2507.07966) and the GitHub repo [here](https://github.com/NVlabs/Long-RL).

## Model Performance
| Models             | VideoMME (w/o sub) | VideoMME (w sub) | ActivityNet-QA (test) | LongVideoBench (val) | PerceptionTest (val) | NExT-QA (mc) | VNBench (val) |
|:-------------------|:------------------:|:----------------:|:---------------------:|:--------------------:|:--------------------:|:--------:|:-------------:|
| **LongVILA-7B**    |      **60.1**      |     **65.1**     |       **59.5**        |       **57.1**       |       **58.1**       | **80.7** |   **63.0**    |
| **LongVILA-R1-7B** |      **65.1**      |     **71.1**     |       **64.8**        |       **58.0**       |       **68.9**       | **81.5** |   **75.5**    |

## Table of Contents
1.  [Introduction](#introduction)
2.  [LongVILA-R1 Model Usage](#longvila-r1-model-usage)
3.  [Supported Features](#supported-features)
4.  [Installation](#installation)
5.  [Training](#training)
6.  [LongVideo-Reason](#longvideo-reason)
7.  [Examples](#examples)
8.  [How to contribute](#how-to-contribute)
9.  [Core Contributors](#core-contributors)
10. [Citation](#citation)
11. [Acknowledgement](#acknowledgement)

## Introduction

Long-RL is a complete framework that can scale Vision-Language Models (VLMs) reasoning on long videos, built on a deep integration of Reinforcement Learning. In addition to video-based models, Long-RL provides support for text, audio, and image-based models for various training and inference use cases.

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

*   **Open-ended reward**:
    *   Set `--worker.rollout.open_ended_reward=True` in the training script.
    *   Export your OpenAI API key with `export OPENAI_API_KEY=xxx`.
*   **Cached video embeddings**:
    *   Follow `verl/utils/cache_video_embeds_vila.py` to cache video embeddings in a local directory.
    *   Set `--data.cache_dir` and `--worker.actor.cached_embeds_dir` in the training script.
*   **Chunked gathering**:
    *   Set `--worker.rollout.num_chunk_seq` in the training script (e.g., 8/16/32).

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

For single node (within 8 GPUs), refer to training scripts in the `examples` directory. For example:

```bash
bash examples/new_supports/qwen2_5_vl_3b_video_grpo.sh $VIDEO_PATH
```

### Multi-nodes

For multi-node jobs, see the EasyR1 repo [here](https://github.com/hiyouga/EasyR1/tree/main?tab=readme-ov-file#how-to-run-70b-model-in-multi-node-environment).

Example `sbatch` script:

```bash
bash scripts/srun_multi_nodes.sh $TRAIN_SCRIPT $NNODES
```

Example:

```bash
bash scripts/srun_multi_nodes.sh examples/new_supports/qwen2_5_vl_3b_video_grpo.sh 2
```

### Merge Checkpoint in Hugging Face Format

This follows the EasyR1 repo instructions.

```bash
python3 scripts/model_merger.py --local_dir checkpoints/easy_r1/exp_name/global_step_1/actor
```

## LongVideo-Reason

Find detailed instructions on data generation and model evaluation on the `LongVideo-Reason` benchmark in the [`longvideo-reason`](longvideo-reason/) directory.

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

## How to contribute

*   Ensure Git is installed.
*   Create a [fork](https://github.com/NVlabs/Long-RL/fork) of the project.
*   Clone the repository locally.
*   Refer to the installation instructions.
*   Commit and push your changes.
*   Create a pull request.

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

*   [EasyR1](https://github.com/hiyouga/EasyR1): The codebase Long-RL is built upon.
*   [verl](https://github.com/volcengine/verl): The RL training framework.
*   [vllm](https://github.com/vllm-project/vllm): Used as the rollout engine.
*   [Flow-GRPO](https://github.com/yifan123/flow_grpo): For image/video generation RL guidance.