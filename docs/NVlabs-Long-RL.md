<p align="center" width="100%">
<img src="assets/long-rl-logo.png" alt="Long-RL" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

# Long-RL: Revolutionizing Long Video Reasoning with Reinforcement Learning

**Tackle the challenge of long video reasoning with Long-RL, a cutting-edge framework that scales vision-language models using reinforcement learning.**  Read the [paper](https://arxiv.org/abs/2507.07966) and explore the code on [GitHub](https://github.com/NVlabs/Long-RL) or try the [demo](https://long-rl.hanlab.ai).

[![Paper](https://img.shields.io/badge/ArXiv-Paper-brown)](https://arxiv.org/abs/2507.07966)
[![Code](https://img.shields.io/badge/GitHub-Long%20RL-blue)](https://github.com/NVlabs/Long-RL)
[![Model](https://img.shields.io/badge/HuggingFace-Model-yellow)](https://huggingface.co/Efficient-Large-Model/LongVILA-R1-7B)
[![Video](https://img.shields.io/badge/YouTube-Video-red)](https://www.youtube.com/watch?v=ykbblK2jiEg)
[![Demo](https://img.shields.io/badge/Gradio-Demo-bron)](https://long-rl.hanlab.ai)

<div align="center">

[![Watch the video](assets/demo_video_first_frame.png)](https://www.youtube.com/watch?v=ykbblK2jiEg)

</div>

## Key Features

*   **Scalability:** Supports RL training on hour-long videos (3,600 frames - 256k tokens) with sequence parallelism, even on a single A100 node (8 GPUs).
*   **Multi-Modal RL:** Enables RL training on various modalities, including text, video, and audio, supporting omni-models.
*   **Generative Model RL:**  Supports RL training on image and video generation models such as Stable Diffusion and Wan series models.
*   **Open-Ended Reward Support:** Train on open-ended QA scenarios, using external APIs to measure the quality of your model's answers.
*   **Efficient Training:** Includes features like cached video embeddings and chunked gathering to accelerate training, especially for long videos and large batch sizes.
*   **High Performance:** LongVILA-R1-7B achieves strong performance on video benchmarks, consistently outperforming previous models.

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

*   **\[2025.7.30]** **LongVILA-R1-7B** now supports processing up to **8,192** video frames per video, with configurable FPS settings. See [usage instructions](#longvila-r1-model-usage).
*   **\[2025.7.24]** Gradio demo released: [https://long-rl.hanlab.ai](https://long-rl.hanlab.ai) (powered by LongVILA-R1-7B).
*   **\[2025.7.24]**  Model weights of **LongVILA-R1-7B** released on HuggingFace: [https://huggingface.co/Efficient-Large-Model/LongVILA-R1-7B](https://huggingface.co/Efficient-Large-Model/LongVILA-R1-7B).  Achieves **65.1% / 71.1%** on VideoMME. Supports multiple-choice and open-ended questions, and non-thinking mode.
*   **\[2025.7.19]** Detailed instructions and scripts for data generation of the LongVideo-Reason dataset are available in the [`longvideo-reason`](longvideo-reason/) directory.
*   **\[2025.7.18]** New features released, including *Open-ended reward*, *Cached video embeddings*, and *Chunked gathering* (see [Supported Features](#supported-features)).
*   **\[2025.7.10]** Paper ([https://arxiv.org/abs/2507.07966](https://arxiv.org/abs/2507.07966)) and this GitHub repo [Long-RL](https://github.com/NVlabs/Long-RL) released.

## Highlights

1.  **Hour-level long video RL training on a single node:** Train on hour-long videos (3,600 frames) with sequence parallelism using a single A100 node (8 GPUs). See `examples/new_supports/qwen2_5_vl_3b_video_1h.sh`.
2.  **Omni-model RL:** Train on omni-models that take text, video, and audio as input. See `examples/new_supports/qwen2_5_omni_3b_grpo.sh`.
3.  **Image/video generation RL:** Train on image/video generation models. Examples:  `examples/new_supports/sd3_image_grpo.sh` and `examples/new_supports/wan_video_grpo.sh`.

## Introduction

Long-RL is a comprehensive framework designed to scale vision-language models (VLMs) for reasoning on long videos. It leverages reinforcement learning (RL) to overcome the challenges of long-context video understanding.

**Supported Models:**

*   VILA series models on image and video (with SP support)
*   Qwen-VL series models on text, image, video, and audio (with SP support)
*   Image and video diffusion model RL

**Supported Algorithms:**

*   GRPO, DAPO & Reinforce, all with Sequence Parallelism (SP) support.

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

*   **Open-ended reward**:  Train for open-ended question answering.  Set `--worker.rollout.open_ended_reward=True` in the training script and export your OpenAI API key (`export OPENAI_API_KEY=xxx`).
*   **Cached video embeddings**: Accelerate training by using pre-computed video embeddings. Follow `verl/utils/cache_video_embeds_vila.py` to cache embeddings, then set `--data.cache_dir` and `--worker.actor.cached_embeds_dir` in the training script.
*   **Chunked gathering**: Mitigate potential CPU OOM errors by setting `--worker.rollout.num_chunk_seq` in the training script (e.g., 8/16/32).  Larger values use less memory but may take longer.

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

Refer to the training scripts in the `examples` directory. For instance:

```bash
bash examples/new_supports/qwen2_5_vl_3b_video_grpo.sh $VIDEO_PATH
```

### Multi-Nodes

Follow instructions from the EasyR1 repo for multi-node setups.  Example `sbatch` scripts:

```bash
bash scripts/srun_multi_nodes.sh $TRAIN_SCRIPT $NNODES
```

Example:

```bash
bash scripts/srun_multi_nodes.sh examples/new_supports/qwen2_5_vl_3b_video_grpo.sh 2
```

### Merge Checkpoint in Hugging Face Format

Use the `model_merger.py` script as described in the EasyR1 repo:

```bash
python3 scripts/model_merger.py --local_dir checkpoints/easy_r1/exp_name/global_step_1/actor
```

## LongVideo-Reason

Detailed instructions on data generation and model evaluation are available in the [`longvideo-reason`](longvideo-reason/) directory.

## Examples

[Example 1 (Football Video)](https://drive.google.com/file/d/1QJ-ZsDrmYS8v1XU4eWfYu5oHuXeyGSdK/view?usp=share_link)

<p align="center" width="100%">
<img src="assets/example-football.png" alt="Football Example" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

[Example 2 (Texas Hold’em Video)](https://drive.google.com/file/d/1U0N563a2s24o_NDie1VfWauxFuSu31wC/view?usp=share_link)

<p align="center" width="100%">
<img src="assets/example-TexasHold.png" alt="Texas Hold'em Example" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

[Example 3 (Starcraft II Video)](https://drive.google.com/file/d/1rnF4I6-EBpqhzA0SnwyajpxbAhMezDCn/view?usp=share_link)

<p align="center" width="100%">
<img src="assets/example-starcraft2.png" alt="Starcraft II Example" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

[Example 4 (Moving Cup Video)](https://drive.google.com/file/d/1lo1E_bXXnMmWnFRudaSUgxMNxetEDHP9/view?usp=share_link)

<p align="center" width="100%">
<img src="assets/example-movingcup.png" alt="Moving Cup Example" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

## How to Contribute

*   Install Git.
*   Fork the project.
*   Clone your fork locally.
*   Follow the [Installation](#installation) instructions.
*   Make your changes, commit, and push.
*   Submit a pull request.

## Core Contributors

[Yukang Chen](https://yukangchen.com/), [Wei Huang](https://aaron-weihuang.com/), [Shuai Yang](https://andysonys.github.io), [Qinghao Hu](https://tonyhao.xyz/), [Baifeng Shi](https://bfshi.github.io/), [Hanrong Ye](https://sites.google.com/site/yhrspace/home), [Ligeng Zhu](https://lzhu.me/).

We welcome and acknowledge all contributions.

## Citation

Please cite our paper and framework if they are helpful for your research.

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

*   [EasyR1](https://github.com/hiyouga/EasyR1): Base codebase.
*   [verl](https://github.com/volcengine/verl): RL training framework.
*   [vllm](https://github.com/vllm-project/vllm): Rollout engine.
*   [Flow-GRPO](https://github.com/yifan123/flow_grpo): Image/video generation RL inspiration.