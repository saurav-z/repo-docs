<p align="center" width="100%">
<img src="assets/long-rl-logo.png" alt="Long-RL Logo" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

# Long-RL: Revolutionizing Long Video Reasoning with Reinforcement Learning

**Long-RL leverages reinforcement learning to enable advanced reasoning in vision-language models (VLMs) for long-form videos, achieving state-of-the-art results.** [Explore the code on GitHub](https://github.com/NVlabs/Long-RL)!

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

*   **Scalable RL for Long Videos:** Train on hour-long videos (e.g., 3,600 frames) on a single A100 node with sequence parallelism.
*   **Multi-Modal Support:** Trains on models that take text, video, and audio as input.
*   **Image and Video Generation RL:** Support for RL training on image/video generation models like Stable Diffusion and Wan series models.
*   **High Performance:** LongVILA-R1-7B achieves impressive accuracy on video benchmarks, demonstrating significant improvements over existing models.
*   **Efficient Training Infrastructure:** Utilizes Multi-modal Reinforcement Sequence Parallelism (MR-SP) for efficient rollout and prefilling, achieving up to 2.1x speedup in RL training.
*   **Open-Ended Reward Support:** Train for open-ended QA.
*   **Cached Video Embeddings:** Improve training speed with cached video embeddings.
*   **Chunked Gathering:** Reduce memory consumption during training for large batches or long videos.

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

*   **[2025.07.30]** LongVILA-R1-7B now supports processing up to **8,192** video frames per video, with configurable FPS settings. See [usage instructions](#longvila-r1-model-usage).
*   **[2025.07.24]** Gradio demo released: [https://long-rl.hanlab.ai](https://long-rl.hanlab.ai).
*   **[2025.07.24]** Model weights of **LongVILA-R1-7B** released on Hugging Face: [https://huggingface.co/Efficient-Large-Model/LongVILA-R1-7B](https://huggingface.co/Efficient-Large-Model/LongVILA-R1-7B). It achieves **65.1% / 71.1%** on VideoMME and supports reasoning on multiple-choice and open-ended questions.
*   **[2025.07.19]** Detailed instructions and scripts for generating the LongVideo-Reason dataset released in the [`longvideo-reason`](longvideo-reason/) directory.
*   **[2025.07.18]** New features released: *Open-ended reward*, *Cached video embeddings*, and *Chunked gathering*. See [Supported Features](#supported-features).
*   **[2025.07.10]** Paper ([https://arxiv.org/abs/2507.07966](https://arxiv.org/abs/2507.07966)) and GitHub repo [Long-RL](https://github.com/NVlabs/Long-RL) released.

## Highlights

1.  **Hour-level Long Video RL Training:** Support for RL training on hour-long videos (3,600 frames - 256k tokens) with sequence parallel, on a single A100 node (8 GPUs). Example: `examples/new_supports/qwen2_5_vl_3b_video_1h.sh`
2.  **Omni-Model RL:** RL training support for omni models that take text, video, and audio inputs. Example: `examples/new_supports/qwen2_5_omni_3b_grpo.sh`
3.  **Image/Video Generation RL:** RL training support for image/video generation models like [Stable Diffusion](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium) and [Wan](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers) series models. Examples: `examples/new_supports/sd3_image_grpo.sh` and `examples/new_supports/wan_video_grpo.sh`.

## Introduction

Long-RL is a framework designed to scale up reasoning in vision-language models (VLMs) to handle long videos, leveraging reinforcement learning (RL). It tackles the challenges of long video reasoning with:

*   **LongVideo-Reason Dataset:** A large-scale dataset with 104K long video QA pairs with high-quality reasoning annotations across diverse domains.
*   **Two-Stage Training Pipeline:** Extends VLMs with chain-of-thought supervised fine-tuning (CoT-SFT) and reinforcement learning (RL).
*   **Multi-modal Reinforcement Sequence Parallelism (MR-SP):**  A training infrastructure for long video RL, incorporating sequence parallelism and a vLLM-based engine tailored for long videos, using cached video embeddings for efficient rollout and prefilling.

**Supported Models:**

*   VILA series models (image and video) with SP support.
*   Qwen-VL series models (text, image, video, and audio) with SP support.
*   Image and video diffusion model RL.

**Supported Algorithms:**

*   GRPO, DAPO & Reinforce, with SP support.

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

Requires installation and modification of the original code; follow the steps in the original readme for this.

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

use_thinking = True  # Switching between thinking and non-thinking modes
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

*   **Open-ended reward**: Support for training on open-ended QAs. Instructions are provided.
*   **Cached video embeddings**:  Support for using cached video embeddings for faster training. Instructions are provided.
*   **Chunked gathering**: Support for chunked gathering to address CPU OOM issues. Instructions are provided.

## Installation

```bash
git clone https://github.com/NVlabs/Long-RL.git
cd Long-RL
pip install -e .
```

For Qwen-Omni models, also run:

```bash
bash vllm_replace.sh
```

## Training

### Single Node

Refer to training scripts in the `examples` directory:

```bash
bash examples/new_supports/qwen2_5_vl_3b_video_grpo.sh $VIDEO_PATH
```

### Multi-nodes

Instructions and scripts for running training on multiple nodes are provided in the EasyR1 repo (linked in the original README). Example sbatch script:

```bash
bash scripts/srun_multi_nodes.sh $TRAIN_SCRIPT $NNODES
```

### Merge Checkpoint in Hugging Face Format

```bash
python3 scripts/model_merger.py --local_dir checkpoints/easy_r1/exp_name/global_step_1/actor
```

## LongVideo-Reason

Detailed instructions on the data generation process and evaluation are available in the [`longvideo-reason`](longvideo-reason/) directory.

## Examples

<div align="center">
    <a href="https://drive.google.com/file/d/1QJ-ZsDrmYS8v1XU4eWfYu5oHuXeyGSdK/view?usp=share_link">Football Video</a>
    <img src="assets/example-football.png" alt="Football Example" style="width: 80%; min-width: 300px; display: block; margin: auto;">
</div>

<div align="center">
    <a href="https://drive.google.com/file/d/1U0N563a2s24o_NDie1VfWauxFuSu31wC/view?usp=share_link">Texas Hold’em Video</a>
    <img src="assets/example-TexasHold.png" alt="Texas Hold’em Example" style="width: 80%; min-width: 300px; display: block; margin: auto;">
</div>

<div align="center">
    <a href="https://drive.google.com/file/d/1rnF4I6-EBpqhzA0SnwyajpxbAhMezDCn/view?usp=share_link">Starcraft II Video</a>
    <img src="assets/example-starcraft2.png" alt="Starcraft II Example" style="width: 80%; min-width: 300px; display: block; margin: auto;">
</div>

<div align="center">
    <a href="https://drive.google.com/file/d/1lo1E_bXXnMmWnFRudaSUgxMNxetEDHP9/view?usp=share_link">Moving Cup Video</a>
    <img src="assets/example-movingcup.png" alt="Moving Cup Example" style="width: 80%; min-width: 300px; display: block; margin: auto;">
</div>

## How to Contribute

*   Install git.
*   Create a fork of the project.
*   Clone the repository.
*   Read the `Installation` sections.
*   Commit and push changes.
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

*   [EasyR1](https://github.com/hiyouga/EasyR1)
*   [verl](https://github.com/volcengine/verl)
*   [vllm](https://github.com/vllm-project/vllm)
*   [Flow-GRPO](https://github.com/yifan123/flow_grpo)