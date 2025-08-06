<p align="center" width="100%">
<img src="assets/long-rl-logo.png" alt="Long-RL Logo" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

# Long-RL: Supercharge Long Video Reasoning with Reinforcement Learning

**Tired of limitations in long video analysis?** Long-RL is a cutting-edge framework designed to scale reinforcement learning (RL) to handle long video sequences, unlocking advanced reasoning capabilities in vision-language models (VLMs).  For more details, check out the original repo: [Long-RL on GitHub](https://github.com/NVlabs/Long-RL).

[![Paper](https://img.shields.io/badge/ArXiv-Paper-brown)](https://arxiv.org/abs/2507.07966)
[![Code](https://img.shields.io/badge/GitHub-Long%20RL-blue)](https://github.com/NVlabs/Long-RL)
[![Model](https://img.shields.io/badge/HuggingFace-Model-yellow)](https://huggingface.co/Efficient-Large-Model/LongVILA-R1-7B)
[![Video](https://img.shields.io/badge/YouTube-Video-red)](https://www.youtube.com/watch?v=ykbblK2jiEg)
[![Demo](https://img.shields.io/badge/Gradio-Demo-bron)](https://long-rl.hanlab.ai)

<div align="center">

[![Watch the video](assets/demo_video_first_frame.png)](https://www.youtube.com/watch?v=ykbblK2jiEg)

</div>

Key Features:

*   **Scalable RL for Long Videos:** Train and deploy RL models on hour-long video data.
*   **State-of-the-Art Performance:** Achieve impressive results on video benchmarks with the LongVILA-R1-7B model.
*   **Multi-Modal Support:** Train on various modalities including video, text, and audio.
*   **Flexible Architecture:** Supports VILA, Qwen, and image/video generation models.
*   **Efficient Training:** Utilize Multi-modal Reinforcement Sequence Parallelism (MR-SP) for fast RL training.
*   **Open-Ended Reward Support**: Train for open-ended QAs with flexible reward configurations.
*   **Cached Video Embeddings:** Boost training efficiency using pre-computed video embeddings.
*   **Chunked Gathering**: Mitigate memory issues and handle large batches and long video frames.

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
*   **\[2025.07.30]** LongVILA-R1-7B supports up to **8,192** video frames with configurable FPS. [usage instructions](#longvila-r1-model-usage)
*   **\[2025.07.24]** Gradio demo released: (https://long-rl.hanlab.ai) and LongVILA-R1-7B weights on HuggingFace (https://huggingface.co/Efficient-Large-Model/LongVILA-R1-7B). It achieves **65.1% / 71.1%** on VideoMME and supports multiple-choice and open-ended questions.
*   **\[2025.07.19]** Detailed instructions for LongVideo-Reason dataset generation are available in the [`longvideo-reason`](longvideo-reason/) directory.
*   **\[2025.07.18]** New features released: *Open-ended reward*, *Cached video embeddings*, and *Chunked gathering*. [Supported Features](#supported-features)
*   **\[2025.07.10]** Paper ([https://arxiv.org/abs/2507.07966](https://arxiv.org/abs/2507.07966)) and GitHub repo ([https://github.com/NVlabs/Long-RL](https://github.com/NVlabs/Long-RL)) released.

## Highlights
1.  **Hour-Level Long Video RL Training:** Train on hour-long videos (3,600 frames - 256k tokens) on a single A100 node.
2.  **Omni-Model RL Support:** Supports RL training with text, video, and audio inputs.
3.  **Image/Video Generation RL:**  Supports RL training on image/video generation models.

## Introduction

**Key Capabilities:**

*   **VILA Series Models:** Supports VILA models for image and video with Sequence Parallelism (SP) support.
*   **Qwen-VL Series Models:** Supports Qwen-VL models for text, image, video, and audio, with SP support.
*   **Image and Video Diffusion Model RL:** RL training with Image and Video diffusion models.

**Supported Algorithms:**

*   GRPO, DAPO & Reinforce algorithms with SP support.

## LongVILA-R1 Model Usage

[Include the example code provided in the original README here, formatted as code blocks.]

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

*   **Open-ended reward**: Support for training open-ended QA.
*   **Cached video embeddings**:  Accelerate training with cached video embeddings.
*   **Chunked gathering**: Optimize memory usage for large batches/long videos.

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

[Provide brief instructions, pointing to the examples directory.  Example:]

Refer to the training scripts in the `examples` directory for single-node training (within 8 GPUs). For example:
```bash
bash examples/new_supports/qwen2_5_vl_3b_video_grpo.sh $VIDEO_PATH
```

### Multi-Node

[Provide brief instructions for multi-node training, referencing EasyR1 repo.]

For multi-node training, see the instructions in the EasyR1 repo.

Provide `sbatch` script examples and explain their usage.

### Merge Checkpoint in Hugging Face Format

[Provide instructions for merging checkpoints using the provided script.]

## LongVideo-Reason

Detailed instructions on the data generation process and model evaluation are available in the [`longvideo-reason`](longvideo-reason/) directory.

## Examples

[Include example images and links to the example videos as provided in the original README]

## How to Contribute

*   Ensure you have Git installed.
*   Create a [fork](https://github.com/NVlabs/Long-RL/fork) of the project.
*   Clone the repository locally.
*   Follow the [Installation](#installation) instructions.
*   Commit and push your changes.
*   Create a pull request.

## Core Contributors

[List the core contributors here]

We welcome all contributions and will acknowledge all contributors.

## Citation

[Include the provided bibtex citations]

## Acknowledgement

*   [EasyR1](https://github.com/hiyouga/EasyR1)
*   [verl](https://github.com/volcengine/verl)
*   [vllm](https://github.com/vllm-project/vllm)
*   [Flow-GRPO](https://github.com/yifan123/flow_grpo)