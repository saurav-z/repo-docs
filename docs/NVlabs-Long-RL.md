<p align="center" width="100%">
<img src="assets/long-rl-logo.png" alt="Long-RL Logo" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

# Long-RL: Revolutionizing Long Video Reasoning with Reinforcement Learning

**Tackle long video reasoning challenges with Long-RL, a full-stack framework that leverages reinforcement learning for state-of-the-art performance.**  [Explore the Code on GitHub](https://github.com/NVlabs/Long-RL)

[![Paper](https://img.shields.io/badge/ArXiv-Paper-brown)](https://arxiv.org/abs/2507.07966)
[![Code](https://img.shields.io/badge/GitHub-Long%20RL-blue)](https://github.com/NVlabs/Long-RL)
[![Model](https://img.shields.io/badge/HuggingFace-Model-yellow)](https://huggingface.co/Efficient-Large-Model/LongVILA-R1-7B)
[![Video](https://img.shields.io/badge/YouTube-Video-red)](https://www.youtube.com/watch?v=ykbblK2jiEg)
[![Demo](https://img.shields.io/badge/Gradio-Demo-bron)](https://long-rl.hanlab.ai)

<div align="center">

[![Watch the video](assets/demo_video_first_frame.png)](https://www.youtube.com/watch?v=ykbblK2jiEg)

</div>

## Key Features

*   **Enhanced Long Video Reasoning**: Achieves state-of-the-art results on video benchmarks.
*   **Multi-modal Support**: Trains on video, text, and audio inputs.
*   **Efficient Training Infrastructure**:  MR-SP (Multi-modal Reinforcement Sequence Parallelism) for faster RL training on long videos.
*   **Large-Scale Dataset**: Leverages the LongVideo-Reason dataset with 104K high-quality video QA pairs.
*   **Extended Context Window**: Supports processing up to 8,192 video frames.
*   **Flexible FPS Control**: Configurable frame rate settings for diverse video content.
*   **Omni-Model RL**: Supports RL training on models that take text, video, and audio for inputs.
*   **Image/Video Generation RL**: Supports RL training on image/video generation models such as Stable Diffusion and Wan series models.

## News

*   **[2025.7.30]** LongVILA-R1-7B now supports up to **8,192** video frames with configurable FPS.
*   **[2025.7.24]** Gradio demo and model weights for **LongVILA-R1-7B** are released on Hugging Face, with enhanced reasoning capabilities.
*   **[2025.7.19]** Detailed instructions and scripts for LongVideo-Reason dataset generation available.
*   **[2025.7.18]** New features released: Open-ended reward, Cached video embeddings, and Chunked gathering.
*   **[2025.7.10]** Paper and GitHub repo released.

## Highlights

*   **Hour-level Long Video RL Training**: Supports RL training on hour-long videos (3,600 frames - 256k tokens) using sequence parallelism on a single A100 node (8 GPUs).
*   **Omni-Model RL**: Supports RL training on omni models, which take text, video, and audio as inputs.
*   **Image/Video Generation RL**: Supports RL training on image/video generation models, like [Stable Diffusion](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium) and [Wan](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers) series models.

## Introduction

Long-RL is designed to scale Reinforcement Learning (RL) for Vision-Language Models (VLMs) to handle long videos. It's built on three core components:

*   **LongVideo-Reason**:  A large-scale dataset (104K QA pairs).
*   **Two-stage Training Pipeline**: Extends VLMs using Chain-of-Thought Supervised Fine-tuning (CoT-SFT) and RL.
*   **Multi-modal Reinforcement Sequence Parallelism (MR-SP)**: Training infrastructure for long-video RL, including sequence parallelism and vLLM-based engine.

**Support models**:
-   VILA series models on image and video, with SP support
-   Qwen-VL series models on text, image, video, and audio, with SP support
-   Image and video diffusion model RL
    
**Support algorithms**:
-   In addition to GRPO, DAPO & Reinforce supported, with SP support

## LongVILA-R1 Model Usage

**General Inference (Example)**

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

**vLLM Engine Integration (Example)**

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

*   **Open-ended reward**: Supports training for open-ended QA. Requires setting `--worker.rollout.open_ended_reward=True` and exporting your OpenAI API key.
*   **Cached video embeddings**: Supports using cached embeddings. Set `--data.cache_dir` and `--worker.actor.cached_embeds_dir` for faster training.
*   **Chunked gathering**: Supports chunked gathering to address CPU OOM issues. Use `--worker.rollout.num_chunk_seq`.

## Installation

```bash
git clone https://github.com/NVlabs/Long-RL.git
cd Long-RL
pip install -e .
```

For Qwen-Omni models:

```bash
bash vllm_replace.sh
```

## Training

### Single Node

See the `examples` directory for training scripts.  Example:

```bash
bash examples/new_supports/qwen2_5_vl_3b_video_grpo.sh $VIDEO_PATH
```

### Multi-Node

Use `sbatch` scripts. Example:

```bash
bash scripts/srun_multi_nodes.sh examples/new_supports/qwen2_5_vl_3b_video_grpo.sh 2
```

### Merge Checkpoint in Hugging Face Format

```bash
python3 scripts/model_merger.py --local_dir checkpoints/easy_r1/exp_name/global_step_1/actor
```

## LongVideo-Reason

Detailed instructions for data generation and evaluation are in the [`longvideo-reason`](longvideo-reason/) directory.

## Examples

*(Include the example images and links from the original README here)*

## How to Contribute

*   Fork the repository.
*   Clone your fork.
*   Install dependencies (see above).
*   Make your changes, commit, and push.
*   Create a pull request.

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

*   [EasyR1](https://github.com/hiyouga/EasyR1): the codebase we built upon. Thanks for their wonderful work.
*   [verl](https://github.com/volcengine/verl): the RL training framework we built upon.
*   [vllm](https://github.com/vllm-project/vllm): we built upon vllm for the rollout engine.
*   [Flow-GRPO](https://github.com/yifan123/flow_grpo): we refer to the Flow-GRPO for the image/video generation RL part.