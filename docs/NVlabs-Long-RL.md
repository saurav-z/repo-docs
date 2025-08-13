<p align="center" width="100%">
<img src="assets/long-rl-logo.png" alt="Long-RL Logo" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

# Long-RL: Revolutionizing Long Video Reasoning with Reinforcement Learning

**Tackle long video understanding with Long-RL, a cutting-edge framework that scales Reinforcement Learning (RL) to process and reason over extensive video sequences!** ([Original Repo](https://github.com/NVlabs/Long-RL))

[![Paper](https://img.shields.io/badge/ArXiv-Paper-brown)](https://arxiv.org/abs/2507.07966)
[![Code](https://img.shields.io/badge/GitHub-Long%20RL-blue)](https://github.com/NVlabs/Long-RL)
[![Model](https://img.shields.io/badge/HuggingFace-Model-yellow)](https://huggingface.co/Efficient-Large-Model/LongVILA-R1-7B)
[![Video](https://img.shields.io/badge/YouTube-Video-red)](https://www.youtube.com/watch?v=ykbblK2jiEg)
[![Demo](https://img.shields.io/badge/Gradio-Demo-bron)](https://long-rl.hanlab.ai)

<div align="center">

[![Watch the video](assets/demo_video_first_frame.png)](https://www.youtube.com/watch?v=ykbblK2jiEg)

</div>

**Key Features:**

*   **Scalable Architecture:** Process and reason over **long video sequences** with ease.
*   **Advanced RL Training:** Leverages a two-stage training pipeline incorporating chain-of-thought supervised fine-tuning (CoT-SFT) and reinforcement learning (RL).
*   **Multi-Modal Support:** Trains on video, text, and audio data.
*   **High-Performance Models:** LongVILA-R1-7B achieves strong performance on video benchmarks.
*   **Efficient Training:** Utilizes Multi-modal Reinforcement Sequence Parallelism (MR-SP) for faster RL training, with up to **2.1x speedup**.
*   **Comprehensive Dataset:** Uses the LongVideo-Reason dataset, with 104k long video QA pairs.
*   **Versatile Model Support:** Supports VILA, Qwen series, image and video generation models (Stable Diffusion, Wan series).
*   **Flexible Frame Rate Control:** Supports processing **up to 8,192 video frames** per video with configurable FPS settings.
*   **Open-ended reward**: Support training for open-ended QAs, like setting `--worker.rollout.open_ended_reward=True` in training script.
*   **Cached video embeddings**: Support using cached video embeddings for video RL training, like setting `--data.cache_dir` and `--worker.actor.cached_embeds_dir` in training script.
*   **Chunked gathering**: Support chunked gathering for `all_gather_data_proto`, like setting `--worker.rollout.num_chunk_seq` in training script.

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
*   **[2025.7.30]** **LongVILA-R1-7B** supports processing up to **8,192** video frames per video, with configurable FPS settings. Please refer to its [usage instructions](#longvila-r1-model-usage).
*   **[2025.7.24]** Gradio demo released:  [Gradio Demo](https://long-rl.hanlab.ai) with LongVILA-R1-7B.
*   **[2025.7.24]** Model weights of **LongVILA-R1-7B** released on HuggingFace:  [Hugging Face Model](https://huggingface.co/Efficient-Large-Model/LongVILA-R1-7B). Achieves **65.1% / 71.1%** on VideoMME. Supports reasoning on both **multiple-choice** and **open-ended** questions, and can also switch to non-thinking mode.
*   **[2025.7.19]** Detailed instruction and scripts for LongVideo-Reason dataset data generation released in the [`longvideo-reason`](longvideo-reason/) directory.
*   **[2025.7.18]** New supported features, including *Open-ended reward*, *Cached video embeddings*, and *Chunked gathering* introduced in [Supported Features](#supported-features).
*   **[2025.7.10]** [Paper](https://arxiv.org/abs/2507.07966) and GitHub repo [Long-RL](https://github.com/NVlabs/Long-RL) released.

## Highlights
*   **Hour-level Long Video RL Training:**  Supports RL training on hour-long videos (3,600 frames - 256k tokens) with sequence parallel, on a single A100 node (8 GPUs). Example: `examples/new_supports/qwen2_5_vl_3b_video_1h.sh`
*   **Omni-Model RL:**  Supports RL training on omni models, which take text, video, and audio as inputs. Example: `examples/new_supports/qwen2_5_omni_3b_grpo.sh`
*   **Image/Video Generation RL:** Supports RL training on image/video generation models, like [Stable Diffusion](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium) and [Wan](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers) series models. Examples: `examples/new_supports/sd3_image_grpo.sh` and `examples/new_supports/wan_video_grpo.sh`.

## Introduction

Long-RL is designed to scale up vision-language models (VLMs) for long video reasoning using reinforcement learning. The framework leverages:

*   A large-scale dataset, **LongVideo-Reason**, featuring 104K long video QA pairs.
*   A two-stage training pipeline with Chain-of-Thought Supervised Fine-tuning (CoT-SFT) and Reinforcement Learning (RL).
*   **Multi-modal Reinforcement Sequence Parallelism (MR-SP)**, an infrastructure for long video RL, using sequence parallelism and a vLLM-based engine for efficiency.

**Supported Models:**

*   VILA series models (image/video) with SP support.
*   Qwen-VL series models (text, image, video, audio) with SP support.
*   Image and video diffusion model RL.

**Supported Algorithms:**

*   GRPO, DAPO & Reinforce with SP support.

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
    *   Export your openai API with `export OPENAI_API_KEY=xxx`.
*   **Cached video embeddings**:
    *   Follow `verl/utils/cache_video_embeds_vila.py` to cache video embeddings in a local directory.
    *   Set `--data.cache_dir` and `--worker.actor.cached_embeds_dir` in the training script.
*   **Chunked gathering**:
    *   Set `--worker.rollout.num_chunk_seq` in the training script. It can be 8/16/32.

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
Refer to the training scripts in the `examples` directory.  For instance:

```bash
bash examples/new_supports/qwen2_5_vl_3b_video_grpo.sh $VIDEO_PATH
```

### Multi-nodes

For multi-node jobs, refer to the EasyR1 repo for guidance. [Here](https://github.com/hiyouga/EasyR1/tree/main?tab=readme-ov-file#how-to-run-70b-model-in-multi-node-environment).

Example `sbatch` script:

```bash
bash scripts/srun_multi_nodes.sh $TRAIN_SCRIPT $NNODES
```

For instance:

```bash
bash scripts/srun_multi_nodes.sh examples/new_supports/qwen2_5_vl_3b_video_grpo.sh 2
```

### Merge Checkpoint in Hugging Face Format

Follow the EasyR1 repo instructions.

```bash
python3 scripts/model_merger.py --local_dir checkpoints/easy_r1/exp_name/global_step_1/actor
```

## LongVideo-Reason

Detailed instructions on data generation and model evaluation on the `LongVideo-Reason` benchmark are available in the [`longvideo-reason`](longvideo-reason/) directory.

## Examples

<div align="center">
<a href="https://drive.google.com/file/d/1QJ-ZsDrmYS8v1XU4eWfYu5oHuXeyGSdK/view?usp=share_link">Football Video</a>
</div>
<p align="center" width="100%">
<img src="assets/example-football.png" alt="Example Football Video" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

<div align="center">
<a href="https://drive.google.com/file/d/1U0N563a2s24o_NDie1VfWauxFuSu31wC/view?usp=share_link">Texas Hold’em Video</a>
</div>
<p align="center" width="100%">
<img src="assets/example-TexasHold.png" alt="Example Texas Hold'em Video" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

<div align="center">
<a href="https://drive.google.com/file/d/1rnF4I6-EBpqhzA0SnwyajpxbAhMezDCn/view?usp=share_link">Starcraft II Video</a>
</div>
<p align="center" width="100%">
<img src="assets/example-starcraft2.png" alt="Example Starcraft II Video" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

<div align="center">
<a href="https://drive.google.com/file/d/1lo1E_bXXnMmWnFRudaSUgxMNxetEDHP9/view?usp=share_link">Moving Cup Video</a>
</div>
<p align="center" width="100%">
<img src="assets/example-movingcup.png" alt="Example Moving Cup Video" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

## How to contribute
*   Install `git`.
*   Create a [fork](https://github.com/NVlabs/Long-RL/fork).
*   Clone the repository locally.
*   Follow the [Installation](#installation) instructions.
*   Commit and push your changes.
*   Submit a pull request.

## Core Contributors

[Yukang Chen](https://yukangchen.com/), [Wei Huang](https://aaron-weihuang.com/), [Shuai Yang](https://andysonys.github.io), [Qinghao Hu](https://tonyhao.xyz/), [Baifeng Shi](https://bfshi.github.io/), [Hanrong Ye](https://sites.google.com/site/yhrspace/home), [Ligeng Zhu](https://lzhu.me/).

Contributions are welcome and will be acknowledged.

## Citation

Please cite our work if it's helpful for your research.

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