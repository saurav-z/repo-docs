<p align="center" width="100%">
<img src="assets/long-rl-logo.png" alt="Long-RL Logo" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

# Long-RL: Revolutionizing Long Video Reasoning with Reinforcement Learning

**Scale your vision-language models to analyze long videos with Long-RL, a cutting-edge framework leveraging reinforcement learning!** Dive into the details on the [original repo](https://github.com/NVlabs/Long-RL).

[![Paper](https://img.shields.io/badge/ArXiv-Paper-brown)](https://arxiv.org/abs/2507.07966)
[![Code](https://img.shields.io/badge/GitHub-Long%20RL-blue)](https://github.com/NVlabs/Long-RL)
[![Model](https://img.shields.io/badge/HuggingFace-Model-yellow)](https://huggingface.co/Efficient-Large-Model/LongVILA-R1-7B)
[![Video](https://img.shields.io/badge/YouTube-Video-red)](https://www.youtube.com/watch?v=ykbblK2jiEg)
[![Demo](https://img.shields.io/badge/Gradio-Demo-bron)](https://long-rl.hanlab.ai)

<div align="center">

[![Watch the video](assets/demo_video_first_frame.png)](https://www.youtube.com/watch?v=ykbblK2jiEg)

</div>

## Key Features:

*   **Long Video Support:** Process up to 8,192 video frames with configurable FPS.
*   **High Performance:**  LongVILA-R1-7B achieves impressive accuracy on video benchmarks.
*   **Efficient Training:** Utilize Multi-modal Reinforcement Sequence Parallelism (MR-SP) for up to 2.1x speedup in RL training.
*   **Omni-Modal RL:** Supports RL training with text, video, and audio inputs.
*   **Image/Video Generation RL:** Supports RL training on image/video generation models like Stable Diffusion and Wan series.
*   **Open-ended Reward Support:** Supports open-ended QA, with setting `--worker.rollout.open_ended_reward=True` and `OPENAI_API_KEY`.
*   **Cached Video Embeddings:** Supports cached video embeddings for faster training with `--data.cache_dir` and `--worker.actor.cached_embeds_dir`.
*   **Chunked Gathering:** Addresses potential CPU OOM issues with `--worker.rollout.num_chunk_seq`.

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

*   **[2025.7.30]** LongVILA-R1-7B now supports processing up to **8,192** video frames with configurable FPS.
*   **[2025.7.24]** Gradio demo released: [https://long-rl.hanlab.ai](https://long-rl.hanlab.ai) and model weights for **LongVILA-R1-7B** on HuggingFace: [https://huggingface.co/Efficient-Large-Model/LongVILA-R1-7B](https://huggingface.co/Efficient-Large-Model/LongVILA-R1-7B).
*   **[2025.7.24]** LongVILA-R1-7B achieves **65.1% / 71.1%** on VideoMME.
*   **[2025.7.19]** Data generation instructions and scripts for LongVideo-Reason released in the [`longvideo-reason`](longvideo-reason/) directory.
*   **[2025.7.18]** New features: Open-ended reward, Cached video embeddings, and Chunked gathering added.
*   **[2025.7.10]** Paper and GitHub repo [Long-RL](https://github.com/NVlabs/Long-RL) released.

## Highlights

1.  **Hour-Level Long Video RL Training:** Supports RL training on hour-long videos (3,600 frames - 256k tokens) on a single A100 node (8 GPUs) using sequence parallelism (`examples/new_supports/qwen2_5_vl_3b_video_1h.sh`).
2.  **Omni-Model RL:** Supports RL training on models with text, video, and audio inputs (`examples/new_supports/qwen2_5_omni_3b_grpo.sh`).
3.  **Image/Video Generation RL:** Supports RL training for image/video generation models like Stable Diffusion and Wan series (`examples/new_supports/sd3_image_grpo.sh` and `examples/new_supports/wan_video_grpo.sh`).

## Introduction

Long-RL is a comprehensive framework designed to scale reinforcement learning for vision-language models (VLMs) to handle long video sequences. It utilizes a two-stage training pipeline: Chain-of-Thought Supervised Fine-tuning (CoT-SFT) followed by Reinforcement Learning (RL).  The framework incorporates several critical components: a large-scale dataset called LongVideo-Reason, a two-stage training pipeline and a training infrastructure optimized for long video RL (MR-SP). This allows for high performance on multiple video benchmarks while addressing the unique challenges of long-form video analysis.

**Supported Models:**

*   VILA series models (image and video, with SP support)
*   Qwen-VL series models (text, image, video, and audio, with SP support)
*   Image and video diffusion model RL

**Supported Algorithms:**

*   GRPO, DAPO & Reinforce, with SP support

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

*   **Open-ended reward**: Support for open-ended QAs (non-multi-choices QAs). Set `--worker.rollout.open_ended_reward=True` and export your openai API with `export OPENAI_API_KEY=xxx`.
*   **Cached video embeddings**: Use cached video embeddings for faster video RL training.  Follow `verl/utils/cache_video_embeds_vila.py` to cache video embeddings and set `--data.cache_dir` and `--worker.actor.cached_embeds_dir`.
*   **Chunked gathering**: Supports chunked gathering for `all_gather_data_proto`. Set `--worker.rollout.num_chunk_seq` in the training script.

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

Refer to the training scripts in the `examples` directory for single-node training (within 8 GPUs).  For example:

```bash
bash examples/new_supports/qwen2_5_vl_3b_video_grpo.sh $VIDEO_PATH
```

### Multi-Nodes

Use the EasyR1 repo's methods for multi-node jobs.  Refer to the instructions [here](https://github.com/hiyouga/EasyR1/tree/main?tab=readme-ov-file#how-to-run-70b-model-in-multi-node-environment).

Example `sbatch` scripts:

```bash
bash scripts/srun_multi_nodes.sh $TRAIN_SCRIPT $NNODES
```

Example:
```bash
bash scripts/srun_multi_nodes.sh examples/new_supports/qwen2_5_vl_3b_video_grpo.sh 2
```

### Merge Checkpoint in Hugging Face Format

Following EasyR1:

```bash
python3 scripts/model_merger.py --local_dir checkpoints/easy_r1/exp_name/global_step_1/actor
```

## LongVideo-Reason

Find detailed instructions on data generation and model evaluation for the `LongVideo-Reason` benchmark in the [`longvideo-reason`](longvideo-reason/) directory.

## Examples

[Football Video](https://drive.google.com/file/d/1QJ-ZsDrmYS8v1XU4eWfYu5oHuXeyGSdK/view?usp=share_link)

<p align="center" width="100%">
<img src="assets/example-football.png" alt="Football Example" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

[Texas Hold’em Video](https://drive.google.com/file/d/1U0N563a2s24o_NDie1VfWauxFuSu31wC/view?usp=share_link)

<p align="center" width="100%">
<img src="assets/example-TexasHold.png" alt="Texas Hold'em Example" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

[Starcraft II Video](https://drive.google.com/file/d/1rnF4I6-EBpqhzA0SnwyajpxbAhMezDCn/view?usp=share_link)

<p align="center" width="100%">
<img src="assets/example-starcraft2.png" alt="Starcraft II Example" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

[Moving Cup Video](https://drive.google.com/file/d/1lo1E_bXXnMmWnFRudaSUgxMNxetEDHP9/view?usp=share_link)

<p align="center" width="100%">
<img src="assets/example-movingcup.png" alt="Moving Cup Example" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

## How to Contribute

*   Ensure git is installed.
*   Create a [fork](https://github.com/NVlabs/Long-RL/fork) of the project.
*   Clone the repository locally.
*   Follow the installation instructions.
*   Commit, push, and create a pull request.

## Core Contributors

[Yukang Chen](https://yukangchen.com/), [Wei Huang](https://aaron-weihuang.com/), [Shuai Yang](https://andysonys.github.io), [Qinghao Hu](https://tonyhao.xyz/), [Baifeng Shi](https://bfshi.github.io/), [Hanrong Ye](https://sites.google.com/site/yhrspace/home), [Ligeng Zhu](https://lzhu.me/).

We welcome all contributions.

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

*   [EasyR1](https://github.com/hiyouga/EasyR1): The foundation of this codebase.
*   [verl](https://github.com/volcengine/verl): The RL training framework utilized.
*   [vllm](https://github.com/vllm-project/vllm): Used for the rollout engine.
*   [Flow-GRPO](https://github.com/yifan123/flow_grpo): Inspiration for image/video generation RL implementation.