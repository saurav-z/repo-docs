<p align="center" width="100%">
<img src="assets/long-rl-logo.png" alt="Long-RL Logo" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

# Long-RL: Revolutionizing Long Video Reasoning with RL

**Tackle long video reasoning with Long-RL, a cutting-edge framework leveraging Reinforcement Learning to achieve state-of-the-art results.**

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

**Explore Long-RL's capabilities through our [Gradio Demo](https://long-rl.hanlab.ai) and dive deeper with the [LongVILA-R1-7B model on Hugging Face](https://huggingface.co/Efficient-Large-Model/LongVILA-R1-7B)!**

## Key Features

*   **Scalable RL for Long Videos:** Train on hour-long videos (e.g., 3,600 frames) with sequence parallelism on a single A100 node (8 GPUs).
*   **Multi-Modal Support:** Train on models that take text, video, and audio as inputs.
*   **Image and Video Generation RL:** Supports RL training on image/video generation models (e.g., Stable Diffusion, Wan).
*   **High-Performance LongVILA-R1-7B Model:** Achieves state-of-the-art performance on video benchmarks, supports up to 8,192 video frames, and configurable FPS settings.
*   **Optimized Training Infrastructure:**  Utilizes Multi-modal Reinforcement Sequence Parallelism (MR-SP) for efficient long video RL training, achieving up to a 2.1x speedup.
*   **Open-ended reward:**  Supports training for open-ended QAs (non-multi-choices QAs).
*   **Cached video embeddings:** Supports using cached video embeddings for video RL training.
*   **Chunked gathering:**  Supports chunked gathering for `all_gather_data_proto`.

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

*   \[2025.07.30] **LongVILA-R1-7B** supports processing up to **8,192** video frames per video, with configurable FPS settings.  See [usage instructions](#longvila-r1-model-usage).
*   \[2025.07.24] Gradio demo released: [https://long-rl.hanlab.ai](https://long-rl.hanlab.ai)
*   \[2025.07.24] Model weights of **LongVILA-R1-7B** released on HuggingFace: [https://huggingface.co/Efficient-Large-Model/LongVILA-R1-7B](https://huggingface.co/Efficient-Large-Model/LongVILA-R1-7B).
*   \[2025.07.19] Data generation instructions for LongVideo-Reason dataset released in the [`longvideo-reason`](longvideo-reason/) directory.
*   \[2025.07.18] New features released: *Open-ended reward*, *Cached video embeddings*, and *Chunked gathering* (see [Supported Features](#supported-features)).
*   \[2025.07.10] Paper and GitHub repository [Long-RL](https://github.com/NVlabs/Long-RL) released.

## Highlights

1.  **Hour-Level Long Video RL Training:** Train on hour-long videos (3,600 frames - 256k tokens) with sequence parallelism on a single A100 node (8 GPUs). Example: `examples/new_supports/qwen2_5_vl_3b_video_1h.sh`
2.  **Omni-Model RL:** Supports RL training on models with text, video, and audio inputs. Example: `examples/new_supports/qwen2_5_omni_3b_grpo.sh`
3.  **Image/Video Generation RL:** Supports RL training on image/video generation models (e.g., Stable Diffusion, Wan). Examples: `examples/new_supports/sd3_image_grpo.sh` and `examples/new_supports/wan_video_grpo.sh`.

## Introduction

Long-RL is a full-stack framework designed to scale reasoning in vision-language models (VLMs) to long videos using reinforcement learning. This is achieved by integrating:

*   A large-scale dataset, **LongVideo-Reason**, with 104K long video QA pairs and high-quality reasoning annotations.
*   A two-stage training pipeline: chain-of-thought supervised fine-tuning (CoT-SFT) followed by reinforcement learning (RL).
*   **Multi-modal Reinforcement Sequence Parallelism (MR-SP)**, a training infrastructure optimized for long video RL, utilizing sequence parallelism and a vLLM-based engine with cached video embeddings.

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

Tested on `vllm==0.9.1`.  Requires remote code.

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

*   **Open-ended reward**: Support for training on open-ended QA tasks.  Requires setting `--worker.rollout.open_ended_reward=True` in the training script and setting the `OPENAI_API_KEY` environment variable.
*   **Cached video embeddings**: Use cached embeddings for faster video RL training. Cache embeddings using `verl/utils/cache_video_embeds_vila.py` and specify the cache directory using `--data.cache_dir` and `--worker.actor.cached_embeds_dir` in the training script.
*   **Chunked gathering**:  Support for chunked gathering for `all_gather_data_proto` to mitigate CPU OOM issues, especially for large batches or long videos. Configure with `--worker.rollout.num_chunk_seq` (e.g., 8/16/32).

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

Refer to example training scripts in the `examples` directory (e.g., `examples/new_supports/qwen2_5_vl_3b_video_grpo.sh $VIDEO_PATH`).

### Multi-Nodes

Follow the instructions in the EasyR1 repo ([here](https://github.com/hiyouga/EasyR1/tree/main?tab=readme-ov-file#how-to-run-70b-model-in-multi-node-environment)) for multi-node training.

Example `sbatch` script:

```bash
bash scripts/srun_multi_nodes.sh $TRAIN_SCRIPT $NNODES
```

Example:

```bash
bash scripts/srun_multi_nodes.sh examples/new_supports/qwen2_5_vl_3b_video_grpo.sh 2
```

### Merge Checkpoint in Hugging Face Format

Follow the instructions in the EasyR1 repo.

```bash
python3 scripts/model_merger.py --local_dir checkpoints/easy_r1/exp_name/global_step_1/actor
```

## LongVideo-Reason

Detailed instructions on the data generation process and evaluation methods for the `LongVideo-Reason` benchmark are available in the [`longvideo-reason`](longvideo-reason/) directory.

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
<img src="assets/example-TexasHold.png" alt="Texas Hold'em Video Example" style="width: 100%; min-width: 300px; display: block; margin: auto;">
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

## How to Contribute

*   Ensure you have git installed.
*   Create your own [fork](https://github.com/NVlabs/Long-RL/fork) of the project.
*   Clone the repository locally.
*   Follow the [Installation](#installation) instructions.
*   Make your changes and commit them.
*   Create a pull request.

## Core Contributors

[Yukang Chen](https://yukangchen.com/), [Wei Huang](https://aaron-weihuang.com/), [Shuai Yang](https://andysonys.github.io), [Qinghao Hu](https://tonyhao.xyz/), [Baifeng Shi](https://bfshi.github.io/), [Hanrong Ye](https://sites.google.com/site/yhrspace/home), [Ligeng Zhu](https://lzhu.me/).

All contributions are welcome.

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

*   [EasyR1](https://github.com/hiyouga/EasyR1): for providing a foundation.
*   [verl](https://github.com/volcengine/verl): for providing the RL training framework.
*   [vllm](https://github.com/vllm-project/vllm): for powering the rollout engine.
*   [Flow-GRPO](https://github.com/yifan123/flow_grpo): inspiration for the image/video generation RL.