<p align="center" width="100%">
<img src="assets/long-rl-logo.png" alt="Long-RL Logo" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

# Long-RL: Revolutionizing Long Video Reasoning with Reinforcement Learning

**Scale your vision-language models to conquer long video reasoning tasks with Long-RL, a comprehensive framework that leverages reinforcement learning for superior performance.**

[![Paper](https://img.shields.io/badge/ArXiv-Paper-brown)](https://arxiv.org/abs/2507.07966)
[![Code](https://img.shields.io/badge/GitHub-Long%20RL-blue)](https://github.com/NVlabs/Long-RL)
[![Model](https://img.shields.io/badge/HuggingFace-Model-yellow)](https://huggingface.co/Efficient-Large-Model/LongVILA-R1-7B)
[![Video](https://img.shields.io/badge/YouTube-Video-red)](https://www.youtube.com/watch?v=ykbblK2jiEg)
[![Demo](https://img.shields.io/badge/Gradio-Demo-bron)](https://long-rl.hanlab.ai)

<div align="center">

[![Watch the video](assets/demo_video_first_frame.png)](https://www.youtube.com/watch?v=ykbblK2jiEg)

</div>

**For detailed information, visit the original repository: [Long-RL GitHub](https://github.com/NVlabs/Long-RL)**

## Key Features

*   **Enhanced Performance:** Achieve state-of-the-art results on video reasoning benchmarks, including VideoMME and ActivityNet-QA.  LongVILA-R1-7B achieves **65.0% / 70.7%** on VideoMME.
*   **Long Video Support:**  Train on and reason about hour-long videos (e.g., 3,600 frames / ~256k tokens) with efficient sequence parallelism.
*   **Multi-Modal Training:**  Supports reinforcement learning training across various modalities, including video, text, and audio.
*   **Model Agnostic:** Works with a wide range of models, including VILA, Qwen series, and image/video generation models (Stable Diffusion, Wan series).
*   **Optimized Infrastructure:** Utilizes Multi-modal Reinforcement Sequence Parallelism (MR-SP) for up to 2.1x speedup in long video RL training.
*   **Flexible Deployment**: Supports both thinking and non-thinking modes.
*   **Comprehensive Dataset**:  Leverages the LongVideo-Reason dataset, a large-scale dataset with 104K long video QA pairs, enhancing reasoning capabilities across diverse domains.
*   **Efficient Infrastructure:** The Multi-modal Reinforcement Sequence Parallelism (MR-SP) system supports hour-long video RL training on a single A100 node (8 GPUs) and achieves up to a 2.1x speedup.

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

*   \[2025.07.27] **LongVILA-R1-7B** supports processing up to **2,048** video frames per video, with configurable FPS settings. Please refer to its [usage instructions](#longvila-r1-model-usage).
*   \[2025.07.24] Gradio demo released: [https://long-rl.hanlab.ai](https://long-rl.hanlab.ai)
*   \[2025.07.24] **LongVILA-R1-7B** model weights released on Hugging Face: [https://huggingface.co/Efficient-Large-Model/LongVILA-R1-7B](https://huggingface.co/Efficient-Large-Model/LongVILA-R1-7B).
*   \[2025.07.19] Instructions and scripts for LongVideo-Reason dataset generation released in [`longvideo-reason/`](longvideo-reason/).
*   \[2025.07.18] New features: Open-ended reward, Cached video embeddings, and Chunked gathering (see [Supported Features](#supported-features)).
*   \[2025.07.10] Paper released on arXiv: [https://arxiv.org/abs/2507.07966](https://arxiv.org/abs/2507.07966) and GitHub repo [Long-RL](https://github.com/NVlabs/Long-RL).

## Highlights

1.  **Single-Node, Hour-Level Training:** Train on hour-long videos (3,600 frames, ~256k tokens) using sequence parallelism on a single A100 node (8 GPUs). Example: `examples/new_supports/qwen2_5_vl_3b_video_1h.sh`
2.  **Omni-Model RL Support:** Perform RL training on models that accept text, video, and audio inputs. Example: `examples/new_supports/qwen2_5_omni_3b_grpo.sh`
3.  **Image/Video Generation RL:**  Train RL models on image/video generation models like Stable Diffusion and Wan series. Examples: `examples/new_supports/sd3_image_grpo.sh` and `examples/new_supports/wan_video_grpo.sh`.

## Introduction

Long-RL is a full-stack framework for scaling reasoning in vision-language models (VLMs) to long videos using reinforcement learning.  It includes a large-scale dataset (LongVideo-Reason), a two-stage training pipeline, and a training infrastructure designed for long video RL.

**Supported Models:**

*   VILA series models (image and video) with SP support:
    *   `examples/new_supports/nvila_2b_clevr_grpo.sh`
    *   `examples/new_supports/nvila_2b_video_grpo.sh`
    *   `examples/new_supports/longvila_7b_video_grpo.sh`
*   Qwen-VL series models (text, image, video, and audio) with SP support:
    *   `examples/new_supports/qwen2_5_3b_math_grpo.sh`
    *   `examples/new_supports/qwen2_5_vl_3b_video_grpo.sh`
    *   `examples/new_supports/qwen2_5_omni_3b_grpo.sh`
*   Image and video diffusion model RL:
    *   `examples/new_supports/sd3_image_grpo.sh`
    *   `examples/new_supports/wan_video_grpo.sh`

**Supported Algorithms:**

*   GRPO, DAPO & Reinforce supported, with SP support
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

Tested on `vllm==0.9.1`.  Requires obtaining the remote code first.

```bash
mkdir remote_code
cp path_to/Efficient-Large-Model/LongVILA-R1-7B/*.py remote_code
```

Then, use the following code:

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

*   **Open-ended reward:**  Supports training for open-ended QA tasks.
    *   Set `--worker.rollout.open_ended_reward=True` in the training script.
    *   Set your OpenAI API key: `export OPENAI_API_KEY=xxx`.
*   **Cached video embeddings:** Use cached embeddings for faster training.
    *   Cache video embeddings using `verl/utils/cache_video_embeds_vila.py`.
    *   Set `--data.cache_dir` and `--worker.actor.cached_embeds_dir` in the training script.
*   **Chunked gathering:**  Optimize for memory efficiency with large batches or long video frames.
    *   Set `--worker.rollout.num_chunk_seq` in the training script (e.g., 8, 16, or 32).

## Installation

```bash
git clone https://github.com/NVlabs/Long-RL.git
cd Long-RL
pip install -e .
```

If training Qwen-Omni models:

```bash
bash vllm_replace.sh
```

## Training

### Single node

Refer to the example scripts in the `examples` directory. For instance:

```bash
bash examples/new_supports/qwen2_5_vl_3b_video_grpo.sh $VIDEO_PATH
```

### Multi-nodes

Follow the instructions in the EasyR1 repository [here](https://github.com/hiyouga/EasyR1/tree/main?tab=readme-ov-file#how-to-run-70b-model-in-multi-node-environment) for multi-node training setup.

Example `sbatch` script usage:

```bash
bash scripts/srun_multi_nodes.sh $TRAIN_SCRIPT $NNODES
```

For example:

```bash
bash scripts/srun_multi_nodes.sh examples/new_supports/qwen2_5_vl_3b_video_grpo.sh 2
```

### Merge Checkpoint in Hugging Face Format

Use the following script from the EasyR1 repo.

```bash
python3 scripts/model_merger.py --local_dir checkpoints/easy_r1/exp_name/global_step_1/actor
```

## LongVideo-Reason

Detailed instructions for the data generation process and model evaluation on the `LongVideo-Reason` benchmark are available in the [`longvideo-reason`](longvideo-reason/) directory.

## Examples

Example videos demonstrating Long-RL's capabilities are available:

*   [Football Video](https://drive.google.com/file/d/1QJ-ZsDrmYS8v1XU4eWfYu5oHuXeyGSdK/view?usp=share_link)
    <p align="center" width="100%">
    <img src="assets/example-football.png" alt="Football Example" style="width: 100%; min-width: 300px; display: block; margin: auto;">
    </p>

*   [Texas Hold’em Video](https://drive.google.com/file/d/1U0N563a2s24o_NDie1VfWauxFuSu31wC/view?usp=share_link)
    <p align="center" width="100%">
    <img src="assets/example-TexasHold.png" alt="Texas Hold'em Example" style="width: 100%; min-width: 300px; display: block; margin: auto;">
    </p>

*   [Starcraft II Video](https://drive.google.com/file/d/1rnF4I6-EBpqhzA0SnwyajpxbAhMezDCn/view?usp=share_link)
    <p align="center" width="100%">
    <img src="assets/example-starcraft2.png" alt="Starcraft II Example" style="width: 100%; min-width: 300px; display: block; margin: auto;">
    </p>

*   [Moving Cup Video](https://drive.google.com/file/d/1lo1E_bXXnMmWnFRudaSUgxMNxetEDHP9/view?usp=share_link)
    <p align="center" width="100%">
    <img src="assets/example-movingcup.png" alt="Moving Cup Example" style="width: 100%; min-width: 300px; display: block; margin: auto;">
    </p>

## How to Contribute

1.  Install Git.
2.  Create a [fork](https://github.com/NVlabs/Long-RL/fork) of the project.
3.  Clone your fork locally.
4.  Follow the [Installation](#installation) instructions.
5.  Make your changes, commit, and push.
6.  Submit a pull request.

## Core Contributors

[Yukang Chen](https://yukangchen.com/), [Wei Huang](https://aaron-weihuang.com/), [Shuai Yang](https://andysonys.github.io), [Qinghao Hu](https://tonyhao.xyz/), [Baifeng Shi](https://bfshi.github.io/), [Hanrong Ye](https://sites.google.com/site/yhrspace/home), [Ligeng Zhu](https://lzhu.me/).

We welcome all contributions and will acknowledge all contributors.

## Citation

Please cite our paper and framework if they are helpful in your research:

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

*   [EasyR1](https://github.com/hiyouga/EasyR1):  Base codebase.
*   [verl](https://github.com/volcengine/verl):  RL training framework.
*   [vllm](https://github.com/vllm-project/vllm):  Rollout engine.
*   [Flow-GRPO](https://github.com/yifan123/flow_grpo):  Reference for image/video generation RL.