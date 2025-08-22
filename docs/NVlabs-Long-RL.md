<p align="center" width="100%">
<img src="assets/long-rl-logo.png" alt="Long-RL Logo" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

# Long-RL: Revolutionizing Long Video Reasoning with Reinforcement Learning

**Tackle long video understanding challenges with Long-RL, a cutting-edge framework that scales reinforcement learning to long video sequences!**  Explore the paper, code, models, and demo here:  [https://github.com/NVlabs/Long-RL](https://github.com/NVlabs/Long-RL)

[![Paper](https://img.shields.io/badge/ArXiv-Paper-brown)](https://arxiv.org/abs/2507.07966)
[![Code](https://img.shields.io/badge/GitHub-Long%20RL-blue)](https://github.com/NVlabs/Long-RL)
[![Model](https://img.shields.io/badge/HuggingFace-Model-yellow)](https://huggingface.co/Efficient-Large-Model/LongVILA-R1-7B)
[![Video](https://img.shields.io/badge/YouTube-Video-red)](https://www.youtube.com/watch?v=ykbblK2jiEg)
[![Demo](https://img.shields.io/badge/Gradio-Demo-bron)](https://long-rl.hanlab.ai)

<div align="center">

[![Watch the video](assets/demo_video_first_frame.png)](https://www.youtube.com/watch?v=ykbblK2jiEg)

</div>


## Key Features:

*   **Scalable RL for Long Videos:** Train on hour-long videos (3,600 frames, 256k tokens) with sequence parallelism on a single A100 node (8 GPUs).
*   **Multi-Modal Support:**  Supports RL training with text, video, and audio inputs for comprehensive understanding.
*   **Image/Video Generation RL:**  Experiment with RL training on image/video generation models like Stable Diffusion and Wan series.
*   **High-Quality Dataset:**  Leverages LongVideo-Reason, a large-scale dataset with 104K long video QA pairs and reasoning annotations.
*   **Two-Stage Training Pipeline:**  Employs a two-stage training pipeline, extending VLMs with Chain-of-Thought supervised fine-tuning (CoT-SFT) and reinforcement learning (RL).
*   **Efficient Training Infrastructure:** Utilizes Multi-modal Reinforcement Sequence Parallelism (MR-SP), with sequence parallelism and a vLLM-based engine tailored for long videos.
*   **Strong Performance:**  Achieves state-of-the-art results on video benchmarks, demonstrating significant improvements over existing models.

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

*   [x] **[2025.07.30]** **LongVILA-R1-7B** now supports processing up to **8,192** video frames per video with configurable FPS settings. Refer to [usage instructions](#longvila-r1-model-usage).
*   [x] **[2025.07.24]** Gradio demo released at [https://long-rl.hanlab.ai](https://long-rl.hanlab.ai) with the LongVILA-R1-7B model.
*   [x] **[2025.07.24]** Model weights for **LongVILA-R1-7B** available on Hugging Face ([https://huggingface.co/Efficient-Large-Model/LongVILA-R1-7B](https://huggingface.co/Efficient-Large-Model/LongVILA-R1-7B)).  Achieves **65.1% / 71.1%** on VideoMME and supports both **multiple-choice** and **open-ended** questions.
*   [x] **[2025.07.19]** Detailed instructions and scripts for data generation of the LongVideo-Reason dataset released in the [`longvideo-reason`](longvideo-reason/) directory.
*   [x] **[2025.07.18]** New supported features: *Open-ended reward*, *Cached video embeddings*, and *Chunked gathering*. See [Supported Features](#supported-features).
*   [x] **[2025.07.10]** Paper ([https://arxiv.org/abs/2507.07966](https://arxiv.org/abs/2507.07966)) and this GitHub repo ([https://github.com/NVlabs/Long-RL](https://github.com/NVlabs/Long-RL)) released.

## Highlights

1.  **Hour-level long video RL training on a single node:** Supports RL training on hour-level videos (3,600 frames - 256k tokens) with sequence parallel, on a single A100 node (8 GPUs). `examples/new_supports/qwen2_5_vl_3b_video_1h.sh`
2.  **Omni-model RL:** Supports RL training on omni models that take text, video, and audio for inputs. `examples/new_supports/qwen2_5_omni_3b_grpo.sh`
3.  **Image/video generation RL:** Supports RL training on image/video generation models, like [Stable Diffusion](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium) and [Wan](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers) series models. `examples/new_supports/sd3_image_grpo.sh` and `examples/new_supports/wan_video_grpo.sh`.

## Introduction

Long-RL introduces a full-stack framework that scales up reasoning in vision-language models (VLMs) to long videos by leveraging reinforcement learning, which tackles the unique challenges of long video reasoning.

**Supported Models:**

*   [x] VILA series models on image and video, with SP support
    *   `examples/new_supports/nvila_2b_clevr_grpo.sh`
    *   `examples/new_supports/nvila_2b_video_grpo.sh`
    *   `examples/new_supports/longvila_7b_video_grpo.sh`
*   [x] Qwen-VL series models on text, image, video, and audio, with SP support
    *   `examples/new_supports/qwen2_5_3b_math_grpo.sh`
    *   `examples/new_supports/qwen2_5_vl_3b_video_grpo.sh`
    *   `examples/new_supports/qwen2_5_omni_3b_grpo.sh`
*   [x] Image and video diffusion model RL
    *   `examples/new_supports/sd3_image_grpo.sh`
    *   `examples/new_supports/wan_video_grpo.sh`

**Supported Algorithms:**

*   [x] GRPO, DAPO & Reinforce supported, with SP support
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

*   [x] **Open-ended reward**:
    *   Supports training for open-ended QAs (non-multi-choices QAs).
        *   Set `--worker.rollout.open_ended_reward=True` in the training script.
        *   Export your openai API with `export OPENAI_API_KEY=xxx`.
*   [x] **Cached video embeddings**:
    *   Supports using cached video embeddings for video RL training.
        *   Follow `verl/utils/cache_video_embeds_vila.py` to cache video embeddings.
        *   Set `--data.cache_dir` and `--worker.actor.cached_embeds_dir` in the training script.
*   [x] **Chunked gathering**:
    *   Supports chunked gathering for `all_gather_data_proto`.
        *   Set `--worker.rollout.num_chunk_seq` in the training script (8/16/32 recommended; larger values save memory but increase training time).

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

For single-node training (within 8 GPUs), refer to the example training scripts in the `examples` directory. For instance:

```bash
bash examples/new_supports/qwen2_5_vl_3b_video_grpo.sh $VIDEO_PATH
```

### Multi-nodes

For multi-node jobs, follow the instructions from the EasyR1 repo [here](https://github.com/hiyouga/EasyR1/tree/main?tab=readme-ov-file#how-to-run-70b-model-in-multi-node-environment).

Example `sbatch` scripts are provided, with `TRAIN_SCRIPT` representing the single-node training script and `NNODES` representing the required number of nodes:

```bash
bash scripts/srun_multi_nodes.sh $TRAIN_SCRIPT $NNODES
```

For example,

```bash
bash scripts/srun_multi_nodes.sh examples/new_supports/qwen2_5_vl_3b_video_grpo.sh 2
```

### Merge Checkpoint in Hugging Face Format

This follows the ways in the EasyR1 repo.

```bash
python3 scripts/model_merger.py --local_dir checkpoints/easy_r1/exp_name/global_step_1/actor
```

## LongVideo-Reason

Detailed instructions on data generation and model evaluation on our `LongVideo-Reason` benchmark are available in the [`longvideo-reason`](longvideo-reason/) directory.

## Examples

<div align="center">
<a href="https://drive.google.com/file/d/1QJ-ZsDrmYS8v1XU4eWfYu5oHuXeyGSdK/view?usp=share_link">Football Video</a>
</div>
<p align="center" width="100%">
<img src="assets/example-football.png" alt="Football Example" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

<div align="center">
<a href="https://drive.google.com/file/d/1U0N563a2s24o_NDie1VfWauxFuSu31wC/view?usp=share_link">Texas Hold’em Video</a>
</div>
<p align="center" width="100%">
<img src="assets/example-TexasHold.png" alt="Texas Hold’em Example" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

<div align="center">
<a href="https://drive.google.com/file/d/1rnF4I6-EBpqhzA0SnwyajpxbAhMezDCn/view?usp=share_link">Starcraft II Video</a>
</div>
<p align="center" width="100%">
<img src="assets/example-starcraft2.png" alt="Starcraft II Example" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

<div align="center">
<a href="https://drive.google.com/file/d/1lo1E_bXXnMmWnFRudaSUgxMNxetEDHP9/view?usp=share_link">Moving Cup Video</a>
</div>
<p align="center" width="100%">
<img src="assets/example-movingcup.png" alt="Moving Cup Example" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

## How to contribute

*   Ensure you have git installed.
*   Create a [fork](https://github.com/NVlabs/Long-RL/fork) of the project.
*   Clone the repository to your local machine using `git clone` and the project's URL.
*   Follow the `Installation` instructions above.
*   Make, commit, and push your changes.
*   Submit a pull request when your modifications are finished.

## Core Contributors

[Yukang Chen](https://yukangchen.com/), [Wei Huang](https://aaron-weihuang.com/), [Shuai Yang](https://andysonys.github.io), [Qinghao Hu](https://tonyhao.xyz/), [Baifeng Shi](https://bfshi.github.io/), [Hanrong Ye](https://sites.google.com/site/yhrspace/home), [Ligeng Zhu](https://lzhu.me/).

We welcome all contributions. All contributors will be acknowledged.

## Citation

If you find this work helpful for your research, please cite our paper and the framework.

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

*   [EasyR1](https://github.com/hiyouga/EasyR1):  The base codebase. Thanks for their excellent work.
*   [verl](https://github.com/volcengine/verl): The RL training framework used.
*   [vllm](https://github.com/vllm-project/vllm):  Used for the rollout engine.
*   [Flow-GRPO](https://github.com/yifan123/flow_grpo): Referenced for the image/video generation RL implementation.