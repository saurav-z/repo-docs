<p align="center" width="100%">
<img src="assets/long-rl-logo.png" alt="Long-RL Logo" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

# Long-RL: Revolutionizing Long Video Reasoning with Reinforcement Learning

**Tackle long video understanding challenges with Long-RL, a full-stack framework leveraging reinforcement learning (RL) to scale vision-language models (VLMs).** ([Original Repo](https://github.com/NVlabs/Long-RL))

[![Paper](https://img.shields.io/badge/ArXiv-Paper-brown)](https://arxiv.org/abs/2507.07966)
[![Code](https://img.shields.io/badge/GitHub-Long%20RL-blue)](https://github.com/NVlabs/Long-RL)
[![Model](https://img.shields.io/badge/HuggingFace-Model-yellow)](https://huggingface.co/Efficient-Large-Model/LongVILA-R1-7B)
[![Video](https://img.shields.io/badge/YouTube-Video-red)](https://www.youtube.com/watch?v=ykbblK2jiEg)
[![Demo](https://img.shields.io/badge/Gradio-Demo-bron)](https://long-rl.hanlab.ai)

<div align="center">

[![Watch the video](assets/demo_video_first_frame.png)](https://www.youtube.com/watch?v=ykbblK2jiEg)

</div>

## Key Features:

*   **Scalable Long Video RL:** Train on hour-long videos (3,600 frames) using sequence parallelism on a single A100 node (8 GPUs).
*   **Omni-Model Support:** Train RL models using text, video, and audio inputs.
*   **Image & Video Generation RL:** Supports RL training with image and video generation models like Stable Diffusion and Wan series.
*   **LongVILA-R1-7B**: Process up to 8,192 video frames per video, configurable FPS and achieves state-of-the-art results.
*   **Multi-modal Reinforcement Sequence Parallelism (MR-SP):** Achieve up to 2.1x speedup on long video RL training.
*   **Open-ended Reward Support:** Train for open-ended QA tasks.
*   **Cached Video Embeddings:** Faster training with cached video embeddings.
*   **Chunked Gathering:** Optimize memory usage with chunked gathering for large batches.
*   **Gradio Demo and Hugging Face Model:** Explore the capabilities of LongVILA-R1-7B through a Gradio demo and access the model weights on Hugging Face.

## News

*   **[2025.7.30]** LongVILA-R1-7B supports processing up to 8,192 video frames per video, with configurable FPS settings.
*   **[2025.7.24]** Gradio Demo released (https://long-rl.hanlab.ai).
*   **[2025.7.24]** Model weights of LongVILA-R1-7B released on HuggingFace (https://huggingface.co/Efficient-Large-Model/LongVILA-R1-7B).
*   **[2025.7.19]** Detailed data generation instructions for the LongVideo-Reason dataset released.
*   **[2025.7.18]** New features released: Open-ended reward, Cached video embeddings, and Chunked gathering.
*   **[2025.7.10]** Paper and GitHub repo released.

## Highlights

*   **Hour-level long video RL training on a single node**: Supports RL training on hour-level videos (3,600 frames - 256k tokens) with sequence parallel, on a single A100 node (8 GPUs). `examples/new_supports/qwen2_5_vl_3b_video_1h.sh`
*   **Omni-model RL**: Supports RL training on omni models, that take text, video, and audio for inputs. `examples/new_supports/qwen2_5_omni_3b_grpo.sh`
*   **Image/video generation RL**: Supports RL training on image/video generation models, like [Stable Diffusion](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium) and [Wan](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers) series models. `examples/new_supports/sd3_image_grpo.sh` and `examples/new_supports/wan_video_grpo.sh`.

## Introduction

Long-RL is designed to scale reasoning in vision-language models (VLMs) to long videos using reinforcement learning. It addresses the unique challenges of long video reasoning, integrating:

*   **LongVideo-Reason Dataset:** A large-scale dataset with 104K long video QA pairs.
*   **Two-Stage Training Pipeline:**  Chain-of-thought supervised fine-tuning (CoT-SFT) followed by reinforcement learning (RL).
*   **Multi-modal Reinforcement Sequence Parallelism (MR-SP):** Training infrastructure for long video RL.

### Supported Models:

*   VILA series models
*   Qwen-VL series models
*   Image and video diffusion model RL

### Supported Algorithms:

*   GRPO
*   DAPO
*   Reinforce

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

*   **Open-ended reward**: Train on open-ended QAs using an OpenAI API key.
*   **Cached video embeddings**:  Utilize pre-computed video embeddings for faster training.
*   **Chunked gathering**: Optimize memory usage during distributed training.

## Installation

```bash
git clone https://github.com/NVlabs/Long-RL.git
cd Long-RL
pip install -e .
```

If you want to train Qwen-Omni models, please:

```bash
bash vllm_replace.sh
```

## Training

### Single node

Refer to example scripts in the `examples` directory for single-node training.
For example:

```bash
bash examples/new_supports/qwen2_5_vl_3b_video_grpo.sh $VIDEO_PATH
```

### Multi-nodes

Use `srun` for multi-node training as described in the EasyR1 repo. Examples are provided.

```bash
bash scripts/srun_multi_nodes.sh $TRAIN_SCRIPT $NNODES
```

### Merge Checkpoint in Hugging Face Format

```bash
python3 scripts/model_merger.py --local_dir checkpoints/easy_r1/exp_name/global_step_1/actor
```

## LongVideo-Reason

Detailed instructions on data generation and model evaluation on the `LongVideo-Reason` benchmark are provided in the [`longvideo-reason`](longvideo-reason/) directory.

## Examples

[Include image and links to examples here as in original README]

## How to Contribute

*   Install git.
*   Fork the project.
*   Clone the repository.
*   Read installation sections above.
*   Commit and push changes.
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

*   [EasyR1](https://github.com/hiyouga/EasyR1): Codebase built upon.
*   [verl](https://github.com/volcengine/verl):  RL training framework.
*   [vllm](https://github.com/vllm-project/vllm):  Rollout engine.
*   [Flow-GRPO](https://github.com/yifan123/flow_grpo):  Image/video generation RL inspiration.