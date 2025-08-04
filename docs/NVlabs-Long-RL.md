<p align="center" width="100%">
<img src="assets/long-rl-logo.png" alt="Long-RL Logo" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

# Long-RL: Revolutionizing Long Video Reasoning with Reinforcement Learning

**Long-RL empowers vision-language models to excel in long video reasoning, introducing innovative techniques for efficient and effective training.**  Learn more about this groundbreaking framework on the [original repository](https://github.com/NVlabs/Long-RL).

[![Paper](https://img.shields.io/badge/ArXiv-Paper-brown)](https://arxiv.org/abs/2507.07966)
[![Code](https://img.shields.io/badge/GitHub-Long%20RL-blue)](https://github.com/NVlabs/Long-RL)
[![Model](https://img.shields.io/badge/HuggingFace-Model-yellow)](https://huggingface.co/Efficient-Large-Model/LongVILA-R1-7B)
[![Video](https://img.shields.io/badge/YouTube-Video-red)](https://www.youtube.com/watch?v=ykbblK2jiEg)
[![Demo](https://img.shields.io/badge/Gradio-Demo-bron)](https://long-rl.hanlab.ai)

<div align="center">

[![Watch the video](assets/demo_video_first_frame.png)](https://www.youtube.com/watch?v=ykbblK2jiEg)

</div>

## Key Features

*   🚀 **Scalable Training:**  Supports RL training on hour-long videos (e.g., 3,600 frames / ~256k tokens) on a single A100 node (8 GPUs) using sequence parallelism.
*   🎥 **Omni-Model Support:** Enables RL training for models that accept text, video, and audio inputs.
*   🎨 **Image/Video Generation RL:** Supports RL training for image/video generation models such as Stable Diffusion and Wan series.
*   🧠 **Enhanced Reasoning:**  Achieves state-of-the-art performance on video benchmarks through a two-stage training pipeline.
*   💡 **Efficient Infrastructure:** Introduces Multi-modal Reinforcement Sequence Parallelism (MR-SP) for efficient long video RL training, leading to up to 2.1x speedup.
*   🖼️ **Comprehensive Dataset:** Leverages the LongVideo-Reason dataset with 104K long video QA pairs for high-quality reasoning.
*   ✨ **Model Availability:** The LongVILA-R1-7B model is available on Hugging Face and achieves strong results on VideoMME (65.1% / 71.1%).
*   ⚙️ **Flexible FPS control:** Supports adjusting the frames per second (FPS) value for video processing.
*   🔄 **Open-ended Reward:** Support training for open-ended QAs (non-multi-choices QAs) by setting `--worker.rollout.open_ended_reward=True` in the training script.
*   💾 **Cached Video Embeddings:** Efficient training using cached video embeddings via `--data.cache_dir` and `--worker.actor.cached_embeds_dir` in the training script.
*   🧩 **Chunked Gathering:** Reduce memory usage using chunked gathering for `all_gather_data_proto` with `--worker.rollout.num_chunk_seq` in the training script.

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
11. [Core Contributors](#core-Contributors)
12. [Citation](#citation)
13. [Acknowledgement](#acknowledgement)

## News
-   [x] \[2025.7.30] **LongVILA-R1-7B** now supports processing up to **8,192** video frames per video, with configurable FPS settings. Refer to its [usage instructions](#longvila-r1-model-usage).
-   [x] \[2025.7.24] Gradio demo of LongVILA-R1-7B available at (https://long-rl.hanlab.ai).
-   [x] \[2025.7.24] Weights for **LongVILA-R1-7B** released on HuggingFace (https://huggingface.co/Efficient-Large-Model/LongVILA-R1-7B). Achieves **65.1% / 71.1%** on VideoMME. Supports both **multiple-choice** and **open-ended** questions, and non-thinking mode.
-   [x] \[2025.7.19] Detailed instructions and scripts for LongVideo-Reason dataset generation in the [`longvideo-reason`](longvideo-reason/) directory.
-   [x] \[2025.7.18] New features released, including *Open-ended reward*, *Cached video embeddings*, and *Chunked gathering* (see [Supported Features](#supported-features)).
-   [x] \[2025.7.10] Paper and GitHub repo [Long-RL](https://github.com/NVlabs/Long-RL) released.

## Highlights
*   **Hour-level long video RL training on a single node**: Supports RL training on hour-level videos (3,600 frames - 256k tokens) with sequence parallel, on a single A100 node (8 GPUs). `examples/new_supports/qwen2_5_vl_3b_video_1h.sh`
*   **Omni-model RL**: Supports RL training on omni models, that take text, video, and audio for inputs. `examples/new_supports/qwen2_5_omni_3b_grpo.sh`
*   **Image/video generation RL**: Supports RL training on image/video generation models, like [Stable Diffusion](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium) and [Wan](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers) series models. `examples/new_supports/sd3_image_grpo.sh` and `examples/new_supports/wan_video_grpo.sh`.

## Introduction
**Support models**:
-   [x] VILA series models on image and video, with SP support
    -   `examples/new_supports/nvila_2b_clevr_grpo.sh`
    -   `examples/new_supports/nvila_2b_video_grpo.sh`
    -   `examples/new_supports/longvila_7b_video_grpo.sh`
-   [x] Qwen-VL series models on text, image, video, and audio, with SP support
    -   `examples/new_supports/qwen2_5_3b_math_grpo.sh`
    -   `examples/new_supports/qwen2_5_vl_3b_video_grpo.sh`
    -   `examples/new_supports/qwen2_5_omni_3b_grpo.sh`
-   [x] Image and video diffusion model RL
    -   `examples/new_supports/sd3_image_grpo.sh`
    -   `examples/new_supports/wan_video_grpo.sh`

**Support algorithms**:
-   [x] In addition to GRPO, DAPO & Reinforce supported, with SP support
    -   `examples/new_supports/qwen2_5_vl_3b_video_dapo.sh`
    -   `examples/new_supports/qwen2_5_vl_3b_video_grpo.sh`
    -   `examples/new_supports/qwen2_5_vl_3b_video_reinforce.sh`

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
-   [x] **Open-ended reward**:
    -   Set `--worker.rollout.open_ended_reward=True` in the training script.
    -   Export your OpenAI API key: `export OPENAI_API_KEY=xxx`.
-   [x] **Cached video embeddings**:
    -   Cache video embeddings in a local directory using `verl/utils/cache_video_embeds_vila.py`.
    -   Set `--data.cache_dir` and `--worker.actor.cached_embeds_dir` in the training script.
-   [x] **Chunked gathering**:
    -   Set `--worker.rollout.num_chunk_seq` in the training script (e.g., 8/16/32).

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
For single node (within 8 GPUs), you can refer to the training scripts in the `examples` directory. For example,
```bash
bash examples/new_supports/qwen2_5_vl_3b_video_grpo.sh $VIDEO_PATH
```

### Multi-nodes
For jobs that requires multi-nodes, you can refer to the ways mentioned in the EasyR1 repo, [here](https://github.com/hiyouga/EasyR1/tree/main?tab=readme-ov-file#how-to-run-70b-model-in-multi-node-environment).

We provide additional examples for `sbatch` scripts like, where `TRAIN_SCRIPT` is the script to train on single node, `NNODES` is the number of nodes required.
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
Detailed instructions for data generation and model evaluation are available in the [`longvideo-reason`](longvideo-reason/) directory.

## Examples
<div align="center">
<a href="https://drive.google.com/file/d/1QJ-ZsDrmYS8v1XU4eWfYu5oHuXeyGSdK/view?usp=share_link">Football Video</a>
</div>
<p align="center" width="100%">
<img src="assets/example-football.png" alt="Stanford-Alpaca" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

<div align="center">
<a href="https://drive.google.com/file/d/1U0N563a2s24o_NDie1VfWauxFuSu31wC/view?usp=share_link">Texas Hold’em Video</a>
</div>
<p align="center" width="100%">
<img src="assets/example-TexasHold.png" alt="Stanford-Alpaca" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

<div align="center">
<a href="https://drive.google.com/file/d/1rnF4I6-EBpqhzA0SnwyajpxbAhMezDCn/view?usp=share_link">Starcraft II Video</a>
</div>
<p align="center" width="100%">
<img src="assets/example-starcraft2.png" alt="Stanford-Alpaca" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

<div align="center">
<a href="https://drive.google.com/file/d/1lo1E_bXXnMmWnFRudaSUgxMNxetEDHP9/view?usp=share_link">Moving Cup Video</a>
</div>
<p align="center" width="100%">
<img src="assets/example-movingcup.png" alt="Stanford-Alpaca" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

## How to contribute
-   Install git.
-   Create a [fork](https://github.com/NVlabs/Long-RL/fork) of the project.
-   Clone the repository locally using `git clone`.
-   Follow the [Installation](#installation) instructions above.
-   Commit and push your changes.
-   Create a pull request.

## Core Contributors
[Yukang Chen](https://yukangchen.com/), [Wei Huang](https://aaron-weihuang.com/), [Shuai Yang](https://andysonys.github.io), [Qinghao Hu](https://tonyhao.xyz/), [Baifeng Shi](https://bfshi.github.io/), [Hanrong Ye](https://sites.google.com/site/yhrspace/home), [Ligeng Zhu](https://lzhu.me/).

Contributions are welcome!

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
-   [EasyR1](https://github.com/hiyouga/EasyR1): the codebase we built upon. Thanks for their wonderful work.
-   [verl](https://github.com/volcengine/verl): the RL training framework we built upon.
-   [vllm](https://github.com/vllm-project/vllm): we built upon vllm for the rollout engine.
-   [Flow-GRPO](https://github.com/yifan123/flow_grpo): we refer to the Flow-GRPO for the image/video generation RL part.