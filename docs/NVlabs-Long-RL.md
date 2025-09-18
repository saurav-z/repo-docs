<p align="center" width="100%">
<img src="assets/long-rl-logo.png" alt="Long-RL Logo" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

# Long-RL: Revolutionizing Long Video Reasoning with Reinforcement Learning

**Tackle long video reasoning with Long-RL, a cutting-edge framework utilizing reinforcement learning to scale vision-language models (VLMs) for comprehensive video understanding.**  Explore the [Long-RL GitHub repository](https://github.com/NVlabs/Long-RL) for the latest advancements!

[![Paper](https://img.shields.io/badge/ArXiv-Paper-brown)](https://arxiv.org/abs/2507.07966)
[![Code](https://img.shields.io/badge/GitHub-Long%20RL-blue)](https://github.com/NVlabs/Long-RL)
[![Model](https://img.shields.io/badge/HuggingFace-Model-yellow)](https://huggingface.co/Efficient-Large-Model/LongVILA-R1-7B)
[![Video](https://img.shields.io/badge/YouTube-Video-red)](https://www.youtube.com/watch?v=ykbblK2jiEg)
[![Demo](https://img.shields.io/badge/Gradio-Demo-bron)](https://long-rl.hanlab.ai)

<div align="center">

[![Watch the video](assets/demo_video_first_frame.png)](https://www.youtube.com/watch?v=ykbblK2jiEg)

</div>

## Key Features

*   **Unprecedented Long Video Processing:** Process up to **8,192 video frames** per video with configurable FPS settings, enabling in-depth analysis of extended video content.
*   **Superior Performance:** LongVILA-R1-7B achieves state-of-the-art results on video benchmarks like VideoMME (65.1% / 71.1% w/o/with subtitles) and consistently outperforms LongVILA-7B.
*   **Advanced RL Training Infrastructure:**  Our Multi-modal Reinforcement Sequence Parallelism (MR-SP) system delivers up to a 2.1x speedup on long video RL training, offering significant efficiency gains.
*   **Omni-Model RL Support**:  RL training on omni models (text, video, and audio input).
*   **Image/Video Generation RL**: RL training on image/video generation models, like Stable Diffusion and Wan series models.
*   **Comprehensive Support**: Supports various modalities, models, and algorithms including GRPO, DAPO & Reinforce.
*   **Easy Deployment**: Released a user-friendly Gradio demo and model weights on Hugging Face for easy access and experimentation.
*   **Single Node Training**: Supports RL training on hour-long videos (e.g., 3,600 frames) on a single A100 node (8 GPUs)
*   **Open-ended reward Support**: Supports open-ended QA (non-multi-choices QA) with setting --worker.rollout.open_ended_reward=True in training script.
*   **Cached video embeddings**: Supports cached video embeddings for video RL training.
*   **Chunked gathering**: Supports chunked gathering for all_gather_data_proto.

## What's New
* [x] [2025.7.30] **LongVILA-R1-7B** supports processing up to **8,192** video frames per video, with configurable FPS settings. Please refer to its [usage instructions](#longvila-r1-model-usage).
* [x] [2025.7.24] We release a gradio demo (https://long-rl.hanlab.ai) with our LongVILA-R1-7B model deployed.
* [x] [2025.7.24] We release the model weights of **LongVILA-R1-7B** on HuggingFace (https://huggingface.co/Efficient-Large-Model/LongVILA-R1-7B). LongVILA-R1-7B achieves **65.1% / 71.1%** on VideoMME. It supports reasoning on both **multiple-choice** and **open-ended** questions, and can also switch to non-thinking mode.
* [x] [2025.7.19] We release a detailed instruction and scripts for the data generation process of our LongVideo-Reason dataset in the [`longvideo-reason`](longvideo-reason/) directory.
* [x] [2025.7.18] We release new supported features, including *Open-ended reward*, *Cached video embeddings*, and *Chunked gathering* as introduced in [Supported Features](#supported-features).
* [x] [2025.7.10] We release [Paper](https://arxiv.org/abs/2507.07966) and this GitHub repo [Long-RL](https://github.com/NVlabs/Long-RL).

## Introduction

Long-RL introduces a comprehensive framework for scaling reasoning in vision-language models (VLMs) to long videos, leveraging the power of reinforcement learning. It incorporates:

*   **LongVideo-Reason Dataset:** A large-scale dataset with 104K long video QA pairs and high-quality reasoning annotations across diverse domains.
*   **Two-Stage Training Pipeline:** Extends VLMs using chain-of-thought supervised fine-tuning (CoT-SFT) followed by reinforcement learning (RL).
*   **Multi-modal Reinforcement Sequence Parallelism (MR-SP):**  A training infrastructure tailored for long video RL. It uses sequence parallelism and a vLLM-based engine optimized for efficient rollout and prefilling.

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
- [x] **Open-ended reward**: 
- We support training for open-ended QAs (non-multi-choices QAs). Please do the following steps if you neet it.
  - Set `--worker.rollout.open_ended_reward=True` in the training script.
  - Export your openai API with `export OPENAI_API_KEY=xxx`.
- [x] **Cached video embeddings**:
- We support using cached video embeddings for video RL training. Because video encoding during training is slow for large batch & long video frames. Please do the following steps if you neet it.
  - Follow `verl/utils/cache_video_embeds_vila.py` to cache video embeddings in a local directory.
  - Set `--data.cache_dir` and `--worker.actor.cached_embeds_dir` in the training script.
- [x] **Chunked gathering**:
- We support chunked gathering for `all_gather_data_proto`. Because it might suffer from CPU OOM if you machine do not have enough CPU memory, and also large batches or long video frames are needed. Please do the following step if you neet it.
  - Set `--worker.rollout.num_chunk_seq` in the training script. It can be 8/16/32. Larger ones cost less memory, but more time.

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
Find details on data generation and model evaluation for the `LongVideo-Reason` benchmark in the [`longvideo-reason`](longvideo-reason/) directory.

## Examples

[Example Football Video](https://drive.google.com/file/d/1QJ-ZsDrmYS8v1XU4eWfYu5oHuXeyGSdK/view?usp=share_link)

<p align="center" width="100%">
<img src="assets/example-football.png" alt="Football Video Example" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

[Example Texas Hold’em Video](https://drive.google.com/file/d/1U0N563a2s24o_NDie1VfWauxFuSu31wC/view?usp=share_link)

<p align="center" width="100%">
<img src="assets/example-TexasHold.png" alt="Texas Hold’em Video Example" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

[Example Starcraft II Video](https://drive.google.com/file/d/1rnF4I6-EBpqhzA0SnwyajpxbAhMezDCn/view?usp=share_link)

<p align="center" width="100%">
<img src="assets/example-starcraft2.png" alt="Starcraft II Video Example" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

[Example Moving Cup Video](https://drive.google.com/file/d/1lo1E_bXXnMmWnFRudaSUgxMNxetEDHP9/view?usp=share_link)

<p align="center" width="100%">
<img src="assets/example-movingcup.png" alt="Moving Cup Video Example" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

## How to Contribute

*   Install git.
*   Fork the project on GitHub.
*   Clone your forked repository locally.
*   Follow the `Installation` instructions above.
*   Commit and push your changes.
*   Submit a pull request.

## Core Contributors

[Yukang Chen](https://yukangchen.com/), [Wei Huang](https://aaron-weihuang.com/), [Shuai Yang](https://andysonys.github.io), [Qinghao Hu](https://tonyhao.xyz/), [Baifeng Shi](https://bfshi.github.io/), [Hanrong Ye](https://sites.google.com/site/yhrspace/home), [Ligeng Zhu](https://lzhu.me/).

We welcome contributions and will acknowledge all contributors.

## Citation

If you find this work helpful, please cite it using the following BibTex entries:

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

*   [EasyR1](https://github.com/hiyouga/EasyR1): The codebase we built upon. Thanks for their work.
*   [verl](https://github.com/volcengine/verl): The RL training framework we built upon.
*   [vllm](https://github.com/vllm-project/vllm): We built upon vllm for the rollout engine.
*   [Flow-GRPO](https://github.com/yifan123/flow_grpo): We refer to Flow-GRPO for the image/video generation RL part.