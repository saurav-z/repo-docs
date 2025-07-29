<p align="center" width="100%">
<img src="assets/long-rl-logo.png" alt="Long-RL Logo" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

# Long-RL: Revolutionizing Reinforcement Learning for Long Video Understanding

Long-RL introduces a groundbreaking framework for scaling reinforcement learning (RL) to handle long video sequences, unlocking new possibilities in video understanding.  Access the original repository on GitHub: [NVlabs/Long-RL](https://github.com/NVlabs/Long-RL).

[![Paper](https://img.shields.io/badge/ArXiv-Paper-brown)](https://arxiv.org/abs/2507.07966)
[![Code](https://img.shields.io/badge/GitHub-Long%20RL-blue)](https://github.com/NVlabs/Long-RL)
[![Model](https://img.shields.io/badge/HuggingFace-Model-yellow)](https://huggingface.co/Efficient-Large-Model/LongVILA-R1-7B)
[![Video](https://img.shields.io/badge/YouTube-Video-red)](https://www.youtube.com/watch?v=ykbblK2jiEg)
[![Demo](https://img.shields.io/badge/Gradio-Demo-bron)](https://long-rl.hanlab.ai)

<div align="center">
<a href="https://www.youtube.com/watch?v=ykbblK2jiEg">
    <img src="assets/demo_video_first_frame.png" alt="Demo Video Preview" style="width: 50%; min-width: 200px; display: block; margin: auto;">
</a>
</div>

## Key Features

*   **Scalable Long Video Reasoning:**  Effectively handles and reasons over extended video sequences.
*   **State-of-the-Art Performance:** Achieves impressive results on video benchmarks with LongVILA-R1-7B, outperforming prior models.
*   **Multi-Modal RL Training:** Supports RL training across various modalities including video, text, and audio.
*   **Flexible Model Support:** Compatible with VILA, Qwen series, and image/video generation models (Stable Diffusion, Wan).
*   **Optimized Training Infrastructure:** Multi-modal Reinforcement Sequence Parallelism (MR-SP) enables efficient RL training on long videos, offering up to 2.1x speedup.
*   **Open-Ended QA Support:** Supports training for open-ended question-answering tasks.
*   **Caching & Chunking:** Provides features like cached video embeddings and chunked gathering for efficient training.

## Introduction

Long-RL empowers vision-language models (VLMs) to understand and reason over lengthy videos, crucial for real-world applications. This is achieved through:

*   **LongVideo-Reason Dataset:** A large-scale dataset with 104K long video QA pairs, facilitating robust model training.
*   **Two-Stage Training Pipeline:**  A pipeline that incorporates chain-of-thought supervised fine-tuning (CoT-SFT) and reinforcement learning (RL) for effective learning.
*   **Multi-modal Reinforcement Sequence Parallelism (MR-SP):**  A training infrastructure designed for long video RL, including sequence parallelism and a vLLM-based engine.

## Key Performance Highlights:

*   **VideoMME:** Achieves 65.0% accuracy without subtitles and 70.7% with subtitles using LongVILA-R1-7B.
*   **Consistent Improvement:** Demonstrated performance gains as the number of input video frames increases.

## Model Usage

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

*   **Open-ended reward:** Supports training for open-ended QAs, requiring an OpenAI API key.
*   **Cached video embeddings:** Speeds up training by caching video embeddings, controlled by `--data.cache_dir` and `--worker.actor.cached_embeds_dir` in the training script.
*   **Chunked gathering:** Optimizes memory usage with chunked gathering for `all_gather_data_proto`, managed by `--worker.rollout.num_chunk_seq`.

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

Refer to training scripts in the `examples` directory. For example:

```bash
bash examples/new_supports/qwen2_5_vl_3b_video_grpo.sh $VIDEO_PATH
```

### Multi-nodes

Refer to the EasyR1 repo for running multi-node jobs. See examples in the `scripts` directory.

```bash
bash scripts/srun_multi_nodes.sh $TRAIN_SCRIPT $NNODES
```

### Merge Checkpoint in Hugging Face Format

```bash
python3 scripts/model_merger.py --local_dir checkpoints/easy_r1/exp_name/global_step_1/actor
```

## LongVideo-Reason

Detailed instructions for data generation and model evaluation are available in the [`longvideo-reason`](longvideo-reason/) directory.

## Examples

*   [Football Video](https://drive.google.com/file/d/1QJ-ZsDrmYS8v1XU4eWfYu5oHuXeyGSdK/view?usp=share_link)
    <p align="center" width="100%">
    <img src="assets/example-football.png" alt="Stanford-Alpaca" style="width: 100%; min-width: 300px; display: block; margin: auto;">
    </p>
*   [Texas Hold’em Video](https://drive.google.com/file/d/1U0N563a2s24o_NDie1VfWauxFuSu31wC/view?usp=share_link)
    <p align="center" width="100%">
    <img src="assets/example-TexasHold.png" alt="Stanford-Alpaca" style="width: 100%; min-width: 300px; display: block; margin: auto;">
    </p>
*   [Starcraft II Video](https://drive.google.com/file/d/1rnF4I6-EBpqhzA0SnwyajpxbAhMezDCn/view?usp=share_link)
    <p align="center" width="100%">
    <img src="assets/example-starcraft2.png" alt="Stanford-Alpaca" style="width: 100%; min-width: 300px; display: block; margin: auto;">
    </p>
*   [Moving Cup Video](https://drive.google.com/file/d/1lo1E_bXXnMmWnFRudaSUgxMNxetEDHP9/view?usp=share_link)
    <p align="center" width="100%">
    <img src="assets/example-movingcup.png" alt="Stanford-Alpaca" style="width: 100%; min-width: 300px; display: block; margin: auto;">
    </p>

## How to Contribute

*   Install git.
*   Fork the project on GitHub.
*   Clone the repository locally.
*   Review the Installation sections above.
*   Make and commit your changes.
*   Submit a pull request.

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

*   [EasyR1](https://github.com/hiyouga/EasyR1): The foundation of our codebase.
*   [verl](https://github.com/volcengine/verl): For the RL training framework.
*   [vllm](https://github.com/vllm-project/vllm):  Used for the rollout engine.
*   [Flow-GRPO](https://github.com/yifan123/flow_grpo): Inspiration for the image/video generation RL part.