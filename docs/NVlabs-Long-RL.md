<p align="center" width="100%">
<img src="assets/long-rl-logo.png" alt="Long-RL Logo" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

# Long-RL: Revolutionizing Long Video Reasoning with Reinforcement Learning

**Tackle long video reasoning challenges and unlock new possibilities with Long-RL, a full-stack framework leveraging reinforcement learning!**  Check out the original repo: [Long-RL on GitHub](https://github.com/NVlabs/Long-RL).

[![Paper](https://img.shields.io/badge/ArXiv-Paper-brown)](https://arxiv.org/abs/2507.07966)
[![Code](https://img.shields.io/badge/GitHub-Long%20RL-blue)](https://github.com/NVlabs/Long-RL)
[![Model](https://img.shields.io/badge/HuggingFace-Model-yellow)](https://huggingface.co/Efficient-Large-Model/LongVILA-R1-7B)
[![Video](https://img.shields.io/badge/YouTube-Video-red)](https://www.youtube.com/watch?v=ykbblK2jiEg)
[![Demo](https://img.shields.io/badge/Gradio-Demo-bron)](https://long-rl.hanlab.ai)

<div align="center">

[![Watch the video](assets/demo_video_first_frame.png)](https://www.youtube.com/watch?v=ykbblK2jiEg)

</div>

## Key Features

*   **Scalable Long Video Reasoning:** Processes up to 8,192 video frames with configurable FPS.
*   **Superior Performance:** Achieves state-of-the-art results on video benchmarks.
*   **Multi-Modal Support:** Supports training with video, text, and audio inputs.
*   **Flexible Training Infrastructure:** Enables RL training on various models (VILA, Qwen series, image/video generation).
*   **Efficient Training:** Offers up to 2.1x speedup on long video RL training with Multi-modal Reinforcement Sequence Parallelism (MR-SP).
*   **Open-ended Reward support:** Now support training for open-ended QAs.

## Highlights

*   **Hour-level Long Video RL Training:** Train on hour-long videos (3,600 frames - 256k tokens) on a single A100 node (8 GPUs).
*   **Omni-Model RL:** Supports RL training on models that take text, video, and audio for inputs (e.g., Qwen-VL series).
*   **Image/Video Generation RL:** Supports RL training on image/video generation models (e.g., Stable Diffusion, Wan series).

## Introduction

Long-RL introduces a comprehensive framework to scale vision-language models (VLMs) for reasoning on long videos using reinforcement learning. It addresses the challenges of long video understanding with:

*   **LongVideo-Reason Dataset:** A large-scale dataset with 104K long video QA pairs.
*   **Two-Stage Training Pipeline:** Extends VLMs using chain-of-thought supervised fine-tuning (CoT-SFT) and reinforcement learning (RL).
*   **Multi-modal Reinforcement Sequence Parallelism (MR-SP):** An efficient training infrastructure that utilizes sequence parallelism and a vLLM-based engine for fast rollout and prefilling.

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

*   **Open-ended reward**: Support for open-ended QAs.
*   **Cached video embeddings**: Optimize training by caching video embeddings.
*   **Chunked gathering**: Reduces CPU memory requirements during training.

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
Refer to the scripts in the `examples` directory.

```bash
bash examples/new_supports/qwen2_5_vl_3b_video_grpo.sh $VIDEO_PATH
```

### Multi-Nodes
Refer to [EasyR1's documentation](https://github.com/hiyouga/EasyR1/tree/main?tab=readme-ov-file#how-to-run-70b-model-in-multi-node-environment) for multi-node setup.

Example `sbatch` script:

```bash
bash scripts/srun_multi_nodes.sh $TRAIN_SCRIPT $NNODES
```

### Merge Checkpoint in Hugging Face Format
```bash
python3 scripts/model_merger.py --local_dir checkpoints/easy_r1/exp_name/global_step_1/actor
```

## LongVideo-Reason Dataset

Find detailed instructions and evaluation on the `LongVideo-Reason` benchmark in the [`longvideo-reason`](longvideo-reason/) directory.

## Examples

[Football Video](https://drive.google.com/file/d/1QJ-ZsDrmYS8v1XU4eWfYu5oHuXeyGSdK/view?usp=share_link)
<p align="center" width="100%">
<img src="assets/example-football.png" alt="Football Example" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

[Texas Hold’em Video](https://drive.google.com/file/d/1U0N563a2s24o_NDie1VfWauxFuSu31wC/view?usp=share_link)
<p align="center" width="100%">
<img src="assets/example-TexasHold.png" alt="Texas Hold’em Example" style="width: 100%; min-width: 300px; display: block; margin: auto;">
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

*   Install git.
*   Fork the project.
*   Clone the repository.
*   Follow the Installation instructions.
*   Commit and push your changes.
*   Create a pull request.

## Core Contributors

[Yukang Chen](https://yukangchen.com/), [Wei Huang](https://aaron-weihuang.com/), [Shuai Yang](https://andysonys.github.io), [Qinghao Hu](https://tonyhao.xyz/), [Baifeng Shi](https://bfshi.github.io/), [Hanrong Ye](https://sites.google.com/site/yhrspace/home), [Ligeng Zhu](https://lzhu.me/).

We welcome all contributions!

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

*   [EasyR1](https://github.com/hiyouga/EasyR1)
*   [verl](https://github.com/volcengine/verl)
*   [vllm](https://github.com/vllm-project/vllm)
*   [Flow-GRPO](https://github.com/yifan123/flow_grpo)