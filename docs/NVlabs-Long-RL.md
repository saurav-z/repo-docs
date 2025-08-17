<p align="center" width="100%">
<img src="assets/long-rl-logo.png" alt="Long-RL Logo" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

# Long-RL: Revolutionizing Long Video Reasoning with Reinforcement Learning

**Tackle the complexities of long video understanding with Long-RL, a cutting-edge framework that scales Reinforcement Learning (RL) to handle extended video sequences.** Explore the full potential of our project on [GitHub](https://github.com/NVlabs/Long-RL).

[![Paper](https://img.shields.io/badge/ArXiv-Paper-brown)](https://arxiv.org/abs/2507.07966)
[![Code](https://img.shields.io/badge/GitHub-Long%20RL-blue)](https://github.com/NVlabs/Long-RL)
[![Model](https://img.shields.io/badge/HuggingFace-Model-yellow)](https://huggingface.co/Efficient-Large-Model/LongVILA-R1-7B)
[![Video](https://img.shields.io/badge/YouTube-Video-red)](https://www.youtube.com/watch?v=ykbblK2jiEg)
[![Demo](https://img.shields.io/badge/Gradio-Demo-bron)](https://long-rl.hanlab.ai)

<div align="center">

[![Watch the video](assets/demo_video_first_frame.png)](https://www.youtube.com/watch?v=ykbblK2jiEg)

</div>

## Key Features

*   **Scalable RL for Long Videos:** Train models on hour-long videos with sequence parallelism, even on a single A100 node (8 GPUs).
*   **Multi-Modal Support:** Train with video, text, and audio inputs, supporting various models like VILA and Qwen series.
*   **Enhanced Performance:**  LongVILA-R1-7B achieves state-of-the-art results, e.g.  65.1% / 71.1% on VideoMME (without/with subtitles).
*   **Flexible Frame Rates:** Configurable FPS settings and support for up to 8,192 video frames per video.
*   **Open-Ended QA Support:** Train models for open-ended questions using openAI API.
*   **Efficient Training:** Utilize cached video embeddings and chunked gathering for faster training and reduced memory consumption.
*   **Comprehensive Support:**  Supports GRPO, DAPO, and Reinforce algorithms.
*   **Image/Video Generation RL:** Supports training on image/video generation models like Stable Diffusion and Wan series.
*   **Publicly Available:** Access the model weights of **LongVILA-R1-7B** on HuggingFace.

## What's New

*   **[2025.7.30]** LongVILA-R1-7B supports processing up to 8,192 video frames per video, with configurable FPS settings.
*   **[2025.7.24]**  Gradio demo (https://long-rl.hanlab.ai) and model weights of **LongVILA-R1-7B** released on HuggingFace (https://huggingface.co/Efficient-Large-Model/LongVILA-R1-7B).
*   **[2025.7.19]** Detailed instructions and scripts for the data generation process of our LongVideo-Reason dataset released in the `longvideo-reason/` directory.
*   **[2025.7.18]** New supported features, including Open-ended reward, Cached video embeddings, and Chunked gathering released.
*   **[2025.7.10]** Paper and this GitHub repo [Long-RL](https://github.com/NVlabs/Long-RL) released.

## Introduction

Long-RL introduces a comprehensive framework for scaling vision-language models (VLMs) to effectively reason about long videos. Our approach combines several key elements:

*   **LongVideo-Reason Dataset:** A large-scale dataset of 104K long video QA pairs with detailed reasoning annotations across sports, games, and vlogs.
*   **Two-Stage Training:** A two-stage pipeline incorporating chain-of-thought supervised fine-tuning (CoT-SFT) and reinforcement learning (RL).
*   **Multi-modal Reinforcement Sequence Parallelism (MR-SP):**  A training infrastructure specifically designed for long video RL. This includes sequence parallelism and a vLLM-based engine optimized for long videos, using cached video embeddings for efficient rollout and prefilling.

## Quick Start: LongVILA-R1 Model Usage

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

**[See the original repo for more detailed usage instructions.](https://github.com/NVlabs/Long-RL)**

## Supported Features
*   **Open-ended reward:**
    *   Set `--worker.rollout.open_ended_reward=True` in the training script.
    *   Export your openai API with `export OPENAI_API_KEY=xxx`.
*   **Cached video embeddings:**
    *   Follow `verl/utils/cache_video_embeds_vila.py` to cache video embeddings in a local directory.
    *   Set `--data.cache_dir` and `--worker.actor.cached_embeds_dir` in the training script.
*   **Chunked gathering:**
    *   Set `--worker.rollout.num_chunk_seq` in the training script. It can be 8/16/32. Larger ones cost less memory, but more time.

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

Explore our detailed instructions for the data generation process and model evaluation using the `LongVideo-Reason` benchmark within the [`longvideo-reason`](longvideo-reason/) directory.

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

## How to Contribute

*   Ensure Git is installed.
*   Create your own [fork](https://github.com/NVlabs/Long-RL/fork) of the project.
*   Clone the repository to your local machine.
*   Review the `Installation` instructions.
*   Commit and push your changes.
*   Submit a pull request.

## Core Contributors

[Yukang Chen](https://yukangchen.com/), [Wei Huang](https://aaron-weihuang.com/), [Shuai Yang](https://andysonys.github.io), [Qinghao Hu](https://tonyhao.xyz/), [Baifeng Shi](https://bfshi.github.io/), [Hanrong Ye](https://sites.google.com/site/yhrspace/home), [Ligeng Zhu](https://lzhu.me/).

All contributions are welcome and will be acknowledged.

## Citation

If you use Long-RL in your research, please cite the following:

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

*   [EasyR1](https://github.com/hiyouga/EasyR1): The codebase we built upon.
*   [verl](https://github.com/volcengine/verl): The RL training framework we built upon.
*   [vllm](https://github.com/vllm-project/vllm): We built upon vllm for the rollout engine.
*   [Flow-GRPO](https://github.com/yifan123/flow_grpo):  We refer to the Flow-GRPO for the image/video generation RL part.