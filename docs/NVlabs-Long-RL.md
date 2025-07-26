<p align="center" width="100%">
<img src="assets/long-rl-logo.png" alt="Long-RL Logo" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

# Long-RL: Revolutionizing Long Video Reasoning with Reinforcement Learning

**Long-RL introduces a full-stack framework that scales reasoning in vision-language models (VLMs) to long videos, leveraging reinforcement learning.**  For more details, check out the [original repository](https://github.com/NVlabs/Long-RL).

*   [Paper](https://arxiv.org/abs/2507.07966)
*   [Code](https://github.com/NVlabs/Long-RL)
*   [Model](https://huggingface.co/Efficient-Large-Model/LongVILA-R1-7B)
*   [Video](https://www.youtube.com/watch?v=ykbblK2jiEg)
*   [Demo](https://long-rl.hanlab.ai)

<div align="center">

[![Watch the video](assets/demo_video_first_frame.png)](https://www.youtube.com/watch?v=ykbblK2jiEg)

</div>

**Key Features:**

*   **Hour-Level RL Training:** Train on videos up to an hour long (3,600 frames / ~256k tokens) using sequence parallelism on a single A100 node (8 GPUs).
*   **Multi-Modal RL Support:** Supports RL training with text, video, and audio inputs, enabling a broad range of applications.
*   **Image/Video Generation RL:**  Provides support for training on image/video generation models like Stable Diffusion and Wan series models.
*   **Open-Ended Reward Support:** Train models for open-ended question answering tasks.
*   **Cached Video Embeddings:** Utilize cached video embeddings for faster training.
*   **Chunked Gathering:** Improves memory efficiency during training, especially with large batch sizes or long video sequences.
*   **Strong Performance:** Achieves state-of-the-art results on various video benchmarks.

**Key Components:**

*   **LongVideo-Reason Dataset:** A large-scale dataset with 104K long video QA pairs, providing high-quality reasoning annotations across various domains.
*   **Two-Stage Training Pipeline:** Extends VLMs with chain-of-thought supervised fine-tuning (CoT-SFT) and reinforcement learning (RL).
*   **Multi-modal Reinforcement Sequence Parallelism (MR-SP):** A training infrastructure for long video RL, including sequence parallelism and a vLLM-based engine optimized for long videos.

**Performance Highlights:**

*   **LongVILA-R1-7B** achieves:
    *   65.0% accuracy on VideoMME (without subtitles)
    *   70.7% accuracy on VideoMME (with subtitles)
*   Consistent performance improvements with increasing input video frame counts.
*   Up to 2.1x speedup on long video RL training with the MR-SP system.

**Table of Contents:**

1.  [News](#news)
2.  [Highlights](#highlights)
3.  [Introduction](#introduction)
4.  [Supported Features](#supported-features)
5.  [Installation](#installation)
6.  [Training](#training)
7.  [LongVideo-Reason](#longvideo-reason)
8.  [Examples](#examples)
9.  [How to contribute](#how-to-contribute)
10. [Core Contributors](#core-contributors)
11. [Citation](#citation)
12. [Acknowledgement](#acknowledgement)

## News

*   \[2025.07.24] Gradio demo released: [https://long-rl.hanlab.ai](https://long-rl.hanlab.ai)
*   \[2025.07.24] LongVILA-R1-7B model weights released on Hugging Face: [https://huggingface.co/Efficient-Large-Model/LongVILA-R1-7B](https://huggingface.co/Efficient-Large-Model/LongVILA-R1-7B)
*   \[2025.07.19] Detailed instructions and scripts for LongVideo-Reason dataset generation released in the [`longvideo-reason`](longvideo-reason/) directory.
*   \[2025.07.18] Added support for Open-ended reward, Cached video embeddings, and Chunked gathering.
*   \[2025.07.10] Paper and GitHub repo released.

## Highlights

1.  **Hour-level long video RL training on a single node:** Supports RL training on hour-level videos (3,600 frames - 256k tokens) with sequence parallel, on a single A100 node (8 GPUs). `examples/new_supports/qwen2_5_vl_3b_video_1h.sh`
2.  **Omni-model RL:** Supports RL training on omni models, that take text, video, and audio for inputs. `examples/new_supports/qwen2_5_omni_3b_grpo.sh`
3.  **Image/video generation RL:** Supports RL training on image/video generation models, like [Stable Diffusion](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium) and [Wan](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers) series models. `examples/new_supports/sd3_image_grpo.sh` and `examples/new_supports/wan_video_grpo.sh`.

## Introduction

**Supported Models:**

*   VILA series models on image and video, with SP support
    *   `examples/new_supports/nvila_2b_clevr_grpo.sh`
    *   `examples/new_supports/nvila_2b_video_grpo.sh`
    *   `examples/new_supports/longvila_7b_video_grpo.sh`
*   Qwen-VL series models on text, image, video, and audio, with SP support
    *   `examples/new_supports/qwen2_5_3b_math_grpo.sh`
    *   `examples/new_supports/qwen2_5_vl_3b_video_grpo.sh`
    *   `examples/new_supports/qwen2_5_omni_3b_grpo.sh`
*   Image and video diffusion model RL
    *   `examples/new_supports/sd3_image_grpo.sh`
    *   `examples/new_supports/wan_video_grpo.sh`

**Supported Algorithms:**

*   GRPO, DAPO & Reinforce supported, with SP support
    *   `examples/new_supports/qwen2_5_vl_3b_video_dapo.sh`
    *   `examples/new_supports/qwen2_5_vl_3b_video_grpo.sh`
    *   `examples/new_supports/qwen2_5_vl_3b_video_reinforce.sh`

## Supported Features

*   **Open-ended reward:** Allows training for open-ended QA tasks.
    *   Set `--worker.rollout.open_ended_reward=True` in the training script.
    *   Export your OpenAI API key: `export OPENAI_API_KEY=xxx`.
*   **Cached video embeddings:** Use pre-computed video embeddings to speed up training.
    *   Cache video embeddings using `verl/utils/cache_video_embeds_vila.py`.
    *   Set `--data.cache_dir` and `--worker.actor.cached_embeds_dir` in the training script.
*   **Chunked gathering:** Reduce CPU memory usage during training.
    *   Set `--worker.rollout.num_chunk_seq` in the training script (e.g., 8/16/32).

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

Refer to the training scripts in the `examples` directory for single-node training (within 8 GPUs). For example:

```bash
bash examples/new_supports/qwen2_5_vl_3b_video_grpo.sh $VIDEO_PATH
```

### Multi-nodes

For multi-node training, you can refer to the examples in the EasyR1 repo, [here](https://github.com/hiyouga/EasyR1/tree/main?tab=readme-ov-file#how-to-run-70b-model-in-multi-node-environment).

Example `sbatch` script:

```bash
bash scripts/srun_multi_nodes.sh $TRAIN_SCRIPT $NNODES
```

Where `TRAIN_SCRIPT` is the script to train on a single node and `NNODES` is the number of nodes required.

For example:

```bash
bash scripts/srun_multi_nodes.sh examples/new_supports/qwen2_5_vl_3b_video_grpo.sh 2
```

### Merge Checkpoint in Hugging Face Format

This follows the ways in the EasyR1 repo.

```bash
python3 scripts/model_merger.py --local_dir checkpoints/easy_r1/exp_name/global_step_1/actor
```

## LongVideo-Reason

Detailed instructions for data generation and model evaluation on the `LongVideo-Reason` benchmark are available in the [`longvideo-reason`](longvideo-reason/) directory.

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

*   Ensure you have git installed.
*   Create a [fork](https://github.com/NVlabs/Long-RL/fork) of the project.
*   Clone the repository locally: `git clone [your fork URL]`
*   Follow the `Installation` instructions above.
*   Commit and push your changes.
*   Create a pull request.

## Core Contributors

[Yukang Chen](https://yukangchen.com/), [Wei Huang](https://aaron-weihuang.com/), [Shuai Yang](https://andysonys.github.io), [Qinghao Hu](https://tonyhao.xyz/), [Baifeng Shi](https://bfshi.github.io/), [Hanrong Ye](https://sites.google.com/site/yhrspace/home), [Ligeng Zhu](https://lzhu.me/).

We welcome all contributions!

## Citation

Please cite our paper and this framework if they are helpful in your research:

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
*   [vllm](https://github.com/vllm-project/vllm): Utilized for the rollout engine.
*   [Flow-GRPO](https://github.com/yifan123/flow_grpo):  Referenced for the image/video generation RL component.