# Long-RL: Revolutionizing Reinforcement Learning for Long Videos

**Tackle the challenge of long video reasoning with Long-RL, a cutting-edge framework that scales reinforcement learning to handle extended video sequences.**  [Explore the Original Repository](https://github.com/NVlabs/Long-RL)

[![Paper](https://img.shields.io/badge/ArXiv-Paper-brown)](https://arxiv.org/abs/2507.07966)
[![Code](https://img.shields.io/badge/GitHub-Long%20RL-blue)](https://github.com/NVlabs/Long-RL)
[![Model](https://img.shields.io/badge/HuggingFace-Model-yellow)](https://huggingface.co/Efficient-Large-Model/LongVILA-R1-7B)
[![Video](https://img.shields.io/badge/YouTube-Video-red)](https://www.youtube.com/watch?v=ykbblK2jiEg)
[![Demo](https://img.shields.io/badge/Gradio-Demo-bron)](https://long-rl.hanlab.ai)

<div align="center">

[![Watch the video](assets/demo_video_first_frame.png)](https://www.youtube.com/watch?v=ykbblK2jiEg)

</div>

**Key Features:**

*   **Hour-Level Video Training:** Train on videos up to an hour long (3,600 frames, ~256k tokens) using sequence parallelism on a single A100 node (8 GPUs).
*   **Omni-Model RL Support:** Train on models that accept text, video, and audio inputs, enabling comprehensive multi-modal understanding.
*   **Image/Video Generation RL:** Experiment with reinforcement learning on image and video generation models, including Stable Diffusion and Wan series.
*   **Open-Ended Reward Support:** Train for open-ended question answering tasks using OpenAI API integration.
*   **Cached Video Embeddings:** Enhance training efficiency by utilizing cached video embeddings.
*   **Chunked Gathering:** Optimize memory usage with chunked gathering for large batches and long video sequences.

**What is Long-RL?**
This project introduces a comprehensive framework designed to scale up reasoning in vision-language models (VLMs) to accommodate long video sequences, employing reinforcement learning. It overcomes the unique challenges of long video reasoning through a combination of a large-scale dataset, a two-stage training pipeline, and an innovative training infrastructure.

**Core Components:**

*   **LongVideo-Reason Dataset:** A large-scale dataset comprising 104K long video QA pairs, providing high-quality reasoning annotations across various domains.
*   **Two-Stage Training Pipeline:** Extends VLMs with Chain-of-Thought Supervised Fine-tuning (CoT-SFT) followed by Reinforcement Learning (RL).
*   **Multi-modal Reinforcement Sequence Parallelism (MR-SP):**  An efficient training infrastructure integrating sequence parallelism and a vLLM-based engine, optimized for long video processing.

**Performance:**
LongVILA-R1-7B demonstrates exceptional performance on video benchmarks, achieving 65.0% and 70.7% accuracy on VideoMME (without and with subtitles, respectively). It consistently outperforms LongVILA-R1 across multiple benchmarks and shows steady performance gains with increasing video frame counts.  MR-SP significantly accelerates long video RL training, achieving up to a 2.1x speedup.

**Highlights:**

*   **Superior Performance:** LongVILA-R1-7B outperforms existing models on key video benchmarks.
*   **Efficiency:** MR-SP achieves significant speedups in long video RL training.
*   **Versatility:** Supports RL training across various modalities (video, text, audio) and model types (VILA, Qwen series, and image/video generation models).

**Table of Results:**

| Models             | VideoMME (w/o sub) | VideoMME (w sub) | ActivityNet-QA (test) | LongVideoBench (val) | PerceptionTest (val) | NExT-QA (mc) | VNBench (val) |
|:-------------------|:------------------:|:----------------:|:---------------------:|:--------------------:|:--------------------:|:--------:|:-------------:|
| **LongVILA-7B**    |      **60.1**      |     **65.1**     |       **59.5**        |       **57.1**       |       **58.1**       | **80.7** |   **63.0**    |
| **LongVILA-R1-7B** |      **65.0**      |     **70.7**     |       **64.8**        |       **58.0**       |       **68.9**       | **81.5** |   **75.5**    |

**Table of Contents**
1.  [News](#news)
2.  [Highlights](#highlights)
3.  [Introduction](#introduction)
4.  [Supported Features](#supported-features)
5.  [Installation](#installation)
6.  [Training](#training)
7.  [LongVideo-Reason](#longvideo-reason)
8.  [Examples](#examples)
9.  [How to Contribute](#how-to-contribute)
10. [Core Contributors](#core-Contributors)
11. [Citation](#citation)
12. [Acknowledgement](#acknowledgement)

**News:**

*   \[2025.7.24] Gradio demo released: (https://long-rl.hanlab.ai)
*   \[2025.7.24] LongVILA-R1-7B model weights released on Hugging Face: (https://huggingface.co/Efficient-Large-Model/LongVILA-R1-7B)
*   \[2025.7.19] Instructions and scripts for LongVideo-Reason dataset generation released in the [`longvideo-reason`](longvideo-reason/) directory.
*   \[2025.7.18] New features released: Open-ended reward, Cached video embeddings, and Chunked gathering.
*   \[2025.7.10] Paper and GitHub repository released.

**Installation:**

```bash
git clone https://github.com/NVlabs/Long-RL.git
cd Long-RL
pip install -e .
```

For Qwen-Omni models:
```bash
bash vllm_replace.sh
```

**Training:**

*   **Single Node:** Refer to scripts in the `examples` directory (e.g., `examples/new_supports/qwen2_5_vl_3b_video_grpo.sh $VIDEO_PATH`).
*   **Multi-Node:**  Use `scripts/srun_multi_nodes.sh $TRAIN_SCRIPT $NNODES`, referencing EasyR1 for multi-node environment setup.
*   **Merge Checkpoint:** Use `python3 scripts/model_merger.py --local_dir checkpoints/easy_r1/exp_name/global_step_1/actor`

**LongVideo-Reason:**

Detailed instructions for data generation and model evaluation are available in the [`longvideo-reason`](longvideo-reason/) directory.

**Examples:**

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

**How to Contribute:**

1.  Fork the project.
2.  Clone the repository.
3.  Read the installation sections.
4.  Commit and push your changes.
5.  Make a pull request.

**Core Contributors:**

[Yukang Chen](https://yukangchen.com/), [Wei Huang](https://aaron-weihuang.com/), [Shuai Yang](https://andysonys.github.io), [Qinghao Hu](https://tonyhao.xyz/), [Baifeng Shi](https://bfshi.github.io/), [Hanrong Ye](https://sites.google.com/site/yhrspace/home), [Ligeng Zhu](https://lzhu.me/).

**Citation:**

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

**Acknowledgement:**

*   [EasyR1](https://github.com/hiyouga/EasyR1)
*   [verl](https://github.com/volcengine/verl)
*   [vllm](https://github.com/vllm-project/vllm)
*   [Flow-GRPO](https://github.com/yifan123/flow_grpo)