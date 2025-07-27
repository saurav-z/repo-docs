# Long-RL: Revolutionizing Long Video Reasoning with Reinforcement Learning

**Tackle long video reasoning with Long-RL, a full-stack framework leveraging reinforcement learning to significantly improve performance in vision-language models (VLMs).** ([Original Repo](https://github.com/NVlabs/Long-RL))

[![Paper](https://img.shields.io/badge/ArXiv-Paper-brown)](https://arxiv.org/abs/2507.07966)
[![Code](https://img.shields.io/badge/GitHub-Long%20RL-blue)](https://github.com/NVlabs/Long-RL)
[![Model](https://img.shields.io/badge/HuggingFace-Model-yellow)](https://huggingface.co/Efficient-Large-Model/LongVILA-R1-7B)
[![Video](https://img.shields.io/badge/YouTube-Video-red)](https://www.youtube.com/watch?v=ykbblK2jiEg)
[![Demo](https://img.shields.io/badge/Gradio-Demo-bron)](https://long-rl.hanlab.ai)

<div align="center">
<a href="https://www.youtube.com/watch?v=ykbblK2jiEg">
  <img src="assets/demo_video_first_frame.png" alt="Demo Video First Frame" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</a>
</div>

## Key Features

*   **Enhanced Performance:** LongVILA-R1-7B achieves **65.0% / 70.7%** accuracy on VideoMME (without/with subtitles).
*   **Hour-Level Training:** Train on videos up to an hour long (3,600 frames, ~256k tokens) using a single A100 node (8 GPUs).
*   **Multi-Modal Support:** Supports RL training with text, video, and audio inputs.
*   **Model Agnostic:** Compatible with a wide range of models including VILA, Qwen-VL, and image/video generation models (Stable Diffusion, Wan series).
*   **Efficient Training:** Utilizes Multi-modal Reinforcement Sequence Parallelism (MR-SP) for up to 2.1x speedup in long video RL training.
*   **Open-Ended Reward Support:**  Train models for open-ended question answering.
*   **Cached Embeddings:**  Optimize training with cached video embeddings.
*   **Chunked Gathering:** Addresses memory constraints for large datasets and long videos.

## Introduction

Long-RL introduces a comprehensive framework for scaling reinforcement learning to address the challenges of long video reasoning in vision-language models. This framework incorporates:

*   **LongVideo-Reason Dataset:** A large-scale dataset with 104K long video QA pairs, featuring high-quality reasoning annotations across diverse domains.
*   **Two-Stage Training Pipeline:** Extends VLMs with Chain-of-Thought Supervised Fine-tuning (CoT-SFT) followed by Reinforcement Learning (RL).
*   **Multi-modal Reinforcement Sequence Parallelism (MR-SP):** An efficient training infrastructure that integrates sequence parallelism and a vLLM-based engine, optimizing for long video processing.

## Supported Features

*   **Open-ended reward:** Support training for open-ended QAs. Requires setting `--worker.rollout.open_ended_reward=True` and setting your OpenAI API key.
*   **Cached video embeddings:** Use cached video embeddings.  Follow `verl/utils/cache_video_embeds_vila.py` to cache video embeddings and then set `--data.cache_dir` and `--worker.actor.cached_embeds_dir` in the training script.
*   **Chunked gathering:** For memory efficiency, set `--worker.rollout.num_chunk_seq` to use chunked gathering.

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

Refer to the `examples` directory for training scripts.

### Single Node

```bash
bash examples/new_supports/qwen2_5_vl_3b_video_grpo.sh $VIDEO_PATH
```

### Multi-Nodes

Utilize the `scripts/srun_multi_nodes.sh` script for multi-node training.

```bash
bash scripts/srun_multi_nodes.sh $TRAIN_SCRIPT $NNODES
```

### Merge Checkpoint in Hugging Face Format

```bash
python3 scripts/model_merger.py --local_dir checkpoints/easy_r1/exp_name/global_step_1/actor
```

## LongVideo-Reason

Detailed instructions on data generation and model evaluation for the `LongVideo-Reason` benchmark can be found in the [`longvideo-reason`](longvideo-reason/) directory.

## Examples

[Football Video Example](https://drive.google.com/file/d/1QJ-ZsDrmYS8v1XU4eWfYu5oHuXeyGSdK/view?usp=share_link)
<p align="center" width="100%">
<img src="assets/example-football.png" alt="Football Video Example" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

[Texas Hold'em Video Example](https://drive.google.com/file/d/1U0N563a2s24o_NDie1VfWauxFuSu31wC/view?usp=share_link)
<p align="center" width="100%">
<img src="assets/example-TexasHold.png" alt="Texas Hold'em Video Example" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

[Starcraft II Video Example](https://drive.google.com/file/d/1rnF4I6-EBpqhzA0SnwyajpxbAhMezDCn/view?usp=share_link)
<p align="center" width="100%">
<img src="assets/example-starcraft2.png" alt="Starcraft II Video Example" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

[Moving Cup Video Example](https://drive.google.com/file/d/1lo1E_bXXnMmWnFRudaSUgxMNxetEDHP9/view?usp=share_link)
<p align="center" width="100%">
<img src="assets/example-movingcup.png" alt="Moving Cup Video Example" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

## How to Contribute

*   Install git.
*   Fork the project.
*   Clone the repository.
*   Follow the `Installation` instructions.
*   Commit and push your changes.
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

*   [EasyR1](https://github.com/hiyouga/EasyR1): the codebase we built upon. Thanks for their wonderful work.
*   [verl](https://github.com/volcengine/verl): the RL training framework we built upon.
*   [vllm](https://github.com/vllm-project/vllm): we built upon vllm for the rollout engine.
*   [Flow-GRPO](https://github.com/yifan123/flow_grpo): we refer to the Flow-GRPO for the image/video generation RL part.