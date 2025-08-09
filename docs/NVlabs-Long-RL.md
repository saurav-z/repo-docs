<p align="center" width="100%">
<img src="assets/long-rl-logo.png" alt="Long-RL Logo" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

# Long-RL: Supercharge Your Vision-Language Models for Long Video Reasoning with RL!

**Tackle the challenges of long video reasoning with Long-RL, a cutting-edge framework that leverages reinforcement learning to scale vision-language models (VLMs).** Learn more at the [original repository](https://github.com/NVlabs/Long-RL).

[![Paper](https://img.shields.io/badge/ArXiv-Paper-brown)](https://arxiv.org/abs/2507.07966)
[![Code](https://img.shields.io/badge/GitHub-Long%20RL-blue)](https://github.com/NVlabs/Long-RL)
[![Model](https://img.shields.io/badge/HuggingFace-Model-yellow)](https://huggingface.co/Efficient-Large-Model/LongVILA-R1-7B)
[![Video](https://img.shields.io/badge/YouTube-Video-red)](https://www.youtube.com/watch?v=ykbblK2jiEg)
[![Demo](https://img.shields.io/badge/Gradio-Demo-bron)](https://long-rl.hanlab.ai)

<div align="center">

[![Watch the video](assets/demo_video_first_frame.png)](https://www.youtube.com/watch?v=ykbblK2jiEg)

</div>

## Key Features

*   **Breakthrough Performance:** Achieve state-of-the-art results on video benchmarks, including up to **71.1% accuracy on VideoMME (with subtitles)** with LongVILA-R1-7B.
*   **Long Video Support:** Process up to **8,192 video frames** per video with configurable frame rate settings.
*   **Efficient Training:** Experience up to **2.1x speedup** in long video RL training with our Multi-modal Reinforcement Sequence Parallelism (MR-SP) system.
*   **Full-Stack Framework:**  Includes a large-scale dataset, LongVideo-Reason, and a two-stage training pipeline, enhancing VLM reasoning on various domains.
*   **Omni-Model and Generation RL Support:** Supports RL training on omni-models (text, video, audio) and image/video generation models.
*   **Easy to Use:** Integrated with open source frameworks like vLLM and easy to get started with provided training scripts

## What's New

*   **[2025.07.30]** LongVILA-R1-7B supports processing up to **8,192** video frames per video, with configurable FPS settings.
*   **[2025.07.24]**  Release Gradio demo and model weights of **LongVILA-R1-7B** on HuggingFace.
*   **[2025.07.19]** Detailed instructions and scripts for LongVideo-Reason dataset generation.
*   **[2025.07.18]** New supported features like *Open-ended reward*, *Cached video embeddings*, and *Chunked gathering*.
*   **[2025.07.10]** Paper and GitHub repository released.

## Introduction

Long-RL provides a comprehensive framework for scaling Reinforcement Learning (RL) to long video sequences. It addresses the challenges of long video reasoning by integrating:

*   **LongVideo-Reason Dataset:** A large-scale dataset with 104K long video QA pairs with annotations.
*   **Two-Stage Training Pipeline:** Extends VLMs with Chain-of-Thought Supervised Fine-tuning (CoT-SFT) and Reinforcement Learning (RL).
*   **Multi-modal Reinforcement Sequence Parallelism (MR-SP):** An efficient training infrastructure for long video RL, optimized for sequence parallelism and vLLM-based engine.

## LongVILA-R1 Model Usage

### General Inference

```python
# (Implementation details as provided in original README, keep this short & focused)
```
### with vLLM engine

```python
# (Implementation details as provided in original README, keep this short & focused)
```

## Supported Features

*   **Open-ended reward**: Train on open-ended QAs with OpenAI API integration.
*   **Cached video embeddings**: Accelerate training with pre-computed video embeddings.
*   **Chunked gathering**: Optimize memory usage for large batches and long videos.

## Installation

```bash
git clone https://github.com/NVlabs/Long-RL.git
cd Long-RL
pip install -e .
```

## Training

Refer to the `examples` directory for single-node training scripts.  Multi-node training can be achieved using scripts similar to those in EasyR1, as referenced in the original README.

## LongVideo-Reason

Detailed instructions on dataset generation and evaluation are available in the [`longvideo-reason`](longvideo-reason/) directory.

## Examples

*   **[Football Video](https://drive.google.com/file/d/1QJ-ZsDrmYS8v1XU4eWfYu5oHuXeyGSdK/view?usp=share_link)**
*   **[Texas Hold’em Video](https://drive.google.com/file/d/1U0N563a2s24o_NDie1VfWauxFuSu31wC/view?usp=share_link)**
*   **[Starcraft II Video](https://drive.google.com/file/d/1rnF4I6-EBpqhzA0SnwyajpxbAhMezDCn/view?usp=share_link)**
*   **[Moving Cup Video](https://drive.google.com/file/d/1lo1E_bXXnMmWnFRudaSUgxMNxetEDHP9/view?usp=share_link)**

## How to Contribute

Follow standard Git practices to contribute to the project (fork, clone, pull requests).

## Core Contributors

[Yukang Chen](https://yukangchen.com/), [Wei Huang](https://aaron-weihuang.com/), [Shuai Yang](https://andysonys.github.io), [Qinghao Hu](https://tonyhao.xyz/), [Baifeng Shi](https://bfshi.github.io/), [Hanrong Ye](https://sites.google.com/site/yhrspace/home), [Ligeng Zhu](https://lzhu.me/).

## Citation

```bibtex
# (Include the BibTex entries from the original README)
```
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