# R-Zero: Self-Evolving Reasoning LLM from Zero Data

**R-Zero empowers Large Language Models to learn and refine their reasoning skills autonomously, without any pre-existing datasets.**

[Check out the original repository](https://github.com/Chengsong-Huang/R-Zero) for more details.

## Key Features

*   **Autonomous Learning:** R-Zero starts with a base model and requires no external data or labeled examples.
*   **Co-Evolutionary Loop:** A "Challenger" model generates increasingly difficult problems, driving the "Solver" model to continuously improve.
*   **Enhanced Reasoning:**  R-Zero demonstrably boosts reasoning performance across various benchmarks.
*   **Generalization:** Reasoning skills learned in specific domains transfer effectively to general reasoning tasks.
*   **Model Agnostic:** R-Zero improves the performance of different LLM backbones.

## What's New

*   **2025-08-27:** Analysis of iteration scaling and a model taking on two roles added.
*   **2025-08-25:** Code updates for smoother training (via stopit).
*   **2025-08-08:** R-Zero was recognized as `#2 Paper of the day` on [Hugging Face Daily Paper](https://huggingface.co/papers/2508.05004).
*   **2025-08-07:** Paper and code released.

## Overview

R-Zero is a novel framework designed to enhance the reasoning abilities of LLMs through a self-evolving process. Unlike traditional methods that rely on extensive, curated datasets, R-Zero enables LLMs to learn and adapt autonomously, beginning with only a base model.

[<img src="./figs/abstract.png" alt="Abstract" width="500"/>](https://arxiv.org/abs/2508.05004)

The core of R-Zero involves a dynamic co-evolutionary loop between two instances of the same base model:

*   **Challenger 🎯:** This model probes the Solver for weaknesses by generating challenging problems.
*   **Solver 🧠:** This model aims to continuously improve by solving the problems posed by the Challenger.

This interaction creates a targeted, adaptive curriculum. The Challenger learns to ask more effective questions, and the Solver learns to find better answers, all within a self-contained cycle using techniques like majority voting for pseudo-labels and relative policy optimization.

## Quickstart Guide

Get started with R-Zero in a few simple steps:

### 1. Configure Environment and Prepare Directories

```bash
git clone https://github.com/Chengsong-Huang/R-Zero.git

cd R-Zero

pip install -r requirements.txt

export STORAGE_PATH="/path/to/your/storage"
export HUGGINGFACENAME="yourhuggingfacename"

mkdir -p \
  "$STORAGE_PATH/evaluation" \
  "$STORAGE_PATH/models" \
  "$STORAGE_PATH/generated_question" \
  "$STORAGE_PATH/temp_results"
```

### 2. Add API Keys

You'll need to provide API keys for the following:

*   **Hugging Face** and **WandB** (for logging) in `tokens.json`.
*   **OpenAI GPT** (for evaluation) in `evaluation/results_recheck.py`.

### 3. Run the Experiments

Replicate the experimental results with a single script:

```bash
# Format: bash scripts/main.sh [Base_Model_Name] [Abbreviation]

# Example:
bash scripts/main.sh Qwen/Qwen3-4B-Base qwen3-4b
```

## Impressive Results

The table below demonstrates the performance gains achieved by R-Zero.

| Model Name | Overall AVG | MATH AVG | SuperGPQA | MMLU-Pro | BBEH |
|:---|:---:|:---:|:---:|:---:|:---:|
| ***Qwen3-4B-Base*** | | | | | |
| &emsp;Base Model | 27.10 | 42.58 | 20.88 | 37.38 | 7.57 |
| &emsp;Base Challenger | 30.83 | 44.36 | 24.77 | 47.59 | 6.59 |
| &emsp;R-Zero (Iter 1) | 34.27 | 48.06 | **27.92** | 51.69 | 9.42 |
| &emsp;R-Zero (Iter 2) | **34.92** | 48.44 | 27.72 | **53.75** | 9.76 |
| &emsp;R-Zero (Iter 3) | 34.64 | **49.07** | 27.55 | 51.53 | **10.42** |
| ***Qwen3-8B-Base*** | | | | | |
| &emsp;Base Model | 34.49 | 49.18 | 28.33 | 51.80 | 8.63 |
| &emsp;Base Challenger | 36.43 | 51.87 | 30.12 | 54.14 | 9.60 |
| &emsp;R-Zero (Iter 1) | 37.93 | 53.39 | 31.26 | 57.17 | 9.91 |
| &emsp;R-Zero (Iter 2) | 38.45 | 53.84 | **31.58** | 58.20 | 10.20 |
| &emsp;R-Zero (Iter 3) | **38.73** | **54.69** | 31.38 | **58.23** | **10.60** |
| ***OctoThinker-3B*** | | | | | |
| &emsp;Base Model | 12.27 | 26.64 | 10.09 | 10.87 | 1.46 |
| &emsp;Base Challenger | 14.41 | 27.51 | 11.19 | 14.53 | **4.40** |
| &emsp;R-Zero (Iter 1) | 14.93 | 27.76 | 12.21 | 15.72 | 4.05 |
| &emsp;R-Zero (Iter 2) | 15.11 | 28.20 | 12.43 | 16.08 | 3.74 |
| &emsp;R-Zero (Iter 3) | **15.67** | **29.32** | **12.44** | **16.71** | 4.20 |
| ***OctoThinker-8B*** | | | | | |
| &emsp;Base Model | 16.81 | 32.11 | 13.26 | 20.21 | 1.64 |
| &emsp;Base Challenger | 25.08 | 36.41 | 16.99 | 41.46 | 5.46 |
| &emsp;R-Zero (Iter 1) | 26.44 | 37.80 | 19.15 | **42.05** | 6.77 |
| &emsp;R-Zero (Iter 2) | 26.77 | 38.23 | 19.27 | 41.34 | **8.25** |
| &emsp;R-Zero (Iter 3) | **26.88** | **38.52** | **19.82** | 40.92 | **8.25** |

## FAQ for Developers

### Q: What are the hardware requirements?

**A:** Experiments were conducted on an 8-GPU server using models that can run on a single GPU (4B or 8B).  For larger models or different hardware, you may need to modify the code.

### Q: How to handle environment configuration issues?

**A:** This framework is inspired by [EasyR1](https://github.com/hiyouga/EasyR1/tree/main). Refer to their setup instructions or their Docker environment for troubleshooting.

### Q: Where are training logs and model checkpoints saved?

**A:** All generated data, including logs, datasets, and model checkpoints, will be saved in the `STORAGE_PATH` directory. Datasets will also be sent to Hugging Face via `HUGGINGFACENAME`.

### Q: What if the code gets stuck during training?

**A:** A timeout control mitigates potential infinite loops in the `math_verify` library. Restart training from the last checkpoint if this occurs.

## Acknowledgements

R-Zero builds upon the work of [**EasyR1**](https://github.com/hiyouga/EasyR1/tree/main), incorporating its core functionalities. The evaluation process references work from [**General-Reasoner**](https://github.com/TIGER-AI-Lab/General-Reasoner).

## Citation

If you find this work useful, please cite our paper:

```
@article{huang2025rzeroselfevolvingreasoningllm,
      title={R-Zero: Self-Evolving Reasoning LLM from Zero Data}, 
      author={Chengsong Huang and Wenhao Yu and Xiaoyang Wang and Hongming Zhang and Zongxia Li and Ruosen Li and Jiaxin Huang and Haitao Mi and Dong Yu},
      year={2025},
      eprint={2508.05004},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2508.05004}, 
}
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Chengsong-Huang/R-Zero&type=Date)](https://star-history.com/#Chengsong-Huang/R-Zero&Date)