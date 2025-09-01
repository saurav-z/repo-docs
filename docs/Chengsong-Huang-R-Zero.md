# R-Zero: Self-Evolving Reasoning LLM from Zero Data

**R-Zero revolutionizes how Large Language Models learn to reason by enabling them to autonomously improve without any initial training data.** ([Original Repo](https://github.com/Chengsong-Huang/R-Zero))

R-Zero empowers LLMs to learn and evolve their reasoning capabilities, starting from scratch without the need for curated datasets. This innovative framework leverages a self-evolving process to significantly enhance reasoning performance.

## Key Features:

*   **Autonomous Learning:** Starts with a base model and requires *no* pre-existing datasets or labeled examples.
*   **Co-Evolutionary Architecture:** Features a "Challenger" and "Solver" dynamic, creating a tailored and adaptive learning curriculum.
*   **Proven Performance Gains:** Achieves significant improvements on various reasoning benchmarks.
*   **Strong Generalization:** Reasoning skills learned in specific domains effectively transfer to general reasoning tasks.
*   **Model Agnostic:** Consistently boosts the performance of diverse backbone LLMs.

## Updates
* [2025-8-27] Analysis on iteration scaling and one model taking on two roles added.
* [2025-8-25] Code updated for smoother training (using stopit).
* [2025-8-8] R-Zero recognized as `#2 Paper of the day` on [huggingface daily paper](https://huggingface.co/papers/2508.05004).
* [2025-8-7] Paper and code released.

## Overview

R-Zero employs a unique co-evolutionary loop between two instances of the same base model:

1.  **Challenger:** Probes the Solver for weaknesses by generating challenging problems.
2.  **Solver:** Continuously improves by solving the tasks presented by the Challenger.

This self-contained cycle uses techniques like majority voting for pseudo-labels and relative policy optimization.

## Quickstart Guide

Follow these steps to get started:

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

Add your API keys for Hugging Face and WandB (for logging) in `tokens.json`, and your OpenAI GPT API key in `evaluation/results_recheck.py`.

### 3. Run Experiments

Run the following script to replicate experimental results:

```bash
# Format: bash scripts/main.sh [Base_Model_Name] [Abbreviation]
# Example:
bash scripts/main.sh Qwen/Qwen3-4B-Base qwen3-4b
```

## Results

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

**Q: What is the hardware setup?**

**A:** Experiments were conducted on an 8-GPU server, using models suitable for a single GPU (4B or 8B). Adapt code for larger models or different hardware.

**Q: What if I encounter environment configuration issues?**

**A:**  Refer to [EasyR1](https://github.com/hiyouga/EasyR1/tree/main) for setup guidance or use their Docker environment.

**Q: Where are the training logs and checkpoints saved?**

**A:** Generated data, including logs, datasets, and model checkpoints, are saved in the `STORAGE_PATH` directory and also sent to Hugging Face via `HUGGINGFACENAME`.

**Q: What if the code gets stuck during questioner training?**

**A:**  This may be due to a bug in `math_verify`. Restart training from the last saved checkpoint.

## Acknowledgements

R-Zero is built upon the work of [**EasyR1**](https://github.com/hiyouga/EasyR1/tree/main) and referenced [**General-Reasoner**](https://github.com/TIGER-AI-Lab/General-Reasoner) for evaluation.

## Citation

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