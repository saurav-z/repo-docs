# R-Zero: Revolutionizing Reasoning in LLMs with Self-Evolution

**R-Zero enables Large Language Models to autonomously learn and improve their reasoning abilities from scratch, without requiring any pre-existing datasets.** ([Original Repo](https://github.com/Chengsong-Huang/R-Zero))

## Key Features

*   **Fully Autonomous Learning:** R-Zero eliminates the need for external data, starting with a base LLM and evolving its reasoning capabilities through self-generated challenges.
*   **Co-Evolutionary Loop:**  A "Challenger" and "Solver" dynamic creates a targeted, adaptive curriculum, pushing the model to continuously improve.
*   **Performance Boosts:** Demonstrates significant improvements in reasoning benchmark performance across various LLM backbones.
*   **Strong Generalization:**  Reasoning skills learned on specific domains successfully transfer to more general reasoning tasks.
*   **Model-Agnostic:**  R-Zero enhances the reasoning capabilities of a wide range of LLM architectures.

##  Updates

*   [2025-8-27] Analysis of iteration scaling and one model taking on two roles added.
*   [2025-8-25] Code updated for smoother training (using `stopit`).
*   [2025-8-8]  R-Zero recognized as `#2 Paper of the day` on Hugging Face's daily paper list.
*   [2025-8-7]  Paper and code released.

## Overview

![Abstract](./figs/abstract.png)

R-Zero offers a novel approach to training reasoning LLMs, bypassing the traditional reliance on costly and extensive human-curated datasets.  This framework empowers LLMs to self-improve their reasoning skills from the ground up.

The core of R-Zero lies in a co-evolutionary loop that leverages two instances of the same base LLM:

1.  **Challenger:**  Generates challenging problems designed to expose the Solver's weaknesses and push its limits.
2.  **Solver:**  Focuses on solving the increasingly difficult problems posed by the Challenger, leading to continuous improvement.

This cyclical process generates a tailored, adaptive curriculum for learning. The Challenger becomes better at questioning, and the Solver becomes more proficient at answering, all within a self-contained learning environment.

## Quickstart Guide

Follow these steps to get started:

### 1. Set Up Environment and Directories

```bash
git clone https://github.com/Chengsong-Huang/R-Zero.git
cd R-Zero
pip install -r requirements.txt

export STORAGE_PATH="/path/to/your/storage"
export HUGGINGFACENAME="yourhuggingfacename"

mkdir -p "$STORAGE_PATH/evaluation" "$STORAGE_PATH/models" "$STORAGE_PATH/generated_question" "$STORAGE_PATH/temp_results"
```

### 2. Add API Keys

*   Add your Hugging Face and WandB API keys to `tokens.json`.
*   Insert your OpenAI GPT API key for evaluation in `evaluation/results_recheck.py`.

### 3. Run Experiments

Replicate the experiments with a single script:

```bash
# Example using Qwen/Qwen3-4B-Base:
bash scripts/main.sh Qwen/Qwen3-4B-Base qwen3-4b
```

## Impressive Results

R-Zero demonstrates substantial performance improvements across various reasoning benchmarks.  See the detailed results table below:

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

**Q: What is the hardware setup for the experiments?**

**A:** Experiments were conducted on an 8-GPU server using models suitable for single-GPU operation (e.g., 4B or 8B). Modifications may be needed for larger models or different hardware.

**Q: What if I encounter environment configuration issues during installation?**

**A:** This framework is inspired by [EasyR1](https://github.com/hiyouga/EasyR1/tree/main). Refer to their setup instructions or Docker environment as a reference.

**Q: Where are the training logs and model checkpoints saved?**

**A:**  All generated data, including logs, datasets, and model checkpoints, is saved in the directory specified by the `STORAGE_PATH` environment variable, and datasets are sent to Hugging Face via the `HUGGINGFACENAME` variable.

**Q: What if the code gets stuck during the questioner training process?**

**A:**  A potential issue may stem from a bug in the `math_verify` lib, which may cause infinite loops.  A timeout control is in place, but restarting training from the last saved checkpoint is the recommended solution if this issue arises.

## Acknowledgements

This framework is based on the work of [EasyR1](https://github.com/hiyouga/EasyR1/tree/main). The evaluation process referenced work from [General-Reasoner](https://github.com/TIGER-AI-Lab/General-Reasoner).

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
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Chengsong-Huang/R-Zero&type=Date)](https://star-history.com/#Chengsong-Huang/R-Zero&Date)