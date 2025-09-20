# R-Zero: Self-Evolving Reasoning LLM from Zero Data

**R-Zero enables Large Language Models to autonomously evolve their reasoning abilities, starting from scratch without any pre-existing datasets.**

[View the original repository on GitHub](https://github.com/Chengsong-Huang/R-Zero)

## Key Features

*   **Autonomous Learning:** R-Zero requires *no* initial data or human-labeled examples.
*   **Co-Evolutionary Architecture:** Utilizes a Challenger-Solver loop for continuous self-improvement.
*   **Adaptive Curriculum:** The Challenger generates increasingly challenging problems, tailoring the learning process.
*   **Model-Agnostic:** Enhances the reasoning capabilities of various base LLMs.
*   **Strong Performance:** Achieves significant performance gains on reasoning benchmarks.
*   **Generalization:** Reasoning skills learned in specific domains transfer to general reasoning tasks.

## What's New

*   **[2025-8-27]** Analysis on iteration scaling and one model taking on two roles added.
*   **[2025-8-25]** Code updates for smoother training (using stopit).
*   **[2025-8-8]** Recognized as `#2 Paper of the day` on [Hugging Face Papers](https://huggingface.co/papers/2508.05004).
*   **[2025-8-7]** Paper and code released.

## Overview

[![R-Zero Overview](./figs/abstract.png)](https://arxiv.org/abs/2508.05004)

Training reasoning models typically demands massive, curated datasets, which are expensive and challenging to scale. R-Zero addresses this challenge with a novel framework. It allows LLMs to autonomously improve their reasoning skills. The process begins without any pre-existing tasks or labels, creating a self-evolving system that learns from the ground up.

R-Zero's core is a dynamic co-evolutionary loop between two instances of the same base model:

1.  **Challenger 🎯:** Probes the Solver for weaknesses by generating challenging problems.
2.  **Solver 🧠:** Continuously improves by tackling the increasingly complex tasks from the Challenger.

This creates a tailored, adaptive curriculum. The Challenger learns to formulate more complex questions, while the Solver learns to provide improved answers. The entire cycle is self-contained, employing majority voting for pseudo-labels and relative policy optimization.

## Quickstart Guide

### 1. Environment Setup

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

### 2. API Keys

*   Add your API keys for **Hugging Face** and **WandB** in `tokens.json`.
*   Add your **OpenAI GPT** API key in `evaluation/results_recheck.py` for evaluation.

### 3. Run Experiments

Replicate the results using this script:

```bash
# The script takes the base model name and an abbreviation as arguments
# The abbreviation is used for creating a directory to save the model.
# Format: bash scripts/main.sh [Base_Model_Name] [Abbreviation]

# Example using Qwen/Qwen3-4B-Base:
bash scripts/main.sh Qwen/Qwen3-4B-Base qwen3-4b
```

## Impressive Results

*   **R-Zero shows significant improvements over base models and baseline challengers.**

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

### Hardware Setup?

Experiments used an 8-GPU server.  Modify code as needed for different hardware.

### Environment Configuration?

Refer to [EasyR1](https://github.com/hiyouga/EasyR1/tree/main) for setup guidance, or use their Docker environment.

### Where are training logs and model checkpoints saved?

Logs, datasets, and model checkpoints are stored in the `STORAGE_PATH`. Datasets are also sent to Hugging Face under `HUGGINGFACENAME`.

### Code getting stuck?

Restart training from the last saved checkpoint if the questioner training process gets stuck (likely due to an infinite loop in the math\_verify lib).

## Acknowledgements

This project is based on the work of [**EasyR1**](https://github.com/hiyouga/EasyR1/tree/main) and references the work from [**General-Reasoner**](https://github.com/TIGER-AI-Lab/General-Reasoner).

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