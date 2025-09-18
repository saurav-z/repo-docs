# R-Zero: Self-Evolving Reasoning for LLMs

**R-Zero empowers Large Language Models to reason and evolve autonomously, achieving remarkable results without any initial training data.** Check out the original repo [here](https://github.com/Chengsong-Huang/R-Zero).

## Key Features

*   **Autonomous Learning:** R-Zero starts with a base model and learns to reason without any pre-existing datasets or labels.
*   **Co-Evolutionary Architecture:**  A Challenger-Solver loop continuously improves the model's reasoning abilities by generating and solving increasingly complex tasks.
*   **Proven Performance:**  Demonstrates significant performance boosts on reasoning benchmarks.
*   **Strong Generalization:** Reasoning skills learned on specific domains transfer effectively to general reasoning tasks.
*   **Model-Agnostic:** Compatible with various backbone LLMs, enhancing their performance.

## Updates

*   **2025-8-27:** Analysis on iteration scaling and a single model taking on dual roles.
*   **2025-8-25:** Code updates for smoother training (by `stopit`).
*   **2025-8-8:** R-Zero recognized as `#2 Paper of the day` on [huggingface daily paper](https://huggingface.co/papers/2508.05004).
*   **2025-8-7:** Paper and code released.

## Overview

R-Zero revolutionizes LLM training by enabling self-improvement through a novel, self-contained framework.  Instead of relying on large, manually curated datasets, R-Zero utilizes a dynamic co-evolutionary loop between a **Challenger** and a **Solver**.

*   **Challenger:**  Probes the Solver, generating challenging problems to expose weaknesses and pushing the model to its limits.
*   **Solver:**  Continuously improves by solving the problems posed by the Challenger.

This iterative process, leveraging techniques like majority voting for pseudo-labels and relative policy optimization, creates a self-adaptive curriculum for continuous learning.

## Quickstart Guide

Get started with R-Zero in a few simple steps:

### 1. Configure Environment and Prepare Directories

```bash
git clone https://github.com/Chengsong-Huang/R-Zero.git

# Navigate into the new directory
cd R-Zero
# Install the required packages
pip install -r requirements.txt
# Set environment variables
export STORAGE_PATH="/path/to/your/storage"
export HUGGINGFACENAME="yourhuggingfacename"

mkdir -p \
  "$STORAGE_PATH/evaluation" \
  "$STORAGE_PATH/models" \
  "$STORAGE_PATH/generated_question" \
  "$STORAGE_PATH/temp_results"
```

### 2. Add API Keys

*   Add your Hugging Face and WandB API keys to `tokens.json`.
*   Add your OpenAI GPT API key to `evaluation/results_recheck.py` for evaluation.

### 3. Run the Experiments

Replicate the experimental results using a single script:

```bash
# The script takes the base model name and an abbreviation as arguments
# The abbreviation is used for creating a directory to save the model.
# Format: bash scripts/main.sh [Base_Model_Name] [Abbreviation]

# Example using Qwen/Qwen3-4B-Base:
bash scripts/main.sh Qwen/Qwen3-4B-Base qwen3-4b
```

## Impressive Results

R-Zero significantly improves the performance of various base models.  Here's a comparison of results, with peak performance highlighted in **bold**:

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

**Q: What hardware is required for the experiments?**

**A:** Experiments were conducted on an 8-GPU server. Models that can run on a single GPU (e.g., 4B or 8B) are recommended. Adjustments to the code might be needed for larger models or different hardware.

**Q:  What to do if you encounter environment configuration issues?**

**A:**  The framework is inspired by [EasyR1](https://github.com/hiyouga/EasyR1/tree/main). Refer to their setup instructions or Docker environment for assistance.

**Q:  Where are training logs and checkpoints saved?**

**A:** All generated data, including logs, datasets, and model checkpoints, will be saved in the directory set by the `STORAGE_PATH` environment variable. Datasets will be sent to huggingface via `HUGGINGFACENAME`.

**Q:  What if the code gets stuck during the questioner training process?**

**A:** A timeout control mitigates issues related to the `math_verify` lib. Restart training from the last saved checkpoint if needed.
<!-- >> I suddenly find there is a lib named `timeout_decorator` which can solve this problem after I complete most of the experiments...... (not sure whether it will introduce new problems.) -->

## Acknowledgements

This work builds upon the foundation of [**EasyR1**](https://github.com/hiyouga/EasyR1/tree/main) and references the evaluation process from [**General-Reasoner**](https://github.com/TIGER-AI-Lab/General-Reasoner).

## Citation

If R-Zero is helpful for your work, please cite our paper:

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