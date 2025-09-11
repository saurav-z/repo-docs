# R-Zero: Self-Evolving Reasoning LLM from Zero Data

**R-Zero revolutionizes Large Language Models by enabling them to autonomously develop reasoning skills from scratch, without any pre-existing data.**

[View the original repository on GitHub](https://github.com/Chengsong-Huang/R-Zero)

## Key Features

*   **Autonomous Learning:** Trains LLMs from a blank slate, requiring no initial datasets or human annotations.
*   **Co-Evolutionary Loop:**  Employs a Challenger-Solver dynamic for continuous, adaptive learning. The Challenger generates challenging problems, and the Solver improves by solving them.
*   **Superior Performance:** Achieves significant performance improvements across multiple reasoning benchmarks.
*   **Strong Generalization:** Reasoning skills learned on specific domains transfer effectively to general reasoning tasks.
*   **Model-Agnostic:** Enhances the reasoning capabilities of various base LLMs.

## 🔥 What's New

*   **[2025-8-27]** Analysis on iteration scaling and one model taking on two roles added.
*   **[2025-8-25]** Code updates to improve training smoothness (using stopit).
*   **[2025-8-8]** R-Zero recognized as `#2 Paper of the day` on [huggingface daily paper](https://huggingface.co/papers/2508.05004).
*   **[2025-8-7]** Paper and code released.

## 🧠 Overview

R-Zero is a groundbreaking framework that empowers LLMs to self-improve their reasoning abilities. Unlike traditional methods that rely on large, curated datasets, R-Zero starts with a base model and fosters a dynamic co-evolution between two key components:

*   **The Challenger:**  Generates challenging problems to push the Solver to its limits.
*   **The Solver:**  Continuously improves by solving the problems posed by the Challenger.

This iterative process creates a tailored learning curriculum.  The Challenger learns to ask more effective questions, and the Solver learns to provide more accurate answers. R-Zero utilizes techniques like majority voting and relative policy optimization within a self-contained loop, eliminating the need for external data.

[![](./figs/abstract.png)](https://arxiv.org/abs/2508.05004)

## 🚀 Quickstart Guide

Get started with R-Zero in a few simple steps:

### 1. Set Up Your Environment

```bash
git clone https://github.com/Chengsong-Huang/R-Zero.git
cd R-Zero
pip install -r requirements.txt
export STORAGE_PATH="/path/to/your/storage" # Set your storage directory
export HUGGINGFACENAME="yourhuggingfacename" # Set your Hugging Face username

mkdir -p "$STORAGE_PATH/evaluation" "$STORAGE_PATH/models" "$STORAGE_PATH/generated_question" "$STORAGE_PATH/temp_results"
```

### 2.  Configure API Keys

*   Add your Hugging Face and WandB API keys in `tokens.json`.
*   Provide your OpenAI GPT API key in `evaluation/results_recheck.py` for evaluation.

### 3. Run the Experiments

Replicate the experiments with this script:

```bash
# Replace with your base model name and abbreviation
bash scripts/main.sh [Base_Model_Name] [Abbreviation]

# Example:
bash scripts/main.sh Qwen/Qwen3-4B-Base qwen3-4b
```

## 📈 Results

| Model Name | Overall AVG | MATH AVG | SuperGPQA | MMLU-Pro | BBEH |
|---|:---:|:---:|:---:|:---:|:---:|
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

## ❓ Frequently Asked Questions (FAQ)

### Hardware Setup for Experiments?

**A:** Experiments were performed on an 8-GPU server. You might need to adjust the code for larger models or different hardware.

### Environment Configuration Issues?

**A:**  Refer to the setup instructions or Docker environment of [EasyR1](https://github.com/hiyouga/EasyR1/tree/main) for guidance.

### Training Logs and Checkpoints?

**A:** All generated data is stored in the directory specified by the `STORAGE_PATH` environment variable.  Datasets are sent to Hugging Face via `HUGGINGFACENAME`.

### Code Getting Stuck During Training?

**A:** A timeout mechanism is in place to address potential infinite loops. If the issue persists, restart training from the last saved checkpoint.

## 🙏 Acknowledgements

This project is built upon the foundational work of [**EasyR1**](https://github.com/hiyouga/EasyR1/tree/main) for its core functionality, and leverages evaluation methods from [**General-Reasoner**](https://github.com/TIGER-AI-Lab/General-Reasoner). We are deeply grateful for their contributions.

## 💬 Citation

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