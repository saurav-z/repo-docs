# R-Zero: Self-Evolving Reasoning LLM from Zero Data

**R-Zero revolutionizes LLM reasoning by enabling models to autonomously learn and improve from scratch, without requiring any initial training data.**  [Check out the original repo!](https://github.com/Chengsong-Huang/R-Zero)

## Key Features

*   **Fully Autonomous Learning:**  R-Zero starts with no pre-existing datasets or human-labeled examples.
*   **Co-Evolutionary Architecture:** Leverages a Challenger-Solver loop to create a dynamic, adaptive curriculum for continual improvement.
*   **Significant Performance Boosts:** Delivers notable improvements across various reasoning benchmarks.
*   **Strong Generalization:** Reasoning skills learned in specific domains transfer effectively to general reasoning tasks.
*   **Model Agnostic:** Enhances the performance of various LLM backbones.

## Updates

*   **[2025-8-27]** Analysis on iteration scaling and one model taking on two roles added.
*   **[2025-8-25]** Code updated to improve training stability (using `stopit`).
*   **[2025-8-8]** R-Zero was recognized as the `#2 Paper of the day` on [huggingface daily paper](https://huggingface.co/papers/2508.05004).
*   **[2025-8-7]** Paper and code released!

## Overview

[![R-Zero Overview](./figs/abstract.png)](https://arxiv.org/abs/2508.05004)

Training powerful reasoning models typically necessitates extensive human-curated data, which is expensive and difficult to scale. R-Zero is a novel framework that enables LLMs to enhance their reasoning capabilities autonomously, without relying on any pre-existing tasks or labels. It's a self-evolving system designed to learn from the ground up.

R-Zero establishes a dynamic co-evolutionary loop between two instances of the same base model:

1.  **Challenger 🎯:** Generates challenging problems to expose the Solver's weaknesses, pushing it to its limits.
2.  **Solver 🧠:** Continuously improves by tackling increasingly difficult tasks posed by the Challenger.

This process creates a perfectly tailored, adaptive learning curriculum. The Challenger hones its question-asking abilities, while the Solver refines its problem-solving skills. The entire cycle is self-contained, using techniques like majority voting for pseudo-labels and relative policy optimization to guide the learning process.

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

*   Add your Hugging Face and WandB (for logging) API keys to `tokens.json`.
*   Add your OpenAI GPT API key for evaluation in `evaluation/results_recheck.py`.

### 3. Run Experiments

Replicate the results with a single script:

```bash
# The script takes the base model name and an abbreviation as arguments
# The abbreviation is used for creating a directory to save the model.
# Format: bash scripts/main.sh [Base_Model_Name] [Abbreviation]

# Example using Qwen/Qwen3-4B-Base:
bash scripts/main.sh Qwen/Qwen3-4B-Base qwen3-4b
```

## Impressive Results

Performance comparison of the Base Model, a Zero-Shot Challenger baseline, and the iterative R-Zero framework.  Peak performance for each model is highlighted in **bold**.

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

### **Q: What hardware is required for experiments?**

**A:** The experiments were conducted on an 8-GPU server with models that can run on a single GPU (e.g., 4B or 8B). You may need to modify the code for larger models or different hardware setups.

### **Q:  How to handle environment configuration problems during installation?**

**A:** The framework is inspired by [EasyR1](https://github.com/hiyouga/EasyR1/tree/main).  Refer to their setup instructions or their Docker environment for assistance.

### **Q: Where are training logs and model checkpoints saved?**

**A:** All generated data, including logs, datasets, and model checkpoints, are saved in the directory specified by the `STORAGE_PATH` environment variable. Datasets are also uploaded to Hugging Face via `HUGGINGFACENAME`.

### **Q: What if the code gets stuck during the questioner training?**

**A:**  This may be due to a bug in the `math_verify` library.  A timeout control is in place, but if this occurs, restart training from the last saved checkpoint.

## Acknowledgements

This work is built upon the foundational work of [**EasyR1**](https://github.com/hiyouga/EasyR1/tree/main) and the evaluation process referenced the work from [**General-Reasoner**](https://github.com/TIGER-AI-Lab/General-Reasoner).

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