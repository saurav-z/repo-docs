# R-Zero: Revolutionizing Reasoning with Self-Evolving LLMs

**R-Zero empowers Large Language Models to learn and evolve their reasoning abilities autonomously, starting from scratch without any pre-existing data.**  Check out our paper and learn more on our [webpage](https://chengsong-huang.github.io/R-Zero.github.io/).  For the latest updates, visit the original repository at [https://github.com/Chengsong-Huang/R-Zero](https://github.com/Chengsong-Huang/R-Zero).

## Key Features

*   **Zero-Shot Learning:**  No pre-existing datasets or labeled examples required.
*   **Autonomous Evolution:**  Leverages a dynamic co-evolutionary loop between a Challenger and a Solver.
*   **Adaptive Curriculum:**  The Challenger generates progressively challenging tasks tailored for the Solver.
*   **Proven Performance:**  Achieves significant reasoning performance improvements across various benchmarks.
*   **Generalization:**  Reasoning skills learned transfer effectively to broader reasoning tasks.
*   **Model Agnostic:**  Enhances the capabilities of different backbone LLMs.

## Updates

*   **2025-08-08:**  R-Zero was featured as the #2 Paper of the day on [Hugging Face Papers](https://huggingface.co/papers/2508.05004).
*   **2025-08-07:**  Paper and code released, sparking interest in the LLM community.
<!-- - [2025-8-12] Update codes to make training more smooth. -->

## Overview

![Abstract](figs/abstract.png)

Traditional methods for training powerful reasoning models rely on extensive, manually curated datasets, which are costly and difficult to scale. R-Zero introduces a groundbreaking framework that allows LLMs to self-improve their reasoning skills without any pre-existing tasks or labeled data.

R-Zero uses a co-evolutionary loop between two components:

*   **Challenger 🎯:**  Probes the Solver for weaknesses and creates challenging problems.
*   **Solver 🧠:**  Continuously improves by solving the increasingly difficult tasks from the Challenger.

This creates a self-contained, adaptive curriculum where the Challenger asks better questions and the Solver learns better answers.  The system employs techniques like majority voting for pseudo-labels and relative policy optimization to guide learning.

## Quickstart Guide

Get up and running quickly with these steps:

### 1.  Environment Setup

```bash
git clone https://github.com/Chengsong-Huang/R-Zero.git
cd R-Zero
pip install -r requirements.txt
export STORAGE_PATH="/path/to/your/storage"  # Set a path for checkpoints and generated data
export HUGGINGFACENAME="yourhuggingfacename" # Set the Hugging Face name
mkdir -p \
  "$STORAGE_PATH/evaluation" \
  "$STORAGE_PATH/models" \
  "$STORAGE_PATH/generated_question" \
  "$STORAGE_PATH/temp_results"
```

### 2.  API Key Configuration

*   Add your Hugging Face and WandB API keys to `tokens.json`.
*   Add your OpenAI GPT API key to `evaluation/results_recheck.py` for evaluation.

### 3.  Run the Experiments

Replicate our experimental results with a single script:

```bash
# Run with: bash scripts/main.sh [Base_Model_Name] [Abbreviation]
# Example: bash scripts/main.sh Qwen/Qwen3-4B-Base qwen3-4b
```

## Results

Performance comparison between the Base Model, a Zero-Shot Challenger baseline, and R-Zero iterations. Best performance highlighted in **bold**.

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

## Frequently Asked Questions (FAQ)

### Q: What hardware is required for experiments?

**A:** Experiments were performed on an 8-GPU server using models that fit on a single GPU (e.g., 4B or 8B). You may need to adjust the code for larger models or different hardware.

### Q:  How do I resolve environment configuration issues?

**A:** R-Zero's structure is based on [EasyR1](https://github.com/hiyouga/EasyR1/tree/main). Consult their setup instructions or use their Docker environment for guidance.

### Q: Where are training logs and model checkpoints saved?

**A:** All data, including logs, datasets, and checkpoints, is saved in the directory set by the `STORAGE_PATH` environment variable. Datasets are also sent to Hugging Face via `HUGGINGFACENAME`.

### Q: What if the code gets stuck during training?

**A:**  A timeout has been added to `math_verify` to handle potential infinite loops. If problems persist, restart training from the latest checkpoint.

## Acknowledgements

This framework is built upon the foundational work of [**EasyR1**](https://github.com/hiyouga/EasyR1/tree/main) and leverages components from [**General-Reasoner**](https://github.com/TIGER-AI-Lab/General-Reasoner) for evaluation.

## Citation

If you find this work helpful, please cite our paper:

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