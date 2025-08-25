# R-Zero: Revolutionizing Reasoning in LLMs with Self-Evolving Intelligence

**R-Zero is a groundbreaking framework that enables Large Language Models (LLMs) to autonomously improve their reasoning abilities from scratch, without any pre-existing data.** ([Original Repo](https://github.com/Chengsong-Huang/R-Zero))

## Key Features

*   ✅ **Autonomous Learning:** R-Zero starts with a base model and requires no external datasets or human-labeled examples.
*   🔄 **Co-Evolutionary Loop:**  A "Challenger" model generates increasingly complex problems, driving the "Solver" model to continually enhance its reasoning skills.
*   📈 **Performance Boosts:** Demonstrates significant improvements on various reasoning benchmarks.
*   🌐 **Strong Generalization:** Reasoning skills learned in specific domains effectively transfer to broader reasoning tasks.
*   ⚙️ **Model-Agnostic:** R-Zero enhances the performance of diverse base LLMs.

## 🚀 Updates

*   [2025-08-12] Updated codes to improve training smoothness.
*   [2025-08-08]  R-Zero was recognized as `#2 Paper of the day` on [huggingface daily paper](https://huggingface.co/papers/2508.05004).
*   [2025-08-07]  Released our [paper](https://arxiv.org/abs/2508.05004) and code.

## 🧐 Overview

[![R-Zero Overview](figs/abstract.png)](https://arxiv.org/abs/2508.05004)

Traditional methods for training powerful reasoning models rely on large, manually curated datasets, which are costly and difficult to scale. R-Zero offers a novel approach: a self-evolving system that empowers LLMs to learn and improve their reasoning abilities autonomously.

R-Zero leverages a dynamic co-evolutionary loop between two instances of the same base model:

1.  **Challenger 🎯:**  Probes the Solver for weaknesses, creating challenging problems that push its boundaries.
2.  **Solver 🧠:**  Continuously improves by solving the increasingly difficult tasks posed by the Challenger.

This process creates a perfectly tailored, adaptive learning curriculum.  The Challenger learns to ask better questions, and the Solver learns to find better answers, all self-contained and using techniques like majority voting and relative policy optimization.

## 🏁 Quickstart Guide

Get started with R-Zero quickly using these steps:

### 1. Set Up Your Environment

```bash
git clone https://github.com/Chengsong-Huang/R-Zero.git
cd R-Zero
pip install -r requirements.txt
export STORAGE_PATH="/path/to/your/storage"  # Set your storage directory
export HUGGINGFACENAME="yourhuggingfacename" # Set your Hugging Face username
mkdir -p "$STORAGE_PATH/evaluation" "$STORAGE_PATH/models" "$STORAGE_PATH/generated_question" "$STORAGE_PATH/temp_results"
```

### 2. Configure API Keys

*   Add your **Hugging Face** and **WandB** API keys to `tokens.json`.
*   Add your **OpenAI GPT** API key in `evaluation/results_recheck.py` for evaluation purposes.

### 3. Run Experiments

Replicate our results using a single script:

```bash
# Specify the base model name and an abbreviation
#  e.g., bash scripts/main.sh Qwen/Qwen3-4B-Base qwen3-4b
bash scripts/main.sh [Base_Model_Name] [Abbreviation]
```

## 📊 Impressive Results

The table below summarizes the performance improvements of R-Zero on several reasoning benchmarks:

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

## ❓ FAQ for Developers

### Q: What is the hardware setup for experiments?

**A:** All experiments were conducted on an 8-GPU server using models that can be run on a single GPU (4B or 8B). Adapt the code as needed for larger models or different hardware.

### Q: Environment configuration issues during installation?

**A:**  Refer to the setup instructions or Docker environment of [EasyR1](https://github.com/hiyouga/EasyR1/tree/main) for troubleshooting.

### Q: Where are training logs and model checkpoints saved?

**A:**  Generated data, including logs, datasets, and checkpoints, are stored in the directory specified by the `STORAGE_PATH` environment variable. Datasets will also be sent to Hugging Face via `HUGGINGFACENAME`.

### Q: Code stuck during questioner training?

**A:** This may be due to a bug in the `math_verify` library. Restart training from the last saved checkpoint.

## 🙏 Acknowledgements

This work builds directly upon the foundational research of [**EasyR1**](https://github.com/hiyouga/EasyR1/tree/main) and references the evaluation methods from [**General-Reasoner**](https://github.com/TIGER-AI-Lab/General-Reasoner).  We are grateful for their contributions.

## 💬 Citation

If you find our work valuable, please cite our paper:

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