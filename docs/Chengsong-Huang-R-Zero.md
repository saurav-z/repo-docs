# R-Zero: Evolving Reasoning LLMs from Zero Data 

**Revolutionize your LLMs with R-Zero, a groundbreaking framework that empowers language models to learn and evolve their reasoning abilities autonomously, without any initial training data.**  [Check out the original repo](https://github.com/Chengsong-Huang/R-Zero) for the code and more details.

## Key Features

*   🧠 **Fully Autonomous Learning:**  R-Zero eliminates the need for pre-existing datasets, allowing LLMs to start their reasoning journey from scratch.
*   🔄 **Co-Evolutionary Architecture:**  Experience a dynamic "Challenger-Solver" loop that creates a tailored, adaptive curriculum for continuous improvement.
*   📈 **Demonstrated Performance Gains:** Witness significant performance boosts on diverse reasoning benchmarks across various LLMs.
*   🌍 **Generalization Capabilities:** Benefit from reasoning skills that transfer effectively to general reasoning tasks, even when learned in specific domains.
*   ⚙️ **Model Agnostic:** R-Zero consistently improves performance across a range of base LLMs, providing broad applicability.

## Updates

*   [2025-8-12] Updated codes for smoother training.
*   [2025-8-8] R-Zero recognized as `#2 Paper of the day` on [Hugging Face Daily Papers](https://huggingface.co/papers/2508.05004).
*   [2025-8-7] Paper and code released: [Paper](https://arxiv.org/abs/2508.05004)

## Overview

![](./figs/abstract.png)

R-Zero tackles the challenges of training reasoning models by eliminating the reliance on extensive, manually curated datasets. This innovative framework fosters self-improvement in LLMs, using a unique co-evolutionary loop to drive learning.

The core of R-Zero is a continuous interaction between two instances of the same base LLM:

1.  **Challenger:** This module probes the Solver, creating challenging problems designed to uncover its weaknesses.
2.  **Solver:** This module focuses on continuous improvement by solving the increasingly difficult tasks posed by the Challenger.

This process cultivates a highly effective, adaptive learning curriculum. The Challenger gets better at questioning, and the Solver gets better at answering. The system employs self-contained techniques like majority voting for pseudo-labels and relative policy optimization to guide the learning process.

## Quickstart Guide

Get started with R-Zero in a few simple steps:

### 1. Configure Environment & Prepare Directories

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

### 2. Set Up API Keys

Add your API keys for the following services in the specified files:

*   **Hugging Face** and **WandB** (for logging) in `tokens.json`.
*   **OpenAI GPT** (for evaluation) in `evaluation/results_recheck.py`.

### 3. Run Experiments

Replicate our experimental results using the following script:

```bash
# Format: bash scripts/main.sh [Base_Model_Name] [Abbreviation]
# Example:
bash scripts/main.sh Qwen/Qwen3-4B-Base qwen3-4b
```

## Impressive Results

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

### Q: What hardware is required for experiments?

**A:** Experiments were conducted on an 8-GPU server using models that run on a single GPU (4B or 8B). Modifications may be required for larger models or different hardware.

### Q: What do I do if I encounter environment configuration issues?

**A:** The framework is inspired by [EasyR1](https://github.com/hiyouga/EasyR1/tree/main). Refer to their setup instructions or use their Docker environment as a guide.

### Q: Where are the training logs and checkpoints saved?

**A:**  All generated data, including logs, datasets, and checkpoints, are stored in the directory specified by the `STORAGE_PATH` environment variable. Datasets are also sent to Hugging Face via `HUGGINGFACENAME`.

### Q: What if the code gets stuck during the questioner training process?

**A:** A timeout control is in place to mitigate potential infinite loops from the `math_verify` library. Restart training from the last saved checkpoint if issues persist.

## Acknowledgements

This project is built upon the solid foundation of [**EasyR1**](https://github.com/hiyouga/EasyR1/tree/main) and references work from [**General-Reasoner**](https://github.com/TIGER-AI-Lab/General-Reasoner).  We are grateful for their contributions.

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