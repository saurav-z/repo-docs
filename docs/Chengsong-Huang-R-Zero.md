# R-Zero: Unleash Self-Evolving Reasoning in LLMs from Scratch

**R-Zero revolutionizes Large Language Models by enabling them to autonomously improve their reasoning abilities without the need for any pre-existing data or human intervention.**

[Explore the R-Zero Paper](https://arxiv.org/abs/2508.05004) | [Visit the Project Webpage](https://chengsong-huang.github.io/R-Zero.github.io/) | [View on GitHub](https://github.com/Chengsong-Huang/R-Zero)

## Key Features

*   **Zero-Shot Learning:** R-Zero starts with a base LLM and requires no initial training data, problem sets, or human labels.
*   **Self-Evolving Architecture:** Utilizes a dynamic co-evolutionary loop between a "Challenger" and a "Solver" to create a tailored, adaptive learning curriculum.
*   **Autonomous Improvement:**  The Challenger generates progressively more complex problems, driving the Solver to continuously enhance its reasoning capabilities.
*   **Strong Performance:** Achieves significant performance gains on various reasoning benchmarks.
*   **Generalization Capabilities:** Demonstrates effective transfer of learned reasoning skills across different domains.
*   **Model Agnostic:**  Improves the performance of different backbone LLMs.

## Updates

*   **[2025-08-12]:** Codes updated for smoother training.
*   **[2025-08-08]:**  R-Zero featured as `#2 Paper of the Day` on [Hugging Face Papers](https://huggingface.co/papers/2508.05004).
*   **[2025-08-07]:** Paper and code released.

## Overview

![R-Zero Abstract](./figs/abstract.png)

Traditional methods for training powerful reasoning models rely on extensive, curated datasets, which are costly and challenging to scale. R-Zero provides a novel framework that allows LLMs to enhance their reasoning capabilities autonomously, without requiring any pre-existing tasks or data. This is a truly self-evolving system that learns from scratch, allowing it to achieve improved performance.

At the core of R-Zero is a dynamic co-evolutionary loop comprising two instances of the same base model:

1.  **The Challenger 🎯:**  Identifies weaknesses in the Solver and generates challenging problems right at the edge of its capabilities.
2.  **The Solver 🧠:**  Continuously improves by solving the progressively challenging tasks presented by the Challenger.

This process forms a perfectly customized, adaptive curriculum. The Challenger learns to pose better questions, and the Solver learns to produce more accurate answers. The entire cycle is self-contained and utilizes techniques like majority voting for pseudo-labels and relative policy optimization to guide the learning process.

## Quickstart Guide

Get started with R-Zero in a few easy steps:

### 1. Set Up Environment & Directories

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

### 2. Configure API Keys

*   Add your **Hugging Face** and **WandB** API keys in `tokens.json`.
*   Add your **OpenAI GPT** API key in `evaluation/results_recheck.py`.

### 3. Run Experiments

Replicate the experiments with a single script:

```bash
# bash scripts/main.sh [Base_Model_Name] [Abbreviation]
# Example:
bash scripts/main.sh Qwen/Qwen3-4B-Base qwen3-4b
```

## Results

The following table presents a comparison of Base Models, Zero-Shot Challenger Baselines, and the iterative R-Zero framework. The highest performance for each model is highlighted in **bold**.

| Model Name        | Overall AVG | MATH AVG | SuperGPQA | MMLU-Pro | BBEH  |
|-------------------|-------------|----------|-----------|----------|-------|
| ***Qwen3-4B-Base*** |             |          |           |          |       |
| &emsp;Base Model  | 27.10       | 42.58    | 20.88     | 37.38    | 7.57  |
| &emsp;Base Challenger | 30.83       | 44.36    | 24.77     | 47.59    | 6.59  |
| &emsp;R-Zero (Iter 1)  | 34.27       | 48.06    | **27.92** | 51.69    | 9.42  |
| &emsp;R-Zero (Iter 2)  | **34.92**   | 48.44    | 27.72     | **53.75** | 9.76  |
| &emsp;R-Zero (Iter 3)  | 34.64       | **49.07**   | 27.55     | 51.53    | **10.42** |
| ***Qwen3-8B-Base*** |             |          |           |          |       |
| &emsp;Base Model  | 34.49       | 49.18    | 28.33     | 51.80    | 8.63  |
| &emsp;Base Challenger | 36.43       | 51.87    | 30.12     | 54.14    | 9.60  |
| &emsp;R-Zero (Iter 1)  | 37.93       | 53.39    | 31.26     | 57.17    | 9.91  |
| &emsp;R-Zero (Iter 2)  | 38.45       | 53.84    | **31.58** | 58.20    | 10.20 |
| &emsp;R-Zero (Iter 3)  | **38.73**   | **54.69**   | 31.38     | **58.23** | **10.60** |
| ***OctoThinker-3B*** |             |          |           |          |       |
| &emsp;Base Model  | 12.27       | 26.64    | 10.09     | 10.87    | 1.46  |
| &emsp;Base Challenger | 14.41       | 27.51    | 11.19     | 14.53    | **4.40** |
| &emsp;R-Zero (Iter 1)  | 14.93       | 27.76    | 12.21     | 15.72    | 4.05  |
| &emsp;R-Zero (Iter 2)  | 15.11       | 28.20    | 12.43     | 16.08    | 3.74  |
| &emsp;R-Zero (Iter 3)  | **15.67**   | **29.32**   | **12.44** | **16.71** | 4.20  |
| ***OctoThinker-8B*** |             |          |           |          |       |
| &emsp;Base Model  | 16.81       | 32.11    | 13.26     | 20.21    | 1.64  |
| &emsp;Base Challenger | 25.08       | 36.41    | 16.99     | 41.46    | 5.46  |
| &emsp;R-Zero (Iter 1)  | 26.44       | 37.80    | 19.15     | **42.05** | 6.77  |
| &emsp;R-Zero (Iter 2)  | 26.77       | 38.23    | 19.27     | 41.34    | **8.25** |
| &emsp;R-Zero (Iter 3)  | **26.88**   | **38.52**   | **19.82** | 40.92    | **8.25** |

## Frequently Asked Questions (FAQ)

### Q: What is the hardware setup for the experiments?

**A:** All experiments were conducted on an 8-GPU server. The models used are capable of running on a single GPU (e.g., 4B or 8B). If you need to run experiments with larger models or on different hardware, you will need to modify the code accordingly.

### Q: What should I do if I encounter environment configuration issues during installation?

**A:**  Our framework's structure is inspired by [EasyR1](https://github.com/hiyouga/EasyR1/tree/main). If you have environment-related problems, we suggest checking their setup instructions or using their Docker environment.

### Q: Where are training logs and model checkpoints saved?

**A:** All generated data, including logs, datasets, and model checkpoints, will be saved in the directory specified by the `STORAGE_PATH` environment variable. Datasets are also uploaded to Hugging Face via `HUGGINGFACENAME`.

### Q: What if the code gets stuck during the questioner training process?

**A:** This may be caused by a bug in the `math_verify` library, which can lead to an infinite loop when processing some answers. A timeout control has been added to mitigate this, but it may not catch every case. If you encounter this issue, restart training from the latest saved checkpoint.
>> I suddenly find there is a lib named `timeout_decorator` which can solve this problem after I complete most of the experiments...... (not sure whether it will introduce new problems.)

## Acknowledgements

This framework builds upon the excellent work of [**EasyR1**](https://github.com/hiyouga/EasyR1/tree/main), implementing all of its core functionalities. In addition, the evaluation process references the work from [**General-Reasoner**](https://github.com/TIGER-AI-Lab/General-Reasoner). We are grateful for their contributions.

## Citation

If this work proves useful, please consider citing our paper:

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