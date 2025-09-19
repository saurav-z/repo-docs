# R-Zero: Revolutionizing Reasoning in LLMs with Self-Evolution

**R-Zero empowers Large Language Models to learn and evolve their reasoning abilities from scratch, without any pre-existing data.** Explore the original repo [here](https://github.com/Chengsong-Huang/R-Zero).

## Key Features

*   **Autonomous Learning:** R-Zero starts with zero external data, eliminating the need for curated datasets.
*   **Co-Evolutionary Architecture:**  A "Challenger" and "Solver" co-evolve, creating an adaptive curriculum for continuous improvement.
*   **Performance Boost:** Demonstrates significant performance gains on various reasoning benchmarks.
*   **Strong Generalization:** Reasoning skills learned transfer effectively to general reasoning tasks.
*   **Model-Agnostic:** Enhances performance across diverse LLM backbones.

## What's New

*   **[2025-8-27]** Added analysis on iteration scaling and one model taking on two roles.
*   **[2025-8-25]** Updated codes for smoother training (implemented with `stopit`).
*   **[2025-8-8]** Recognized as `#2 Paper of the day` on [Hugging Face Daily Paper](https://huggingface.co/papers/2508.05004).
*   **[2025-8-7]**  Released the [paper](https://arxiv.org/abs/2508.05004) and associated code.

## Overview

[![R-Zero Architecture](./figs/abstract.png)](https://arxiv.org/abs/2508.05004)

Training effective reasoning models often requires vast, curated datasets, which are costly and difficult to scale. R-Zero presents a novel approach, enabling LLMs to enhance their reasoning capabilities autonomously. This framework uses a unique co-evolutionary loop between two instances of the same base model:

1.  **Challenger:** Creates probing questions to identify the Solver's weaknesses, generating challenging problems that test its boundaries.
2.  **Solver:** Continuously improves by solving tasks posed by the Challenger.

This dynamic creates a self-adaptive curriculum. The Challenger learns to ask more effective questions, while the Solver learns to find better answers. This entire process is self-contained and leverages techniques like majority voting for pseudo-labels and relative policy optimization to guide learning.

## Quickstart Guide

Follow these steps to get started:

### 1.  Environment Setup & Directory Preparation

```bash
git clone https://github.com/Chengsong-Huang/R-Zero.git
cd R-Zero
pip install -r requirements.txt

export STORAGE_PATH="/path/to/your/storage" #  Specify your storage path
export HUGGINGFACENAME="yourhuggingfacename" # Specify your Hugging Face name

mkdir -p \
  "$STORAGE_PATH/evaluation" \
  "$STORAGE_PATH/models" \
  "$STORAGE_PATH/generated_question" \
  "$STORAGE_PATH/temp_results"
```

### 2.  API Key Configuration

You'll need to add API keys for Hugging Face and WandB in `tokens.json`, and your OpenAI GPT API key for evaluation in `evaluation/results_recheck.py`.

### 3. Run the Experiments

Replicate the results with a single script:

```bash
# Arguments: Base Model Name, Abbreviation
# Example: bash scripts/main.sh Qwen/Qwen3-4B-Base qwen3-4b
bash scripts/main.sh [Base_Model_Name] [Abbreviation]
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

## Frequently Asked Questions

*   **Hardware Setup:** Experiments were conducted on an 8-GPU server. Adjust code as needed for larger models or different hardware.
*   **Environment Configuration:** Refer to the EasyR1 repository for setup instructions or use their Docker environment if you encounter issues.
*   **Training Logs and Checkpoints:**  Saved in the directory specified by the `STORAGE_PATH` environment variable, with datasets sent to Hugging Face using `HUGGINGFACENAME`.
*   **Code Stalling:**  Potential issue with `math_verify` lib; restart training from the latest checkpoint.

## Acknowledgements

This project is based on the work of [**EasyR1**](https://github.com/hiyouga/EasyR1/tree/main) and uses the evaluation process from [**General-Reasoner**](https://github.com/TIGER-AI-Lab/General-Reasoner).

## Citation

If this work is helpful, please consider citing:

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