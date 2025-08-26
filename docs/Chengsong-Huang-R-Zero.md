# R-Zero: Revolutionizing LLM Reasoning with Self-Evolving Intelligence

**R-Zero empowers Large Language Models (LLMs) to autonomously learn and improve their reasoning abilities without any pre-existing data, starting from zero.**  Learn more about our innovative approach in our [paper](https://arxiv.org/abs/2508.05004) and on our [webpage](https://chengsong-huang.github.io/R-Zero.github.io/).  ([Back to the Original Repo](https://github.com/Chengsong-Huang/R-Zero))

## Key Features

*   **Zero-Shot Learning:**  No need for labeled datasets or human-annotated examples.
*   **Self-Evolving Architecture:**  Employs a co-evolutionary Challenger-Solver loop for continuous improvement.
*   **Adaptive Curriculum:** The Challenger dynamically generates progressively challenging problems.
*   **High Performance:** Demonstrated significant performance gains on various reasoning benchmarks.
*   **Strong Generalization:** Reasoning skills learned transfer effectively across different domains.
*   **Model Agnostic:** R-Zero improves the performance of various base LLMs.

## Updates
*   **[2025-8-25]** Updated codes for smoother training.
*   **[2025-8-8]** Ranked #2 Paper of the Day on Hugging Face Daily Paper.
*   **[2025-8-7]** Released the research paper and code.

## Overview

![R-Zero Architecture](./figs/abstract.png)

R-Zero provides a novel framework for LLMs to evolve their reasoning abilities autonomously. Unlike traditional methods requiring massive, curated datasets, R-Zero starts with a base model and leverages a dynamic co-evolutionary process:

*   **Challenger:** Probes the Solver for weaknesses and generates challenging problems.
*   **Solver:** Improves by solving the tasks posed by the Challenger.

This iterative process creates a tailored, adaptive curriculum where the Challenger learns to ask better questions, and the Solver learns to provide better answers. Techniques like majority voting and relative policy optimization guide this self-contained learning loop.

## Quickstart Guide

Get up and running with R-Zero in a few easy steps:

### 1. Set Up Your Environment

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

*   Add your Hugging Face and WandB API keys to `tokens.json`.
*   Add your OpenAI GPT API key in `evaluation/results_recheck.py`.

### 3. Run the Experiments

Replicate our results using a single script:

```bash
# Format: bash scripts/main.sh [Base_Model_Name] [Abbreviation]
# Example:
bash scripts/main.sh Qwen/Qwen3-4B-Base qwen3-4b
```

## Results

The table below shows the performance comparison of the base model, a zero-shot challenger baseline, and the R-Zero iterative framework.

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

**Q: What hardware is required for the experiments?**

**A:** Experiments were run on an 8-GPU server. Smaller models can be run on single GPUs.  Adjust the code as needed for larger models or different hardware configurations.

**Q: What if I have environment configuration issues?**

**A:** Refer to the setup instructions or Docker environment from [EasyR1](https://github.com/hiyouga/EasyR1/tree/main), which served as the inspiration for our framework.

**Q: Where are the training logs and checkpoints saved?**

**A:** Checkpoints and all generated data are saved in the directory specified by the `STORAGE_PATH` environment variable.  Datasets will also be sent to Hugging Face via `HUGGINGFACENAME`.

**Q: What if the code gets stuck during questioner training?**

**A:** A timeout has been implemented, but if it continues to get stuck, restart training from the last saved checkpoint.

## Acknowledgements

Our work is heavily based on the advancements from [**EasyR1**](https://github.com/hiyouga/EasyR1/tree/main) and the evaluation methodology of [**General-Reasoner**](https://github.com/TIGER-AI-Lab/General-Reasoner).

## Citation

If you find our work useful, please cite our paper:

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