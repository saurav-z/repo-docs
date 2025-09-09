# R-Zero: Self-Evolving Reasoning LLM from Zero Data

**R-Zero revolutionizes large language models (LLMs) by enabling them to autonomously improve their reasoning abilities without any pre-existing data.** ([Original Repository](https://github.com/Chengsong-Huang/R-Zero))

## Key Features

*   **Autonomous Learning:** R-Zero starts with only a base LLM and learns from scratch, eliminating the need for curated datasets.
*   **Co-Evolutionary Approach:** A "Challenger" model generates increasingly difficult problems, driving the "Solver" model to continuously enhance its reasoning skills.
*   **Adaptive Curriculum:** The Challenger and Solver dynamically adapt to create a perfectly tailored learning path for the LLM.
*   **Strong Performance:** Achieves significant performance improvements on various reasoning benchmarks.
*   **Generalization:** Reasoning abilities learned in specific domains transfer effectively to general reasoning tasks.
*   **Model Agnostic:** R-Zero consistently enhances the performance of different LLM backbones.

## What's New

*   **2025-8-27:** Analysis added for iteration scaling and a single model taking on dual roles.
*   **2025-8-25:** Code updated for smoother training (using stopit).
*   **2025-8-8:** R-Zero recognized as `#2 Paper of the day` on [Hugging Face Papers](https://huggingface.co/papers/2508.05004).
*   **2025-8-7:** Paper and code released.

## Overview

R-Zero introduces a novel framework that allows LLMs to iteratively improve their reasoning abilities without the need for pre-existing task datasets or human-labeled solutions.

<p align="center">
  <img src="./figs/abstract.png" alt="R-Zero Architecture" width="600"/>
</p>

This self-evolving system employs a dynamic co-evolutionary loop:

1.  **Challenger:** Probes the Solver for weaknesses and generates challenging problems at the edge of its capabilities.
2.  **Solver:** Continuously improves by solving the increasingly difficult tasks posed by the Challenger.

This process creates a perfectly tailored, adaptive curriculum. The Challenger learns to ask better questions, and the Solver learns to find better answers. The entire cycle is self-contained, utilizing techniques like majority voting for pseudo-labels and relative policy optimization to guide learning.

## Quickstart Guide

Follow these steps to get started:

### 1.  Set Up the Environment

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

### 2.  Add API Keys

*   Add your API keys for **Hugging Face** and **WandB** (for logging) in `tokens.json`.
*   Add your **OpenAI GPT** API key for evaluation in `evaluation/results_recheck.py`.

### 3.  Run Experiments

Replicate the results using a single script:

```bash
# Format: bash scripts/main.sh [Base_Model_Name] [Abbreviation]
# Example using Qwen/Qwen3-4B-Base:
bash scripts/main.sh Qwen/Qwen3-4B-Base qwen3-4b
```

## Impressive Results

R-Zero significantly improves reasoning performance, as demonstrated in the table below.

| Model Name           | Overall AVG | MATH AVG | SuperGPQA | MMLU-Pro | BBEH  |
| :------------------- | :----------: | :------: | :-------: | :------: | :---: |
| ***Qwen3-4B-Base***  |              |          |           |          |       |
| &emsp;Base Model    |     27.10    |   42.58  |    20.88  |   37.38  |  7.57 |
| &emsp;Base Challenger |     30.83    |   44.36  |    24.77  |   47.59  |  6.59 |
| &emsp;R-Zero (Iter 1) |     34.27    |   48.06  |  **27.92**  |   51.69  |  9.42 |
| &emsp;R-Zero (Iter 2) |   **34.92**  |   48.44  |    27.72  |  **53.75** |  9.76 |
| &emsp;R-Zero (Iter 3) |     34.64    |  **49.07** |    27.55  |   51.53  | **10.42** |
| ***Qwen3-8B-Base***  |              |          |           |          |       |
| &emsp;Base Model    |     34.49    |   49.18  |    28.33  |   51.80  |  8.63 |
| &emsp;Base Challenger |     36.43    |   51.87  |    30.12  |   54.14  |  9.60 |
| &emsp;R-Zero (Iter 1) |     37.93    |   53.39  |    31.26  |   57.17  |  9.91 |
| &emsp;R-Zero (Iter 2) |     38.45    |   53.84  |  **31.58**  |   58.20  | 10.20 |
| &emsp;R-Zero (Iter 3) |   **38.73**  |  **54.69** |    31.38  |  **58.23** | **10.60** |
| ***OctoThinker-3B*** |              |          |           |          |       |
| &emsp;Base Model    |     12.27    |   26.64  |    10.09  |   10.87  |  1.46 |
| &emsp;Base Challenger |     14.41    |   27.51  |    11.19  |   14.53  | **4.40** |
| &emsp;R-Zero (Iter 1) |     14.93    |   27.76  |    12.21  |   15.72  |  4.05 |
| &emsp;R-Zero (Iter 2) |     15.11    |   28.20  |    12.43  |   16.08  |  3.74 |
| &emsp;R-Zero (Iter 3) |   **15.67**  |  **29.32** |  **12.44**  |  **16.71** |  4.20 |
| ***OctoThinker-8B*** |              |          |           |          |       |
| &emsp;Base Model    |     16.81    |   32.11  |    13.26  |   20.21  |  1.64 |
| &emsp;Base Challenger |     25.08    |   36.41  |    16.99  |   41.46  |  5.46 |
| &emsp;R-Zero (Iter 1) |     26.44    |   37.80  |    19.15  |  **42.05** |  6.77 |
| &emsp;R-Zero (Iter 2) |     26.77    |   38.23  |    19.27  |   41.34  | **8.25** |
| &emsp;R-Zero (Iter 3) |   **26.88**  |  **38.52** |  **19.82**  |   40.92  | **8.25** |

## FAQ for Developers

### Q: What hardware is required for experiments?

**A:** Experiments were conducted on an 8-GPU server.  Models can be run on a single GPU (4B or 8B).  Larger models or different hardware will require code modifications.

### Q: How do I resolve environment configuration issues?

**A:** Refer to the [EasyR1](https://github.com/hiyouga/EasyR1/tree/main) setup instructions or use their Docker environment as a reference.

### Q: Where are training logs and model checkpoints saved?

**A:** Saved in the directory specified by the `STORAGE_PATH` environment variable.  Datasets are also sent to Hugging Face via `HUGGINGFACENAME`.

### Q: What if the code gets stuck during the questioner training?

**A:** This may be due to an infinite loop in the `math_verify` library.  Restart training from the last saved checkpoint.

## Acknowledgements

This project is built upon the work of [**EasyR1**](https://github.com/hiyouga/EasyR1/tree/main) and references the evaluation process from [**General-Reasoner**](https://github.com/TIGER-AI-Lab/General-Reasoner).

## Citation

If you find this work useful, please cite our paper:

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