# R-Zero: Self-Evolving Reasoning LLM from Zero Data

**R-Zero revolutionizes how Large Language Models learn to reason by enabling them to improve their abilities autonomously, without relying on any pre-existing datasets.**  Explore the details in our [paper](https://arxiv.org/abs/2508.05004) or [webpage](https://chengsong-huang.github.io/R-Zero.github.io/).

[![Star History Chart](https://api.star-history.com/svg?repos=Chengsong-Huang/R-Zero&type=Date)](https://star-history.com/#Chengsong-Huang/R-Zero&Date)

## Key Features

*   **Autonomous Learning:** R-Zero starts with no external data, eliminating the need for pre-existing datasets or human-annotated solutions.
*   **Co-Evolutionary Loop:** A unique Challenger-Solver dynamic creates a targeted, adaptive curriculum for continuous improvement.
*   **Performance Boost:** Demonstrates significant performance gains on various reasoning benchmarks.
*   **Strong Generalization:** Reasoning skills learned on specific domains transfer effectively to general reasoning tasks.
*   **Model-Agnostic:** Improves performance across various base LLMs.

## Updates

*   **[2025-8-12]** Updated codes to make training more smooth.
*   **[2025-8-8]** Featured as `#2 Paper of the day` on [Hugging Face Daily Paper](https://huggingface.co/papers/2508.05004).
*   **[2025-8-7]** Released [paper](https://arxiv.org/abs/2508.05004) and code.

## Overview

![R-Zero Abstract](./figs/abstract.png)

Training reasoning LLMs typically requires vast, curated datasets, which are expensive and hard to scale. R-Zero offers a novel approach by enabling LLMs to enhance their reasoning abilities autonomously, from scratch, without any pre-existing tasks or labels.

R-Zero leverages a co-evolutionary loop between two instances of the same base model:

1.  **Challenger 🎯:** Probes the Solver for weaknesses, generating challenging problems.
2.  **Solver 🧠:** Continuously improves by solving the increasingly difficult tasks posed by the Challenger.

This self-contained process utilizes techniques like majority voting for pseudo-labels and relative policy optimization.

## Quickstart Guide

Get up and running with R-Zero in a few steps:

### 1.  Configure Environment and Prepare Directories

```bash
git clone https://github.com/Chengsong-Huang/R-Zero.git
cd R-Zero
pip install -r requirements.txt
export STORAGE_PATH="/path/to/your/storage"  # Set your storage path
export HUGGINGFACENAME="yourhuggingfacename"  # Set your Hugging Face name

mkdir -p "$STORAGE_PATH/evaluation" "$STORAGE_PATH/models" "$STORAGE_PATH/generated_question" "$STORAGE_PATH/temp_results"
```

### 2. Add API Keys

Provide necessary API keys in `tokens.json` (Hugging Face & WandB) and `evaluation/results_recheck.py` (OpenAI GPT).

### 3. Run Experiments

Replicate our results with a single script:

```bash
# Format: bash scripts/main.sh [Base_Model_Name] [Abbreviation]
bash scripts/main.sh Qwen/Qwen3-4B-Base qwen3-4b # Example
```

## Impressive Results

R-Zero delivers significant performance improvements.  See the table below for comparative results:

| Model Name          | Overall AVG | MATH AVG | SuperGPQA | MMLU-Pro | BBEH   |
| :------------------ | :----------: | :------: | :-------: | :------: | :-----: |
| ***Qwen3-4B-Base*** |              |          |           |          |         |
| &emsp;Base Model    |    27.10     |  42.58   |   20.88   |  37.38   |  7.57   |
| &emsp;Base Challenger |    30.83     |  44.36   |   24.77   |  47.59   |  6.59   |
| &emsp;R-Zero (Iter 1) |    34.27     |  48.06   |  **27.92**  |  51.69   |  9.42   |
| &emsp;R-Zero (Iter 2) |  **34.92**   |  48.44   |   27.72   |  **53.75**   |  9.76   |
| &emsp;R-Zero (Iter 3) |    34.64     |  **49.07**   |   27.55   |  51.53   |  **10.42**  |
| ***Qwen3-8B-Base*** |              |          |           |          |         |
| &emsp;Base Model    |    34.49     |  49.18   |   28.33   |  51.80   |  8.63   |
| &emsp;Base Challenger |    36.43     |  51.87   |   30.12   |  54.14   |  9.60   |
| &emsp;R-Zero (Iter 1) |    37.93     |  53.39   |   31.26   |  57.17   |  9.91   |
| &emsp;R-Zero (Iter 2) |    38.45     |  53.84   |  **31.58**  |  **58.20**   |  10.20  |
| &emsp;R-Zero (Iter 3) |  **38.73**   |  **54.69**   |   31.38   |  **58.23**   |  **10.60**  |
| ***OctoThinker-3B*** |              |          |           |          |         |
| &emsp;Base Model    |    12.27     |  26.64   |   10.09   |  10.87   |  1.46   |
| &emsp;Base Challenger |    14.41     |  27.51   |   11.19   |  14.53   |  **4.40**   |
| &emsp;R-Zero (Iter 1) |    14.93     |  27.76   |   12.21   |  15.72   |  4.05   |
| &emsp;R-Zero (Iter 2) |    15.11     |  28.20   |   12.43   |  16.08   |  3.74   |
| &emsp;R-Zero (Iter 3) |  **15.67**   |  **29.32**   |  **12.44**  |  **16.71**   |  4.20   |
| ***OctoThinker-8B*** |              |          |           |          |         |
| &emsp;Base Model    |    16.81     |  32.11   |   13.26   |  20.21   |  1.64   |
| &emsp;Base Challenger |    25.08     |  36.41   |   16.99   |  41.46   |  5.46   |
| &emsp;R-Zero (Iter 1) |    26.44     |  37.80   |   19.15   |  **42.05**   |  6.77   |
| &emsp;R-Zero (Iter 2) |    26.77     |  38.23   |   19.27   |  41.34   |  **8.25**   |
| &emsp;R-Zero (Iter 3) |  **26.88**   |  **38.52**   |  **19.82**  |  40.92   |  **8.25**   |

## FAQ for Developers

### **Q: What is the hardware setup for the experiments?**

**A:** Experiments were conducted on an 8-GPU server. Modify the code accordingly for larger models or different hardware.

### **Q: What if I encounter environment configuration issues?**

**A:** Refer to [EasyR1](https://github.com/hiyouga/EasyR1/tree/main) for setup instructions or use their Docker environment as a reference.

### **Q: Where are the training logs and model checkpoints saved?**

**A:** All data, including logs, datasets, and checkpoints, are saved in the directory specified by the `STORAGE_PATH` environment variable. Datasets are also sent to Hugging Face via `HUGGINGFACENAME`.

### **Q: What if the code gets stuck during the questioner training process?**

**A:** This may be due to a bug in the `math_verify` library. Restart training from the last checkpoint.  We've added timeout controls.

## Acknowledgements

We are grateful for the work of [**EasyR1**](https://github.com/hiyouga/EasyR1/tree/main), from which we implemented core functionalities.  Also, we acknowledge the work from [**General-Reasoner**](https://github.com/TIGER-AI-Lab/General-Reasoner) for their work, which was referenced for evaluation.

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