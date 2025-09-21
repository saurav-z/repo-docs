# R-Zero: Revolutionizing LLM Reasoning with Self-Evolution

**R-Zero empowers Large Language Models to autonomously improve their reasoning abilities from scratch, without requiring any pre-existing data or human labels.**  For details, see our [paper](https://arxiv.org/abs/2508.05004) and [webpage](https://chengsong-huang.github.io/R-Zero.github.io/).  ([Original Repo](https://github.com/Chengsong-Huang/R-Zero))

## Key Features

*   **Autonomous Learning:**  R-Zero eliminates the need for labeled data, allowing LLMs to learn reasoning skills independently.
*   **Self-Evolving Framework:**  A Challenger-Solver co-evolution loop fosters continuous improvement and adaptive learning curricula.
*   **Performance Boosts:**  Achieves significant performance gains on various reasoning benchmarks.
*   **Generalization:**  Reasoning skills learned in specific domains effectively transfer to general reasoning tasks.
*   **Model Agnostic:**  R-Zero consistently enhances the performance of various base LLMs.

## Updates

*   **2025-8-27:**  Analysis added on iteration scaling and model role assignment.
*   **2025-8-25:**  Code updated for smoother training (using stopit).
*   **2025-8-8:**  R-Zero recognized as the `#2 Paper of the Day` on [Hugging Face Daily Paper](https://huggingface.co/papers/2508.05004).
*   **2025-8-7:**  Paper and code released.

## Overview

[![R-Zero Overview](./figs/abstract.png)](https://arxiv.org/abs/2508.05004)

Traditional methods of training reasoning models often rely on expensive, human-curated datasets.  R-Zero introduces a novel framework to overcome these limitations, allowing LLMs to autonomously improve their reasoning without any prior tasks or labeled examples.  This self-evolving system starts with nothing but a base model.

At its core, R-Zero establishes a dynamic, co-evolutionary process between two instances of the same base model:

1.  **Challenger 🎯:** Probes the Solver for weaknesses and generates challenging problems at the edge of its capabilities.
2.  **Solver 🧠:** Continuously improves by solving increasingly difficult tasks posed by the Challenger.

This creates a tailored, adaptive curriculum where the Challenger refines its ability to ask questions, and the Solver learns to find better answers.  The entire cycle is self-contained, leveraging techniques like majority voting for pseudo-labels and relative policy optimization to guide learning.

## Quickstart Guide

Get up and running with R-Zero in a few simple steps:

### 1.  Set Up Environment & Directories

```bash
git clone https://github.com/Chengsong-Huang/R-Zero.git

cd R-Zero

pip install -r requirements.txt

export STORAGE_PATH="/path/to/your/storage"  # Set path for checkpoints & data
export HUGGINGFACENAME="yourhuggingfacename" # Your Hugging Face username

mkdir -p \
  "$STORAGE_PATH/evaluation" \
  "$STORAGE_PATH/models" \
  "$STORAGE_PATH/generated_question" \
  "$STORAGE_PATH/temp_results"
```

### 2. Add API Keys

*   Add your Hugging Face and WandB API keys to `tokens.json`.
*   Insert your OpenAI GPT API key into `evaluation/results_recheck.py`.

### 3. Run Experiments

Replicate the experimental results with a single script:

```bash
# Run with a base model name and an abbreviation:
# Format: bash scripts/main.sh [Base_Model_Name] [Abbreviation]

# Example:
bash scripts/main.sh Qwen/Qwen3-4B-Base qwen3-4b
```

## Impressive Results

R-Zero demonstrates significant performance improvements. The table below highlights the performance of various models, comparing the Base Model, a Zero-Shot Challenger baseline, and the iterative R-Zero framework.  Best results are in **bold**.

| Model Name      | Overall AVG | MATH AVG | SuperGPQA | MMLU-Pro | BBEH   |
| :-------------- | :----------: | :------: | :--------: | :------: | :----: |
| ***Qwen3-4B-Base*** |       |      |        |       |      |
| &emsp;Base Model        |    27.10     |   42.58   |    20.88   |  37.38  |  7.57  |
| &emsp;Base Challenger |    30.83     |   44.36   |    24.77   |  47.59  |  6.59  |
| &emsp;R-Zero (Iter 1)  |    34.27     |   48.06   |   **27.92**  |  51.69  |  9.42  |
| &emsp;R-Zero (Iter 2)  |   **34.92**    |   48.44   |    27.72   | **53.75**  |  9.76  |
| &emsp;R-Zero (Iter 3)  |    34.64     |  **49.07**  |    27.55   |  51.53  | **10.42**  |
| ***Qwen3-8B-Base*** |       |      |        |       |      |
| &emsp;Base Model        |    34.49     |   49.18   |    28.33   |  51.80  |  8.63  |
| &emsp;Base Challenger |    36.43     |   51.87   |    30.12   |  54.14  |  9.60  |
| &emsp;R-Zero (Iter 1)  |    37.93     |   53.39   |    31.26   |  57.17  |  9.91  |
| &emsp;R-Zero (Iter 2)  |    38.45     |   53.84   |   **31.58**  | **58.20**  | 10.20  |
| &emsp;R-Zero (Iter 3)  |   **38.73**    |  **54.69**  |    31.38   | **58.23**  | **10.60** |
| ***OctoThinker-3B*** |       |      |        |       |      |
| &emsp;Base Model        |    12.27     |   26.64   |    10.09   |  10.87  |  1.46  |
| &emsp;Base Challenger |    14.41     |   27.51   |    11.19   |  14.53  | **4.40**  |
| &emsp;R-Zero (Iter 1)  |    14.93     |   27.76   |    12.21   |  15.72  |  4.05  |
| &emsp;R-Zero (Iter 2)  |    15.11     |   28.20   |    12.43   |  16.08  |  3.74  |
| &emsp;R-Zero (Iter 3)  |   **15.67**    |  **29.32**  |   **12.44**  | **16.71**  |  4.20  |
| ***OctoThinker-8B*** |       |      |        |       |      |
| &emsp;Base Model        |    16.81     |   32.11   |    13.26   |  20.21  |  1.64  |
| &emsp;Base Challenger |    25.08     |   36.41   |    16.99   |  41.46  |  5.46  |
| &emsp;R-Zero (Iter 1)  |    26.44     |   37.80   |    19.15   |  **42.05** |  6.77  |
| &emsp;R-Zero (Iter 2)  |    26.77     |   38.23   |    19.27   |  41.34  | **8.25**  |
| &emsp;R-Zero (Iter 3)  |   **26.88**    |  **38.52**  |   **19.82**  |  40.92  | **8.25**  |

## FAQ for Developers

### Q: What is the hardware setup for experiments?

**A:**  Experiments were conducted on an 8-GPU server.  Modify the code to accommodate larger models or different hardware setups.

### Q:  What if environment configuration issues arise?

**A:** The framework is inspired by [EasyR1](https://github.com/hiyouga/EasyR1/tree/main).  Refer to their setup instructions or use their Docker environment for assistance.

### Q:  Where are training logs and checkpoints saved?

**A:**  Training logs, datasets, and model checkpoints are saved in the directory specified by the `STORAGE_PATH` environment variable. Datasets are also sent to Hugging Face via `HUGGINGFACENAME`.

### Q:  What if the code gets stuck during questioner training?

**A:**  This could be caused by a bug in the `math_verify` library. Restart training from the last saved checkpoint if this occurs.

## Acknowledgements

R-Zero builds upon the work of [EasyR1](https://github.com/hiyouga/EasyR1/tree/main) (core functionalities) and references the evaluation methodology from [General-Reasoner](https://github.com/TIGER-AI-Lab/General-Reasoner).

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