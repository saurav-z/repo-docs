# R-Zero: Revolutionizing Reasoning LLMs with Self-Evolution (No Data Required)

**R-Zero empowers Large Language Models to autonomously learn and improve their reasoning abilities from scratch, eliminating the need for pre-existing data.**  [Explore the R-Zero Repository](https://github.com/Chengsong-Huang/R-Zero)

## Key Features

*   **Zero-Shot Learning:** Trains reasoning models without any pre-existing datasets or human-labeled examples.
*   **Autonomous Evolution:** Leverages a self-evolving Challenger-Solver loop for continuous improvement.
*   **Adaptive Curriculum:** The Challenger dynamically generates increasingly challenging problems, tailoring the learning process.
*   **Proven Performance:** Achieves significant performance gains on diverse reasoning benchmarks.
*   **Generalization:** Demonstrates strong reasoning skills transferable across different domains.
*   **Model Agnostic:** Works effectively with various underlying LLM architectures.

## Updates

*   [2025-08-12] Updated code for smoother training.
*   [2025-08-08] Featured as "#2 Paper of the day" on [Hugging Face Daily Paper](https://huggingface.co/papers/2508.05004).
*   [2025-08-07] Released the [R-Zero paper](https://arxiv.org/abs/2508.05004) and code.

## Overview

R-Zero introduces a novel framework to train powerful reasoning LLMs without relying on expensive, hand-curated datasets. This approach leverages a dynamic, co-evolutionary system composed of two key components:

*   **The Challenger:**  Identifies weaknesses in the Solver and generates challenging problems.
*   **The Solver:**  Continuously improves its reasoning capabilities by solving the problems posed by the Challenger.

This iterative process creates an adaptive curriculum where the Challenger and Solver refine their abilities, leading to significant performance improvements.  The system uses techniques like majority voting for pseudo-labeling and relative policy optimization to guide the learning process, ensuring self-contained and autonomous development.

*Abstract Image Here*

## Quickstart Guide

Get started with R-Zero in a few simple steps:

### 1.  Environment Setup and Directory Configuration

```bash
git clone https://github.com/Chengsong-Huang/R-Zero.git
cd R-Zero
pip install -r requirements.txt

export STORAGE_PATH="/path/to/your/storage" # Set path for checkpoints and data
export HUGGINGFACENAME="yourhuggingfacename" # Set your Hugging Face username

mkdir -p "$STORAGE_PATH/evaluation" "$STORAGE_PATH/models" "$STORAGE_PATH/generated_question" "$STORAGE_PATH/temp_results"
```

### 2.  API Key Configuration

*   Populate your API keys for **Hugging Face** and **WandB** (for logging) in `tokens.json`.
*   Add your **OpenAI GPT** API key in `evaluation/results_recheck.py` for evaluation.

### 3.  Run the Experiments

Replicate the results with a single script:

```bash
#  Use the script with the base model name and an abbreviation
#  Abbreviation creates a directory for model saving.
#  Example with Qwen/Qwen3-4B-Base:
bash scripts/main.sh Qwen/Qwen3-4B-Base qwen3-4b
```

## Impressive Results

R-Zero demonstrates significant performance improvements across various reasoning tasks.  See the performance comparison:

| Model Name             | Overall AVG | MATH AVG | SuperGPQA | MMLU-Pro | BBEH   |
| ---------------------- | :----------: | :------: | :-------: | :------: | :-----: |
| ***Qwen3-4B-Base***    |      -       |     -    |     -     |     -    |    -    |
| &emsp;Base Model       |    27.10     |   42.58  |   20.88   |   37.38  |   7.57  |
| &emsp;Base Challenger  |    30.83     |   44.36  |   24.77   |   47.59  |   6.59  |
| &emsp;R-Zero (Iter 1)  |    34.27     |   48.06  |  **27.92** |   51.69  |   9.42  |
| &emsp;R-Zero (Iter 2)  |  **34.92**   |   48.44  |   27.72   |  **53.75** |   9.76  |
| &emsp;R-Zero (Iter 3)  |    34.64     |  **49.07** |   27.55   |   51.53  |  **10.42** |
| ***Qwen3-8B-Base***    |      -       |     -    |     -     |     -    |    -    |
| &emsp;Base Model       |    34.49     |   49.18  |   28.33   |   51.80  |   8.63  |
| &emsp;Base Challenger  |    36.43     |   51.87  |   30.12   |   54.14  |   9.60  |
| &emsp;R-Zero (Iter 1)  |    37.93     |   53.39  |   31.26   |   57.17  |   9.91  |
| &emsp;R-Zero (Iter 2)  |    38.45     |   53.84  |  **31.58** |   58.20  |  10.20  |
| &emsp;R-Zero (Iter 3)  |  **38.73**   |  **54.69** |   31.38   |  **58.23** |  **10.60** |
| ***OctoThinker-3B***   |      -       |     -    |     -     |     -    |    -    |
| &emsp;Base Model       |    12.27     |   26.64  |   10.09   |   10.87  |   1.46  |
| &emsp;Base Challenger  |    14.41     |   27.51  |   11.19   |   14.53  |  **4.40** |
| &emsp;R-Zero (Iter 1)  |    14.93     |   27.76  |   12.21   |   15.72  |   4.05  |
| &emsp;R-Zero (Iter 2)  |    15.11     |   28.20  |   12.43   |   16.08  |   3.74  |
| &emsp;R-Zero (Iter 3)  |  **15.67**   |  **29.32** |  **12.44** |  **16.71** |   4.20  |
| ***OctoThinker-8B***   |      -       |     -    |     -     |     -    |    -    |
| &emsp;Base Model       |    16.81     |   32.11  |   13.26   |   20.21  |   1.64  |
| &emsp;Base Challenger  |    25.08     |   36.41  |   16.99   |   41.46  |   5.46  |
| &emsp;R-Zero (Iter 1)  |    26.44     |   37.80  |   19.15   |  **42.05** |   6.77  |
| &emsp;R-Zero (Iter 2)  |    26.77     |   38.23  |   19.27   |   41.34  |  **8.25** |
| &emsp;R-Zero (Iter 3)  |  **26.88**   |  **38.52** |  **19.82** |   40.92  |  **8.25** |

## FAQ for Developers

### Q: Hardware Setup for Experiments?
**A:** Experiments were performed on an 8-GPU server.  Adapt code as needed for larger models or different hardware.

### Q: Troubleshooting Environment Configuration?
**A:** Inspired by [EasyR1](https://github.com/hiyouga/EasyR1/tree/main).  Refer to their setup or Docker environment if you encounter issues.

### Q: Where are logs and checkpoints saved?
**A:** The `STORAGE_PATH` environment variable dictates the location for all generated data, including logs, datasets, and model checkpoints. Datasets will also be sent to Hugging Face via `HUGGINGFACENAME`.

### Q: Code getting stuck during questioner training?
**A:** Possible issue with `math_verify` lib, use timeout control and restart training from the last checkpoint.  (Note: `timeout_decorator` may help, but not fully tested).

## Acknowledgements

This work builds upon the foundation of [**EasyR1**](https://github.com/hiyouga/EasyR1/tree/main), incorporating its core functionalities.  The evaluation process referenced the work from [**General-Reasoner**](https://github.com/TIGER-AI-Lab/General-Reasoner).

## Citation

If R-Zero is valuable for your work, please cite our paper:

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