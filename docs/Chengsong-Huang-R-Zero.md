# R-Zero: Revolutionizing LLM Reasoning Through Self-Evolution

**R-Zero enables Large Language Models to autonomously evolve their reasoning skills from scratch, without any pre-existing data.**

[View the original repository on GitHub](https://github.com/Chengsong-Huang/R-Zero)

## Key Features

*   ✅ **Fully Autonomous Learning:** R-Zero starts with no external data, eliminating the need for pre-existing problem sets or human-annotated solutions.
*   🔄 **Co-Evolutionary Loop:** A dynamic Challenger-Solver setup creates a targeted and adaptive curriculum for continuous reasoning improvement.
*   📈 **Proven Performance Gains:** Achieve significant performance boosts on various reasoning benchmarks.
*   🌍 **Strong Generalization:** Reasoning skills learned in specific domains readily transfer to broader reasoning tasks.
*   ⚙️ **Model-Agnostic:** R-Zero consistently enhances the performance of different LLM architectures.

## Overview

R-Zero is a groundbreaking framework designed to enhance the reasoning abilities of Large Language Models (LLMs) by enabling them to learn and evolve autonomously.  Unlike traditional methods that rely on massive, curated datasets, R-Zero trains LLMs from a "zero-shot" starting point, requiring no pre-existing tasks or labels.

R-Zero centers around a co-evolutionary loop involving two instances of the same base model:

*   **Challenger:** The Challenger probes the Solver for weaknesses by generating challenging problems designed to push the Solver to its limits.
*   **Solver:** The Solver continuously improves its reasoning abilities by tackling the increasingly complex tasks posed by the Challenger.

This dynamic process creates a highly tailored and adaptive learning experience.  The Challenger learns to formulate better questions, while the Solver hones its ability to find better answers, all within a self-contained system. Techniques like majority voting for pseudo-labels and relative policy optimization guide the learning process.

[![](./figs/abstract.png)](https://arxiv.org/abs/2508.05004)

## Quickstart Guide

Get up and running with R-Zero in a few simple steps:

### 1. Configure Environment and Prepare Directories

```bash
git clone https://github.com/Chengsong-Huang/R-Zero.git
cd R-Zero
pip install -r requirements.txt
export STORAGE_PATH="/path/to/your/storage"  # Set your storage path
export HUGGINGFACENAME="yourhuggingfacename"  # Set your Hugging Face username
mkdir -p "$STORAGE_PATH/evaluation" "$STORAGE_PATH/models" "$STORAGE_PATH/generated_question" "$STORAGE_PATH/temp_results"
```

### 2. Add API Keys

*   Add your Hugging Face and WandB API keys to `tokens.json`.
*   Add your OpenAI GPT API key to `evaluation/results_recheck.py`.

### 3. Run Experiments

Replicate the research results with a single script:

```bash
# Format: bash scripts/main.sh [Base_Model_Name] [Abbreviation]
# Example using Qwen/Qwen3-4B-Base:
bash scripts/main.sh Qwen/Qwen3-4B-Base qwen3-4b
```

## Impressive Results

R-Zero delivers significant performance improvements across various reasoning benchmarks.

| Model Name          | Overall AVG | MATH AVG | SuperGPQA | MMLU-Pro | BBEH   |
| :------------------ | :----------: | :------: | :-------: | :------: | :----: |
| ***Qwen3-4B-Base*** |              |          |           |          |        |
| &emsp;Base Model     |    27.10     |  42.58   |   20.88   |  37.38   |  7.57  |
| &emsp;Base Challenger|    30.83     |  44.36   |   24.77   |  47.59   |  6.59  |
| &emsp;R-Zero (Iter 1) |    34.27     |  48.06   |  **27.92**  |  51.69   |  9.42  |
| &emsp;R-Zero (Iter 2) |  **34.92**   |  48.44   |   27.72   |  **53.75** |  9.76  |
| &emsp;R-Zero (Iter 3) |    34.64     |  **49.07** |   27.55   |  51.53   | **10.42**|
| ***Qwen3-8B-Base*** |              |          |           |          |        |
| &emsp;Base Model     |    34.49     |  49.18   |   28.33   |  51.80   |  8.63  |
| &emsp;Base Challenger|    36.43     |  51.87   |   30.12   |  54.14   |  9.60  |
| &emsp;R-Zero (Iter 1) |    37.93     |  53.39   |   31.26   |  57.17   |  9.91  |
| &emsp;R-Zero (Iter 2) |    38.45     |  53.84   |  **31.58**  |  58.20   | 10.20  |
| &emsp;R-Zero (Iter 3) |  **38.73**   |  **54.69** |   31.38   |  **58.23** | **10.60**|
| ***OctoThinker-3B*** |              |          |           |          |        |
| &emsp;Base Model     |    12.27     |  26.64   |   10.09   |  10.87   |  1.46  |
| &emsp;Base Challenger|    14.41     |  27.51   |   11.19   |  14.53   |  **4.40**  |
| &emsp;R-Zero (Iter 1) |    14.93     |  27.76   |   12.21   |  15.72   |  4.05  |
| &emsp;R-Zero (Iter 2) |    15.11     |  28.20   |   12.43   |  16.08   |  3.74  |
| &emsp;R-Zero (Iter 3) |  **15.67**   |  **29.32** |  **12.44**  |  **16.71** |  4.20  |
| ***OctoThinker-8B*** |              |          |           |          |        |
| &emsp;Base Model     |    16.81     |  32.11   |   13.26   |  20.21   |  1.64  |
| &emsp;Base Challenger|    25.08     |  36.41   |   16.99   |  41.46   |  5.46  |
| &emsp;R-Zero (Iter 1) |    26.44     |  37.80   |   19.15   |  **42.05** |  6.77  |
| &emsp;R-Zero (Iter 2) |    26.77     |  38.23   |   19.27   |  41.34   |  **8.25**  |
| &emsp;R-Zero (Iter 3) |  **26.88**   |  **38.52** |  **19.82**  |  40.92   |  **8.25**  |

## FAQ for Developers

**Q: What is the hardware setup for the experiments?**
**A:**  All experiments were conducted on an 8-GPU server using models that can run on a single GPU (e.g., 4B or 8B). Modify the code as necessary if you plan on using larger models or different hardware.

**Q: What should I do if I encounter environment configuration issues during installation?**
**A:**  Our framework's structure is inspired by [EasyR1](https://github.com/hiyouga/EasyR1/tree/main). If you run into environment-related issues, check their setup instructions or use their Docker environment.

**Q: Where are the training logs and model checkpoints saved?**
**A:** All generated data (logs, datasets, and model checkpoints) are saved in the directory specified by the `STORAGE_PATH` environment variable. Datasets will also be sent to Hugging Face, as specified by `HUGGINGFACENAME`.

**Q: What if the code gets stuck during the questioner training process?**
**A:** This may be caused by a bug in the `math_verify` library.  If this happens, restart the training from the last saved checkpoint.

## Acknowledgements

This project builds upon the work of [EasyR1](https://github.com/hiyouga/EasyR1/tree/main), and the evaluation process referenced the work from [General-Reasoner](https://github.com/TIGER-AI-Lab/General-Reasoner). We are very grateful for their contributions.

## Citation

If you find R-Zero useful, please consider citing our paper:

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