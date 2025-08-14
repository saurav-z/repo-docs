<!-- Improved and SEO-optimized README for R-Zero -->

# R-Zero: Revolutionizing LLM Reasoning with Self-Evolution

**R-Zero empowers Large Language Models to autonomously enhance their reasoning skills from scratch, eliminating the need for pre-existing data and transforming how we train intelligent systems.** ([Original Repository](https://github.com/Chengsong-Huang/R-Zero))

## Key Features

*   **Autonomous Learning:** Trains LLMs without any external datasets, labels, or human supervision.
*   **Self-Evolving Architecture:** Leverages a Challenger-Solver co-evolutionary loop for continuous improvement.
*   **Enhanced Performance:** Significantly boosts reasoning abilities on various benchmark tasks.
*   **Generalization:** Reasoning skills learned on specific domains transfer to general reasoning challenges.
*   **Model Agnostic:** Works effectively with various base LLMs, improving their performance.

## What's New

*   **[2025-08-12]** Updated codes to make training smoother.
*   **[2025-08-08]** R-Zero featured as `#2 Paper of the Day` on [Hugging Face's Daily Papers](https://huggingface.co/papers/2508.05004).
*   **[2025-08-07]** Released paper and code.

## Overview

[<img src="./figs/abstract.png" alt="R-Zero Overview" width="600">](https://arxiv.org/abs/2508.05004)

Traditional methods for training reasoning models rely on vast, curated datasets, which are expensive and difficult to scale. R-Zero introduces a novel approach that enables Large Language Models (LLMs) to autonomously improve their reasoning capabilities without requiring pre-existing tasks or labeled data. It's a self-evolving system that learns and improves from the ground up.

At its core, R-Zero establishes a dynamic co-evolutionary loop between two instances of the same base model:

1.  **Challenger 🎯:** Creates challenging problems that probe the Solver's weaknesses, pushing it to its limits.
2.  **Solver 🧠:** Continuously refines its ability to solve the increasingly complex tasks posed by the Challenger.

This process builds an adaptive curriculum tailored to the model's needs. The Challenger hones its question-generation skills, while the Solver perfects its answer-finding abilities. The entire process is self-contained, utilizing techniques like majority voting for pseudo-labeling and relative policy optimization to guide learning.

## Quickstart Guide

Get started with R-Zero using these simple steps:

### 1.  Configure Environment & Prepare Directories

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

### 2.  Set Up API Keys

*   Add your Hugging Face and WandB API keys to `tokens.json`.
*   Add your OpenAI GPT API key to `evaluation/results_recheck.py`.

### 3.  Run Experiments

Replicate our experimental results with a single script:

```bash
# Format: bash scripts/main.sh [Base_Model_Name] [Abbreviation]
# Example using Qwen/Qwen3-4B-Base:
bash scripts/main.sh Qwen/Qwen3-4B-Base qwen3-4b
```

## Performance Results

The following table compares the performance of the Base Model, a Zero-Shot Challenger baseline, and the iterative R-Zero framework.  **Bold** indicates peak performance.

| Model Name        | Overall AVG | MATH AVG | SuperGPQA | MMLU-Pro | BBEH  |
| :---------------- | :----------: | :------: | :-------: | :------: | :---: |
| ***Qwen3-4B-Base*** |      ...     |    ...   |    ...    |    ...   |  ...  |
| ***Qwen3-8B-Base*** |      ...     |    ...   |    ...    |    ...   |  ...  |
| ***OctoThinker-3B*** |      ...     |    ...   |    ...    |    ...   |  ...  |
| ***OctoThinker-8B*** |      ...     |    ...   |    ...    |    ...   |  ...  |

<!-- Add the data from the original README here -->

## FAQ for Developers

### Q: What hardware is required for the experiments?

**A:**  Experiments were conducted on an 8-GPU server. The models used can run on a single GPU (e.g., 4B or 8B). You may need to modify the code for larger models or different hardware setups.

### Q: What if I encounter environment configuration issues?

**A:** Refer to the setup instructions or Docker environment of [EasyR1](https://github.com/hiyouga/EasyR1/tree/main), which inspired the framework.

### Q: Where are the training logs and model checkpoints saved?

**A:** Logs, datasets, and checkpoints are saved in the directory specified by the `STORAGE_PATH` environment variable. The datasets are also sent to Hugging Face via `HUGGINGFACENAME`.

### Q: What should I do if the code gets stuck during questioner training?

**A:** This may be due to a known bug in the `math_verify` library. If this happens, restart training from the last saved checkpoint.

## Acknowledgements

This project is built upon the foundational work of [**EasyR1**](https://github.com/hiyouga/EasyR1/tree/main) and references the evaluation process of [**General-Reasoner**](https://github.com/TIGER-AI-Lab/General-Reasoner).  We are very grateful for their contributions.

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