# R-Zero: Self-Evolving Reasoning for LLMs from Zero Data

**R-Zero revolutionizes Large Language Models by enabling them to autonomously improve their reasoning abilities without relying on any pre-existing data.**

[Read the full paper on arXiv](https://arxiv.org/abs/2508.05004) | [Visit the Project Webpage](https://chengsong-huang.github.io/R-Zero.github.io/) | [View the original repository on GitHub](https://github.com/Chengsong-Huang/R-Zero)

## 🔥 Key Updates

*   **[2025-08-12]** Updated codes for smoother training.
*   **[2025-08-08]** R-Zero was the `#2 Paper of the day` on [Hugging Face Daily Papers](https://huggingface.co/papers/2508.05004).
*   **[2025-08-07]** Released the [paper](https://arxiv.org/abs/2508.05004) and code.

## 💡 Overview

![R-Zero Overview](figs/abstract.png)

Training LLMs to reason effectively typically demands vast, hand-curated datasets, posing significant cost and scalability challenges.  R-Zero provides a novel solution by enabling LLMs to refine their reasoning skills autonomously, without any pre-existing tasks or labels. This innovative approach creates a self-evolving system that learns from scratch.

R-Zero operates through a dynamic co-evolutionary loop between two instances of the same base model:

1.  **The Challenger 🎯:** Generates challenging problems, probing the Solver's weaknesses.
2.  **The Solver 🧠:** Continuously improves its reasoning by solving the increasingly difficult tasks posed by the Challenger.

This iterative process establishes a tailored, adaptive curriculum. The Challenger learns to pose better questions, while the Solver learns to provide better answers. The entire cycle is self-contained, employing techniques like majority voting for pseudo-labels and relative policy optimization to guide learning.

### 🌟 Key Features

*   **Zero-Shot Learning:** No external data required, eliminating the need for pre-existing problem sets or human-annotated solutions.
*   **Co-Evolutionary Loop:** A Challenger-Solver dynamic drives a targeted and adaptive curriculum for continuous improvement.
*   **Performance Boosts:** Achieves significant performance gains across various reasoning benchmarks.
*   **Generalization:**  Reasoning skills learned in specific domains (e.g., math) effectively transfer to general reasoning tasks.
*   **Model Agnostic:** Improves the reasoning abilities of various LLM backbones.

---

## 🚀 Quickstart Guide

Get started with R-Zero in a few easy steps:

### 1. Configure Environment and Prepare Directories

```bash
git clone https://github.com/Chengsong-Huang/R-Zero.git

cd R-Zero
pip install -r requirements.txt

export STORAGE_PATH="/path/to/your/storage"  # Set your storage path
export HUGGINGFACENAME="yourhuggingfacename" # Set your hugging face name

mkdir -p \
  "$STORAGE_PATH/evaluation" \
  "$STORAGE_PATH/models" \
  "$STORAGE_PATH/generated_question" \
  "$STORAGE_PATH/temp_results"
```

### 2. Add API Keys

Add your API keys to `tokens.json`:

*   **Hugging Face** API key
*   **Weights & Biases (WandB)** API key (for logging)

Also, add your **OpenAI GPT** API key in `evaluation/results_recheck.py` for evaluation.

### 3. Run the Experiments

Replicate the experimental results with a single script:

```bash
#  Run with base model name and an abbreviation for directory creation
#  Example using Qwen/Qwen3-4B-Base:
bash scripts/main.sh Qwen/Qwen3-4B-Base qwen3-4b
```

## 📊 Impressive Results

The table below demonstrates the performance improvements of R-Zero.

| Model Name           | Overall AVG | MATH AVG | SuperGPQA | MMLU-Pro | BBEH |
| :------------------- | :----------: | :------: | :-------: | :------: | :--: |
| ***Qwen3-4B-Base***  |              |          |           |          |      |
| &emsp;Base Model     |    27.10     |  42.58   |   20.88   |  37.38   | 7.57 |
| &emsp;Base Challenger|    30.83     |  44.36   |   24.77   |  47.59   | 6.59 |
| &emsp;R-Zero (Iter 1)|    34.27     |  48.06   |   **27.92**   |  51.69   | 9.42 |
| &emsp;R-Zero (Iter 2)|    **34.92**     |  48.44   |   27.72   |  **53.75**   | 9.76 |
| &emsp;R-Zero (Iter 3)|    34.64     |  **49.07**   |   27.55   |  51.53   | **10.42** |
| ***Qwen3-8B-Base***  |              |          |           |          |      |
| &emsp;Base Model     |    34.49     |  49.18   |   28.33   |  51.80   | 8.63 |
| &emsp;Base Challenger|    36.43     |  51.87   |   30.12   |  54.14   | 9.60 |
| &emsp;R-Zero (Iter 1)|    37.93     |  53.39   |   31.26   |  57.17   | 9.91 |
| &emsp;R-Zero (Iter 2)|    38.45     |  53.84   |   **31.58**   |  **58.20**   | 10.20 |
| &emsp;R-Zero (Iter 3)|    **38.73**     |  **54.69**   |   31.38   |  **58.23**   | **10.60** |
| ***OctoThinker-3B*** |              |          |           |          |      |
| &emsp;Base Model     |    12.27     |  26.64   |   10.09   |  10.87   | 1.46 |
| &emsp;Base Challenger|    14.41     |  27.51   |   11.19   |  14.53   | **4.40** |
| &emsp;R-Zero (Iter 1)|    14.93     |  27.76   |   12.21   |  15.72   | 4.05 |
| &emsp;R-Zero (Iter 2)|    15.11     |  28.20   |   12.43   |  16.08   | 3.74 |
| &emsp;R-Zero (Iter 3)|    **15.67**     |  **29.32**   |   **12.44**   |  **16.71**   | 4.20 |
| ***OctoThinker-8B*** |              |          |           |          |      |
| &emsp;Base Model     |    16.81     |  32.11   |   13.26   |  20.21   | 1.64 |
| &emsp;Base Challenger|    25.08     |  36.41   |   16.99   |  41.46   | 5.46 |
| &emsp;R-Zero (Iter 1)|    26.44     |  37.80   |   19.15   |  **42.05**   | 6.77 |
| &emsp;R-Zero (Iter 2)|    26.77     |  38.23   |   19.27   |  41.34   | **8.25** |
| &emsp;R-Zero (Iter 3)|    **26.88**     |  **38.52**   |   **19.82**   |  40.92   | **8.25** |

## ❓ FAQ for Developers

### Q: What hardware is required for the experiments?

**A:** The experiments were conducted on an 8-GPU server. Models used can typically run on a single GPU (e.g., 4B or 8B). Adjust the code if using larger models or different hardware.

### Q: How do I troubleshoot environment configuration issues?

**A:** R-Zero is based on [EasyR1](https://github.com/hiyouga/EasyR1/tree/main). Refer to their setup instructions or Docker environment for guidance.

### Q: Where are training logs and model checkpoints saved?

**A:**  Generated data, including logs, datasets, and model checkpoints, is saved in the directory specified by the `STORAGE_PATH` environment variable. Datasets are also sent to Hugging Face, using the `HUGGINGFACENAME` environment variable.

### Q: What if the code gets stuck during questioner training?

**A:** This is likely due to a bug in the `math_verify` library.  Restart training from the last saved checkpoint if this occurs.

## 🙏 Acknowledgements

This project builds upon the foundational work of [**EasyR1**](https://github.com/hiyouga/EasyR1/tree/main), implementing its core functionalities.  Evaluation also referenced [**General-Reasoner**](https://github.com/TIGER-AI-Lab/General-Reasoner).

## 💬 Citation

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