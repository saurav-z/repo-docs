# R-Zero: Self-Evolving Reasoning LLM from Zero Data

**R-Zero revolutionizes how Large Language Models (LLMs) learn to reason, enabling them to improve their abilities autonomously without any initial data.** ([Original Repo](https://github.com/Chengsong-Huang/R-Zero))

## Key Features

*   **Autonomous Learning:** Trains LLMs from scratch, eliminating the need for pre-existing datasets or human-labeled examples.
*   **Co-Evolutionary Framework:** Employs a Challenger-Solver loop for continuous improvement, creating a targeted and adaptive learning curriculum.
*   **Enhanced Performance:** Achieves significant performance gains on reasoning benchmarks, outperforming baseline models.
*   **Strong Generalization:** Reasoning skills learned in specific domains transfer effectively to general reasoning tasks.
*   **Model Agnostic:** R-Zero consistently improves the reasoning abilities of different backbone LLMs.

## Overview

R-Zero introduces a novel framework that allows LLMs to self-evolve their reasoning capabilities. It leverages a dynamic co-evolutionary loop between two instances of the same base model:

*   **The Challenger:** Probes the Solver for weaknesses by generating challenging problems, pushing the boundaries of its abilities.
*   **The Solver:** Continuously improves by solving the increasingly difficult tasks posed by the Challenger.

This self-contained cycle utilizes techniques like majority voting for pseudo-labels and relative policy optimization, creating an adaptive curriculum for enhanced reasoning.

## 🔥 Updates

*   **[2025-8-27]:** Analysis on iteration scaling and one model taking on two roles added.
*   **[2025-8-25]:** Code updates for smoother training (stopit).
*   **[2025-8-8]:**  R-Zero received `#2 Paper of the day` on Hugging Face.
*   **[2025-8-7]:** Paper and code released.

## ⚡️ Quickstart Guide

Follow these steps to get started:

### 1. Configure Environment and Prepare Directories

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

### 2. Add API Keys

*   Add your **Hugging Face** and **WandB** API keys in `tokens.json`.
*   Add your **OpenAI GPT** API key in `evaluation/results_recheck.py`.

### 3. Run Experiments

```bash
# Format: bash scripts/main.sh [Base_Model_Name] [Abbreviation]

# Example using Qwen/Qwen3-4B-Base:
bash scripts/main.sh Qwen/Qwen3-4B-Base qwen3-4b
```

## 📊 Impressive Results

| Model Name        | Overall AVG | MATH AVG | SuperGPQA | MMLU-Pro | BBEH  |
| :---------------- | :----------: | :------: | :--------: | :------: | :---: |
| ***Qwen3-4B-Base*** |      ...     |    ...   |    ...     |    ...   | :---: |
| ...               |      ...     |    ...   |    ...     |    ...   | :---: |
| **R-Zero (Iter 3)** |   **34.64**  | **49.07** |   27.55    |   51.53  | 10.42 |
| ***Qwen3-8B-Base*** |      ...     |    ...   |    ...     |    ...   | :---: |
| **R-Zero (Iter 3)** |   **38.73**  | **54.69** |   31.38    | **58.23** | 10.60 |
| ***OctoThinker-3B*** |      ...     |    ...   |    ...     |    ...   | :---: |
| **R-Zero (Iter 3)** |   **15.67**  | **29.32** |   **12.44** |  **16.71** | 4.20  |
| ***OctoThinker-8B*** |      ...     |    ...   |    ...     |    ...   | :---: |
| **R-Zero (Iter 3)** |   **26.88**  | **38.52** |   **19.82** |   40.92  | 8.25  |

*(Note: table is truncated, refer to the original repo for the full table)*

## ❓ FAQ for Developers

### Q: What is the hardware setup for the experiments?

**A:** Experiments were performed on an 8-GPU server. You may need to modify the code for larger models or different hardware.

### Q: What if I encounter environment configuration issues during installation?

**A:** Refer to the [EasyR1](https://github.com/hiyouga/EasyR1/tree/main) setup instructions or Docker environment for guidance.

### Q: Where are the training logs and model checkpoints saved?

**A:**  `STORAGE_PATH` (environment variable) directs where generated data, logs, datasets, and model checkpoints are saved. Datasets are also sent to Hugging Face via `HUGGINGFACENAME`.

### Q: What if the code gets stuck during the questioner training process?

**A:** Restart training from the last saved checkpoint if you encounter this issue.

## 🙏 Acknowledgements

This project builds upon the work of [**EasyR1**](https://github.com/hiyouga/EasyR1/tree/main) and references the evaluation process from [**General-Reasoner**](https://github.com/TIGER-AI-Lab/General-Reasoner).

## 💬 Citation

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