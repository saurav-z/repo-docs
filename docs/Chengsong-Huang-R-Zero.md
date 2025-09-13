# R-Zero: Revolutionizing LLM Reasoning with Self-Evolution 

**R-Zero empowers Large Language Models to learn and enhance their reasoning abilities autonomously, without any pre-existing data.**

[![Star History Chart](https://api.star-history.com/svg?repos=Chengsong-Huang/R-Zero&type=Date)](https://star-history.com/#Chengsong-Huang/R-Zero&Date)

[View the original repository on GitHub](https://github.com/Chengsong-Huang/R-Zero)

R-Zero introduces a groundbreaking framework that allows LLMs to evolve their reasoning skills from scratch. This innovative approach eliminates the need for curated datasets, paving the way for more efficient and scalable model training. For in-depth details, please refer to our [paper](https://arxiv.org/abs/2508.05004) and [webpage](https://chengsong-huang.github.io/R-Zero.github.io/).

## Key Features

*   **Autonomous Learning:** R-Zero eliminates the need for external training data.
*   **Co-Evolutionary Design:** A Challenger-Solver loop creates a dynamic learning curriculum.
*   **Performance Gains:** Demonstrates significant improvements on reasoning benchmarks.
*   **Generalization Ability:** Reasoning skills acquired in specific domains translate to broader reasoning tasks.
*   **Model Agnostic:** Effective across diverse LLM architectures.

## Updates

*   **2025-8-27:** Analysis on iteration scaling and one model taking on two roles.
*   **2025-8-25:** Code updates for smoother training (by stopit).
*   **2025-8-8:** Recognized as `#2 Paper of the day` on [huggingface daily paper](https://huggingface.co/papers/2508.05004).
*   **2025-8-7:** Released [paper](https://arxiv.org/abs/2508.05004) and code.

## Overview

![R-Zero Overview](./figs/abstract.png)

R-Zero revolutionizes the way LLMs learn by introducing a self-evolving system. This system features a co-evolutionary loop between two instances of the same base model:

1.  **Challenger 🎯:** Generates complex problems tailored to challenge the Solver's capabilities.
2.  **Solver 🧠:** Continuously improves by tackling increasingly difficult challenges posed by the Challenger.

This closed-loop system uses techniques like majority voting for pseudo-labels and relative policy optimization, leading to continuous improvement in both question generation and answer accuracy.

## Quickstart Guide

Get started with R-Zero in a few simple steps:

### 1. Configure Environment and Prepare Directories

```bash
git clone https://github.com/Chengsong-Huang/R-Zero.git
cd R-Zero
pip install -r requirements.txt
export STORAGE_PATH="/path/to/your/storage"
export HUGGINGFACENAME="yourhuggingfacename"

mkdir -p "$STORAGE_PATH/evaluation" "$STORAGE_PATH/models" "$STORAGE_PATH/generated_question" "$STORAGE_PATH/temp_results"
```

### 2. Add API Keys

*   Add your Hugging Face and WandB API keys in `tokens.json`.
*   Include your OpenAI GPT API key in `evaluation/results_recheck.py`.

### 3. Run Experiments

Replicate the experimental results with a single script:

```bash
# Format: bash scripts/main.sh [Base_Model_Name] [Abbreviation]
# Example:
bash scripts/main.sh Qwen/Qwen3-4B-Base qwen3-4b
```

## Impressive Results

The table below demonstrates the performance gains achieved by R-Zero compared to baseline models across various benchmarks. (**Bold** indicates peak performance).

| Model Name          | Overall AVG | MATH AVG | SuperGPQA | MMLU-Pro | BBEH   |
| :------------------ | :----------: | :------: | :-------: | :------: | :----: |
| ***Qwen3-4B-Base*** |      ...     |    ...   |    ...    |    ...   |  ...   |
| ***Qwen3-8B-Base*** |      ...     |    ...   |    ...    |    ...   |  ...   |
| ***OctoThinker-3B*** |      ...     |    ...   |    ...    |    ...   |  ...   |
| ***OctoThinker-8B*** |      ...     |    ...   |    ...    |    ...   |  ...   |

*(Note: The above table is truncated for brevity. See original README for full results.)*

## FAQ for Developers

### **Q: What is the hardware setup for the experiments?**

**A:** Experiments were run on an 8-GPU server, using models that can run on a single GPU (4B or 8B). Adjust code as needed for different hardware/model sizes.

### **Q: What if I encounter environment configuration issues during installation?**

**A:** Refer to the setup instructions or Docker environment of [EasyR1](https://github.com/hiyouga/EasyR1/tree/main) for guidance.

### **Q: Where are the training logs and model checkpoints saved?**

**A:** All generated data, including logs, datasets, and model checkpoints, are saved in the directory specified by the `STORAGE_PATH` environment variable. Datasets are also uploaded to Hugging Face using `HUGGINGFACENAME`.

### **Q: What if the code gets stuck during the questioner training process?**

**A:** This may be due to a bug in the `math_verify` library. Restart training from the last saved checkpoint.

## Acknowledgements

This work is based on the work of [**EasyR1**](https://github.com/hiyouga/EasyR1/tree/main) and utilizes the evaluation framework of [**General-Reasoner**](https://github.com/TIGER-AI-Lab/General-Reasoner). We are grateful for their contributions.

## Citation

If you find our work helpful, please cite our paper:

```
@article{huang2025rzeroselfevolvingreasoningllm,
      title={R-Zero: Self-Evolving Reasoning LLM from Zero Data}, 
      author={Chengsong Huang and ...},
      year={2025},
      eprint={2508.05004},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2508.05004}, 
}
```