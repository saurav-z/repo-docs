# R-Zero: Self-Evolving Reasoning LLM from Zero Data

> Revolutionize LLM reasoning with R-Zero, a groundbreaking framework that enables Large Language Models to autonomously improve reasoning skills without any prior data or human input!

Check out our [paper](https://arxiv.org/abs/2508.05004) and [webpage](https://chengsong-huang.github.io/R-Zero.github.io/) for more details.

## Key Features

*   **Fully Autonomous Learning:** R-Zero eliminates the need for pre-existing datasets, starting from a blank slate.
*   **Co-Evolutionary Architecture:** A dynamic Challenger-Solver loop generates a tailored curriculum, fostering continuous improvement.
*   **Significant Performance Gains:** Witness substantial performance boosts on various reasoning benchmarks.
*   **Robust Generalization:** Reasoning abilities learned on specific domains effectively transfer to broader reasoning tasks.
*   **Model-Agnostic:** Improves the reasoning capabilities of a wide range of LLMs.

## What's New

*   **[2025-8-27]** Analysis of iteration scaling and dual-role model integration.
*   **[2025-8-25]** Code updates for smoother training (stopit).
*   **[2025-8-8]** Recognized as `#2 Paper of the day` on [Hugging Face Papers](https://huggingface.co/papers/2508.05004).
*   **[2025-8-7]** Released the [paper](https://arxiv.org/abs/2508.05004) and code.

## Overview

[![R-Zero Overview](./figs/abstract.png)](https://arxiv.org/abs/2508.05004)

Traditional methods for training high-performing reasoning models demand vast, manually curated datasets, a process both expensive and difficult to scale. R-Zero offers a novel solution: a self-evolving system that empowers LLMs to enhance their reasoning abilities without any pre-existing tasks or labels. It's a truly self-contained learning system that starts from scratch, building its knowledge base through autonomous interaction.

R-Zero employs a co-evolutionary loop between two instances of the same base model:

1.  **Challenger 🎯:** This module probes the Solver, identifying weaknesses and creating challenging problems that push the boundaries of the Solver's capabilities.
2.  **Solver 🧠:** The Solver continuously improves by addressing the increasingly complex tasks posed by the Challenger.

This creates an adaptive curriculum tailored to the LLM's specific learning needs. The Challenger refines its question-asking skills, and the Solver hones its answer-finding abilities. The entire cycle is self-sufficient, employing techniques like majority voting for pseudo-labels and relative policy optimization to guide the learning process.

## Quickstart Guide

Get started with R-Zero in a few simple steps:

### 1. Environment Setup and Directory Preparation

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

Add your API keys for **Hugging Face** and **WandB** (in `tokens.json`) and your **OpenAI GPT** API key (in `evaluation/results_recheck.py`).

### 3. Run the Experiments

Replicate our experiments with a single script:

```bash
# Specify the base model and abbreviation:
# Format: bash scripts/main.sh [Base_Model_Name] [Abbreviation]
bash scripts/main.sh Qwen/Qwen3-4B-Base qwen3-4b
```

## Impressive Results

The table below showcases the performance of the Base Model, a Zero-Shot Challenger baseline, and the iterative R-Zero framework.  Best performance is shown in **bold**.

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

## FAQ for Developers

### Q: What hardware setup was used for the experiments?

**A:**  Experiments were run on an 8-GPU server using models suitable for a single GPU.  Adjust code for larger models or different hardware.

### Q: How do I troubleshoot environment configuration issues?

**A:**  Refer to the [EasyR1](https://github.com/hiyouga/EasyR1/tree/main) setup instructions or their Docker environment.

### Q: Where are training logs and checkpoints saved?

**A:**  Training data is saved in the directory specified by the `STORAGE_PATH` environment variable, also uploads to HuggingFace via `HUGGINGFACENAME`.

### Q: What if the code stalls during questioner training?

**A:**  This may be due to a bug in the `math_verify` lib; restart training from the last saved checkpoint.

## Acknowledgements

We built upon the excellent work of [**EasyR1**](https://github.com/hiyouga/EasyR1/tree/main) and the evaluation work from [**General-Reasoner**](https://github.com/TIGER-AI-Lab/General-Reasoner).

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

[Back to Top](#r-zero-self-evolving-reasoning-llm-from-zero-data)
```
Key improvements:

*   **SEO Optimization:** Added keywords (Reasoning LLM, self-evolving, zero data) in the title and throughout the text.  Used H2 and H3 headings for better structure.
*   **Stronger Hook:**  The opening sentence is more engaging and highlights the key benefit.
*   **Concise & Clear Language:**  Simplified wording for better readability.
*   **Prioritized Information:**  Focused on the most important aspects.
*   **Call to Action:** Added "Back to Top" links for navigation.
*   **Enhanced Formatting:** Improved formatting (bolding, bullet points) for visual appeal and clarity.
*   **Complete:** Included all relevant sections and made no assumptions about the content.
*   **Internal Links:** Added an internal link "Back to Top".
*   **Removed redundant information:** Removed redundant info in the FAQ.

This improved README is more informative, engaging, and easier to navigate, making it more likely to attract users and improve search engine rankings.