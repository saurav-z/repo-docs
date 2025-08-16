# R-Zero: Revolutionizing Reasoning in LLMs with Self-Evolving Learning

**R-Zero empowers Large Language Models to autonomously learn and evolve their reasoning skills, requiring zero initial data.**

Explore the details in our [paper](https://arxiv.org/abs/2508.05004) and [webpage](https://chengsong-huang.github.io/R-Zero.github.io/).

## 🔥 Updates

*   **[2025-08-08]**  R-Zero was recognized as `#2 Paper of the day` on [huggingface daily paper](https://huggingface.co/papers/2508.05004).
*   **[2025-08-07]**  Paper and code released.

## 🌟 Overview

<p align="center">
  <img src="./figs/abstract.png" alt="R-Zero Overview" width="600"/>
</p>

Traditional methods for training reasoning models rely on vast, curated datasets, which can be costly and difficult to scale.  [**R-Zero**](https://arxiv.org/abs/2508.05004) introduces a groundbreaking framework that enables LLMs to enhance their reasoning abilities *autonomously*, without requiring any pre-existing data or labels.  This self-evolving system learns entirely from scratch.

R-Zero's core innovation is a dynamic co-evolutionary loop between two instances of the same base model:

1.  **The Challenger 🎯:**  Generates challenging problems to expose weaknesses in the Solver.
2.  **The Solver 🧠:**  Continuously improves by solving increasingly difficult tasks posed by the Challenger.

This creates a tailored, adaptive curriculum for continuous learning. The Challenger learns to ask better questions, and the Solver learns to find better answers, all within a self-contained loop. R-Zero uses techniques like majority voting for pseudo-labels and relative policy optimization to guide the learning.

### Key Features

*   **Zero-Shot Learning:**  Starts with no external data, eliminating the need for pre-existing datasets or human-annotated solutions.
*   **Co-Evolutionary Architecture:**  A unique Challenger-Solver dynamic provides a focused, adaptive curriculum for ongoing improvement.
*   **Demonstrated Performance:**  Achieves significant performance gains on multiple reasoning benchmarks.
*   **Strong Generalization:**  Reasoning skills learned in specific domains transfer effectively to general reasoning tasks.
*   **Model Agnostic:**  Enhances the performance of various LLM backbones.

---

## 🚀 Quickstart Guide

Get started with R-Zero in a few easy steps:

### 1. Configure Environment and Prepare Directories

```bash
git clone https://github.com/Chengsong-Huang/R-Zero.git

cd R-Zero
pip install -r requirements.txt

export STORAGE_PATH="/path/to/your/storage"  #  Set your storage path
export HUGGINGFACENAME="yourhuggingfacename"  # Set your Hugging Face name

mkdir -p \
  "$STORAGE_PATH/evaluation" \
  "$STORAGE_PATH/models" \
  "$STORAGE_PATH/generated_question" \
  "$STORAGE_PATH/temp_results"
```

### 2. Set Up API Keys

You will need to configure API keys for the following:

*   Add your **Hugging Face** and **WandB** (for logging) API keys to `tokens.json`.
*   Add your **OpenAI GPT** API key to `evaluation/results_recheck.py` for evaluation.

### 3. Run Experiments

Replicate our results using a single script:

```bash
#  The script takes the base model name and an abbreviation as arguments:
#  bash scripts/main.sh [Base_Model_Name] [Abbreviation]

#  Example:
bash scripts/main.sh Qwen/Qwen3-4B-Base qwen3-4b
```

## 📈 Impressive Results

The following table showcases the performance of the Base Model, a Zero-Shot Challenger baseline, and our iterative R-Zero framework. The best performance for each model is highlighted in **bold**.

| Model Name              | Overall AVG | MATH AVG | SuperGPQA | MMLU-Pro | BBEH   |
|:------------------------|:-----------:|:--------:|:----------:|:--------:|:-------:|
| ***Qwen3-4B-Base***      |             |          |            |          |         |
| &emsp;Base Model        | 27.10       | 42.58    | 20.88      | 37.38    | 7.57    |
| &emsp;Base Challenger   | 30.83       | 44.36    | 24.77      | 47.59    | 6.59    |
| &emsp;R-Zero (Iter 1)   | 34.27       | 48.06    | **27.92**  | 51.69    | 9.42    |
| &emsp;R-Zero (Iter 2)   | **34.92**   | 48.44    | 27.72      | **53.75** | 9.76    |
| &emsp;R-Zero (Iter 3)   | 34.64       | **49.07**| 27.55      | 51.53    | **10.42**|
| ***Qwen3-8B-Base***      |             |          |            |          |         |
| &emsp;Base Model        | 34.49       | 49.18    | 28.33      | 51.80    | 8.63    |
| &emsp;Base Challenger   | 36.43       | 51.87    | 30.12      | 54.14    | 9.60    |
| &emsp;R-Zero (Iter 1)   | 37.93       | 53.39    | 31.26      | 57.17    | 9.91    |
| &emsp;R-Zero (Iter 2)   | 38.45       | 53.84    | **31.58**  | 58.20    | 10.20   |
| &emsp;R-Zero (Iter 3)   | **38.73**   | **54.69**| 31.38      | **58.23** | **10.60**|
| ***OctoThinker-3B***     |             |          |            |          |         |
| &emsp;Base Model        | 12.27       | 26.64    | 10.09      | 10.87    | 1.46    |
| &emsp;Base Challenger   | 14.41       | 27.51    | 11.19      | 14.53    | **4.40**|
| &emsp;R-Zero (Iter 1)   | 14.93       | 27.76    | 12.21      | 15.72    | 4.05    |
| &emsp;R-Zero (Iter 2)   | 15.11       | 28.20    | 12.43      | 16.08    | 3.74    |
| &emsp;R-Zero (Iter 3)   | **15.67**   | **29.32**| **12.44**  | **16.71** | 4.20    |
| ***OctoThinker-8B***     |             |          |            |          |         |
| &emsp;Base Model        | 16.81       | 32.11    | 13.26      | 20.21    | 1.64    |
| &emsp;Base Challenger   | 25.08       | 36.41    | 16.99      | 41.46    | 5.46    |
| &emsp;R-Zero (Iter 1)   | 26.44       | 37.80    | 19.15      | **42.05** | 6.77    |
| &emsp;R-Zero (Iter 2)   | 26.77       | 38.23    | 19.27      | 41.34    | **8.25**|
| &emsp;R-Zero (Iter 3)   | **26.88**   | **38.52**| **19.82**  | 40.92    | **8.25**|

## ❓ Developer FAQ

### **Q: What hardware is required for the experiments?**

**A:** All experiments were performed on an 8-GPU server using models that fit on a single GPU (4B or 8B).  Adapting the code may be necessary if you are using larger models or different hardware.

### **Q: What if I encounter environment configuration issues?**

**A:** This project is inspired by [EasyR1](https://github.com/hiyouga/EasyR1/tree/main).  Consult their setup instructions or use their Docker environment for guidance.

### **Q: Where are training logs and model checkpoints stored?**

**A:** The `STORAGE_PATH` environment variable dictates the directory where all generated data, including logs, datasets, and model checkpoints, is saved. Datasets will be sent to Hugging Face via `HUGGINGFACENAME`.

### **Q: What if the code gets stuck during questioner training?**

**A:** This could be caused by a bug in the `math_verify` library, causing an infinite loop.  We've added a timeout to mitigate this, but you may need to restart from the last saved checkpoint.

## 🙏 Acknowledgements

This project is based on the work of [**EasyR1**](https://github.com/hiyouga/EasyR1/tree/main), implementing all core functionalities.  Our evaluation process references [**General-Reasoner**](https://github.com/TIGER-AI-Lab/General-Reasoner).

## 💬 Citation

If our work is valuable for your research, please cite our paper:

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