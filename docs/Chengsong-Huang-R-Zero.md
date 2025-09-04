# R-Zero: Self-Evolving Reasoning LLM from Zero Data

**R-Zero revolutionizes how Large Language Models learn to reason by enabling them to evolve their abilities autonomously, without any initial training data.**  Check out the original repo [here](https://github.com/Chengsong-Huang/R-Zero).

*For detailed information, please refer to our [paper](https://arxiv.org/abs/2508.05004) and [webpage](https://chengsong-huang.github.io/R-Zero.github.io/).*

## Key Features

*   **Fully Autonomous Learning:** R-Zero starts with a base model and learns entirely from self-generated data, eliminating the need for pre-existing datasets or human-labeled examples.
*   **Co-Evolutionary Framework:**  Leverages a Challenger-Solver dynamic, creating a targeted, adaptive curriculum for continuous improvement in reasoning skills.
*   **Model-Agnostic:**  R-Zero's framework enhances the reasoning capabilities of various backbone LLMs, demonstrating broad applicability.
*   **Proven Performance:** Achieves significant performance boosts on multiple reasoning benchmarks, showcasing its effectiveness.
*   **Strong Generalization:** Skills learned in specific domains (e.g., math) translate effectively to general reasoning tasks.

## Updates

*   **[2025-8-27]** Analysis added on iteration scaling and a model taking on two roles.
*   **[2025-8-25]** Code updates for smoother training (using `stopit`).
*   **[2025-8-8]**  Paper featured as `#2 Paper of the day` on [Hugging Face Daily Papers](https://huggingface.co/papers/2508.05004).
*   **[2025-8-7]**  Paper and code released.

## Overview

<img src="./figs/abstract.png" alt="R-Zero Overview" width="600">

Training reasoning models typically requires massive, curated datasets, which are resource-intensive and hard to scale. R-Zero addresses this by enabling LLMs to improve their reasoning abilities autonomously from scratch.

The core of R-Zero is a dynamic co-evolutionary loop between two instances of the same base model:

1.  **Challenger 🎯:**  Generates challenging problems to expose the Solver's weaknesses.
2.  **Solver 🧠:**  Continuously improves by solving the increasingly difficult problems posed by the Challenger.

This creates a self-tailored curriculum, where the Challenger learns to ask better questions, and the Solver learns to find better answers. The entire process is self-contained, utilizing methods like majority voting for pseudo-labels and relative policy optimization to guide learning.

## Quickstart Guide

Get started with R-Zero in a few simple steps:

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

Configure API keys for the following:

*   **Hugging Face** and **WandB** (for logging) in `tokens.json`.
*   **OpenAI GPT** (for evaluation) in `evaluation/results_recheck.py`.

### 3. Run the Experiments

Replicate the experimental results with this script:

```bash
# Arguments: [Base_Model_Name] [Abbreviation]
# Example using Qwen/Qwen3-4B-Base:
bash scripts/main.sh Qwen/Qwen3-4B-Base qwen3-4b
```

## Impressive Results

| Model Name          | Overall AVG | MATH AVG | SuperGPQA | MMLU-Pro | BBEH  |
|:--------------------|:-----------:|:--------:|:----------:|:--------:|:-----:|
| ***Qwen3-4B-Base*** |             |          |            |          |       |
| &emsp;Base Model    | 27.10       | 42.58    | 20.88      | 37.38    | 7.57  |
| &emsp;Base Challenger | 30.83       | 44.36    | 24.77      | 47.59    | 6.59  |
| &emsp;R-Zero (Iter 1) | 34.27       | 48.06    | **27.92**   | 51.69    | 9.42  |
| &emsp;R-Zero (Iter 2) | **34.92**      | 48.44    | 27.72      | **53.75**   | 9.76  |
| &emsp;R-Zero (Iter 3) | 34.64       | **49.07**   | 27.55      | 51.53    | **10.42** |
| ***Qwen3-8B-Base*** |             |          |            |          |       |
| &emsp;Base Model    | 34.49       | 49.18    | 28.33      | 51.80    | 8.63  |
| &emsp;Base Challenger | 36.43       | 51.87    | 30.12      | 54.14    | 9.60  |
| &emsp;R-Zero (Iter 1) | 37.93       | 53.39    | 31.26      | 57.17    | 9.91  |
| &emsp;R-Zero (Iter 2) | 38.45       | 53.84    | **31.58**   | 58.20    | 10.20 |
| &emsp;R-Zero (Iter 3) | **38.73**      | **54.69**   | 31.38      | **58.23**   | **10.60** |
| ***OctoThinker-3B*** |             |          |            |          |       |
| &emsp;Base Model    | 12.27       | 26.64    | 10.09      | 10.87    | 1.46  |
| &emsp;Base Challenger | 14.41       | 27.51    | 11.19      | 14.53    | **4.40**  |
| &emsp;R-Zero (Iter 1) | 14.93       | 27.76    | 12.21      | 15.72    | 4.05  |
| &emsp;R-Zero (Iter 2) | 15.11       | 28.20    | 12.43      | 16.08    | 3.74  |
| &emsp;R-Zero (Iter 3) | **15.67**      | **29.32**   | **12.44**   | **16.71**   | 4.20  |
| ***OctoThinker-8B*** |             |          |            |          |       |
| &emsp;Base Model    | 16.81       | 32.11    | 13.26      | 20.21    | 1.64  |
| &emsp;Base Challenger | 25.08       | 36.41    | 16.99      | 41.46    | 5.46  |
| &emsp;R-Zero (Iter 1) | 26.44       | 37.80    | 19.15      | **42.05**   | 6.77  |
| &emsp;R-Zero (Iter 2) | 26.77       | 38.23    | 19.27      | 41.34    | **8.25**  |
| &emsp;R-Zero (Iter 3) | **26.88**      | **38.52**   | **19.82**   | 40.92    | **8.25**  |

## FAQ for Developers

### Q: Hardware Setup?
**A:** Experiments were conducted on an 8-GPU server. Adapt the code for larger models or different hardware configurations as needed.

### Q: Environment Configuration Issues?
**A:**  Refer to [EasyR1's setup instructions](https://github.com/hiyouga/EasyR1/tree/main) or their Docker environment for troubleshooting.

### Q: Training Logs and Checkpoints?
**A:**  Logs, datasets, and model checkpoints are saved in the `STORAGE_PATH` directory set via the environment variable. Datasets are also sent to Hugging Face via `HUGGINGFACENAME`.

### Q: Code Sticking During Questioner Training?
**A:** This may be due to a bug in the `math_verify` library. Restart from the last checkpoint if this occurs.

## Acknowledgements

This work builds upon the foundational research of [**EasyR1**](https://github.com/hiyouga/EasyR1/tree/main), with core functionalities implemented based on their work. Evaluation references the methodology from [**General-Reasoner**](https://github.com/TIGER-AI-Lab/General-Reasoner).

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
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Chengsong-Huang/R-Zero&type=Date)](https://star-history.com/#Chengsong-Huang/R-Zero&Date)