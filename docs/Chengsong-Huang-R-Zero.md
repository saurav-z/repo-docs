# R-Zero: Self-Evolving Reasoning LLMs from Zero Data

**Unlock the power of self-improvement for Large Language Models with R-Zero, a revolutionary framework that enables autonomous reasoning and skill enhancement.**

[Visit the original repository on GitHub](https://github.com/Chengsong-Huang/R-Zero)

R-Zero is a novel framework that allows Large Language Models (LLMs) to enhance their reasoning abilities autonomously, without the need for pre-existing datasets or human-labeled examples. This innovative approach fosters self-evolution in LLMs, enabling them to learn and improve from scratch.

## Key Features

*   **Autonomous Learning:** R-Zero initiates learning from a base LLM with no external data requirements.
*   **Co-Evolutionary Design:** Employs a "Challenger-Solver" dynamic to establish a targeted and evolving curriculum for enhanced learning.
*   **Performance Boost:** Demonstrates significant gains across a range of reasoning benchmarks.
*   **Broad Applicability:** Improves the performance of various LLM architectures.
*   **Generalization Ability:** Reasoning skills learned within specific areas, like mathematics, successfully extend to a wider range of reasoning tasks.

## Updates

*   **2025-08-27:** Analysis of iteration scaling and the application of a single model to fulfill dual roles has been added.
*   **2025-08-25:** Code updates implemented to enhance training smoothness (utilizing stopit).
*   **2025-08-08:** R-Zero recognized as `#2 Paper of the day` on [Hugging Face Papers](https://huggingface.co/papers/2508.05004).
*   **2025-08-07:** Paper and associated code released.

## Overview

R-Zero reimagines LLM training by eliminating the need for extensive, pre-curated datasets. The framework's core lies in a dynamic co-evolutionary loop between two instances of the same base model:

*   **Challenger 🎯:** Designed to identify weaknesses within the Solver and generate challenging problems that push its capabilities.
*   **Solver 🧠:** Focused on continuous improvement through solving increasingly complex tasks posed by the Challenger.

This synergistic process constructs an adaptive curriculum. The Challenger refines its question-generation skills, while the Solver perfects its answer accuracy. This system relies on techniques like majority voting for pseudo-labels and relative policy optimization to enhance its learning capabilities.

## Quickstart Guide

Get started with R-Zero in a few easy steps:

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

Provide your API keys in the following files:

*   In `tokens.json`, add your API keys for **Hugging Face** and **WandB** (for logging).
*   In `evaluation/results_recheck.py`, add your **OpenAI GPT** API key for evaluation.

### 3. Run Experiments

Replicate the experimental results with a single script:

```bash
# The script takes the base model name and an abbreviation as arguments
# The abbreviation is used for creating a directory to save the model.
# Format: bash scripts/main.sh [Base_Model_Name] [Abbreviation]

# Example using Qwen/Qwen3-4B-Base:
bash scripts/main.sh Qwen/Qwen3-4B-Base qwen3-4b
```

## Impressive Results

R-Zero delivers significant performance improvements across several benchmarks. See the comparison of Base Model, Zero-Shot Challenger, and R-Zero framework.

| Model Name        | Overall AVG | MATH AVG | SuperGPQA | MMLU-Pro | BBEH |
| :---------------- | :----------: | :------: | :-------: | :------: | :--: |
| ***Qwen3-4B-Base*** |      ...     |    ...   |     ...   |    ...   |  ... |
| &emsp;Base Model     |     27.10    |   42.58  |   20.88   |   37.38  | 7.57 |
| &emsp;Base Challenger |     30.83    |   44.36  |   24.77   |   47.59  | 6.59 |
| &emsp;R-Zero (Iter 1)  |     34.27    |   48.06  |   **27.92**  |   51.69  | 9.42 |
| &emsp;R-Zero (Iter 2)  |   **34.92**  |   48.44  |   27.72   |  **53.75**  | 9.76 |
| &emsp;R-Zero (Iter 3)  |     34.64    |   **49.07**  |   27.55   |   51.53  | **10.42** |
| ***Qwen3-8B-Base*** |      ...     |    ...   |     ...   |    ...   |  ... |
| &emsp;Base Model     |     34.49    |   49.18  |   28.33   |   51.80  | 8.63 |
| &emsp;Base Challenger |     36.43    |   51.87  |   30.12   |   54.14  | 9.60 |
| &emsp;R-Zero (Iter 1)  |     37.93    |   53.39  |   31.26   |   57.17  | 9.91 |
| &emsp;R-Zero (Iter 2)  |     38.45    |   53.84  |   **31.58**  |  **58.20**  | 10.20 |
| &emsp;R-Zero (Iter 3)  |   **38.73**  |   **54.69**  |   31.38   |  **58.23**  | **10.60** |
| ***OctoThinker-3B*** |      ...     |    ...   |     ...   |    ...   |  ... |
| &emsp;Base Model     |     12.27    |   26.64  |   10.09   |   10.87  | 1.46 |
| &emsp;Base Challenger |     14.41    |   27.51  |   11.19   |   14.53  | **4.40** |
| &emsp;R-Zero (Iter 1)  |     14.93    |   27.76  |   12.21   |   15.72  | 4.05 |
| &emsp;R-Zero (Iter 2)  |     15.11    |   28.20  |   12.43   |   16.08  | 3.74 |
| &emsp;R-Zero (Iter 3)  |   **15.67**  |   **29.32**  |   **12.44**  |   **16.71**  | 4.20 |
| ***OctoThinker-8B*** |      ...     |    ...   |     ...   |    ...   |  ... |
| &emsp;Base Model     |     16.81    |   32.11  |   13.26   |   20.21  | 1.64 |
| &emsp;Base Challenger |     25.08    |   36.41  |   16.99   |   41.46  | 5.46 |
| &emsp;R-Zero (Iter 1)  |     26.44    |   37.80  |   19.15   |   **42.05**  | 6.77 |
| &emsp;R-Zero (Iter 2)  |     26.77    |   38.23  |   19.27   |   41.34  | **8.25** |
| &emsp;R-Zero (Iter 3)  |   **26.88**  |   **38.52**  |   **19.82**  |   40.92  | **8.25** |

## FAQ for Developers

### Q: Hardware Setup for Experiments?

**A:** The experiments were conducted on an 8-GPU server, using models designed to run on a single GPU (e.g., 4B or 8B). Adjust the code if you need to run experiments with larger models or different hardware.

### Q: Troubleshooting Environment Configuration Issues?

**A:** The framework structure is inspired by [EasyR1](https://github.com/hiyouga/EasyR1/tree/main). If you run into any environment-related issues, we highly recommend checking out their setup instructions or using their Docker environment as a reference.

### Q: Where are Training Logs and Checkpoints Saved?

**A:** All generated data, including logs, datasets, and model checkpoints, will be saved in the directory set by the `STORAGE_PATH` environment variable. Datasets are also sent to Hugging Face via `HUGGINGFACENAME`.

### Q: What if the code gets stuck during the questioner training process?

**A:** This issue is most likely due to a bug in the `math_verify` lib, which can cause an infinite loop when processing certain answers. While there is a timeout control to mitigate this, it may not catch all cases. If this happens, simply restart training from the last saved checkpoint.

## Acknowledgements

This framework is built upon the foundation of [**EasyR1**](https://github.com/hiyouga/EasyR1/tree/main), implementing all core functionalities. The evaluation process also references the work from [**General-Reasoner**](https://github.com/TIGER-AI-Lab/General-Reasoner).

## Citation

If you find our work beneficial, please consider citing our paper:

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