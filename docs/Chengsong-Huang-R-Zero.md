# R-Zero: Evolve Your LLM Reasoning Skills with Zero Data 🚀

**R-Zero revolutionizes how Large Language Models (LLMs) learn to reason by enabling them to improve autonomously, starting from scratch and without any external datasets.** Explore the code and contribute on [GitHub](https://github.com/Chengsong-Huang/R-Zero).

## Key Features

*   ✅ **Autonomous Learning:** No pre-existing tasks or human-labeled data needed.
*   🔄 **Co-Evolutionary Loop:** A Challenger-Solver dynamic creates a targeted, adaptive curriculum.
*   📈 **Performance Boosts:** Achieves significant improvements on reasoning benchmarks.
*   🌍 **Strong Generalization:** Reasoning skills transfer across different domains.
*   ⚙️ **Model-Agnostic:** Enhances the performance of various LLM architectures.

## Overview

<img src="./figs/abstract.png" alt="R-Zero Overview" width="600">

Training powerful reasoning models typically demands vast, hand-curated datasets, which can be costly and challenging to scale. R-Zero offers a novel approach: a self-evolving framework that empowers LLMs to enhance their reasoning abilities independently, requiring no pre-existing tasks or labels.

R-Zero establishes a dynamic co-evolutionary loop between two instances of the same base model:

*   **Challenger:** Probes the Solver for weaknesses, generating challenging problems.
*   **Solver:** Continuously improves by solving the tasks posed by the Challenger.

This process constructs a tailored, adaptive curriculum, where the Challenger learns to ask more effective questions and the Solver hones its answering skills. This self-contained cycle utilizes techniques like majority voting for pseudo-labels and relative policy optimization to guide learning.

## Quickstart Guide

Get up and running with R-Zero in a few simple steps:

### 1. Configure Environment

```bash
git clone https://github.com/Chengsong-Huang/R-Zero.git
cd R-Zero
pip install -r requirements.txt
export STORAGE_PATH="/path/to/your/storage"
export HUGGINGFACENAME="yourhuggingfacename"
mkdir -p "$STORAGE_PATH/evaluation" "$STORAGE_PATH/models" "$STORAGE_PATH/generated_question" "$STORAGE_PATH/temp_results"
```

### 2. Set API Keys

Add your API keys to:

*   `tokens.json`: Hugging Face and WandB API keys.
*   `evaluation/results_recheck.py`: OpenAI GPT API key.

### 3. Run Experiments

Replicate our experimental results using a single script:

```bash
# Format: bash scripts/main.sh [Base_Model_Name] [Abbreviation]
bash scripts/main.sh Qwen/Qwen3-4B-Base qwen3-4b
```

## Impressive Results

R-Zero consistently enhances performance across various benchmarks.

| Model Name       | Overall AVG | MATH AVG | SuperGPQA | MMLU-Pro | BBEH  |
| :--------------- | :----------: | :------: | :-------: | :------: | :---: |
| ***Qwen3-4B-Base*** |      ...     |   ...    |    ...    |   ...    |  ...  |
|  &emsp;Base Model |    27.10     |  42.58   |   20.88   |  37.38   | 7.57  |
|  &emsp;R-Zero (Iter 3) |    **34.64**     |  **49.07**   |   27.55   |  **51.53**   | **10.42**  |
| ***Qwen3-8B-Base*** |      ...     |   ...    |    ...    |   ...    |  ...  |
| &emsp;R-Zero (Iter 3) |  **38.73**   |  **54.69**   |  31.38   |  **58.23**  | **10.60** |
| ***OctoThinker-3B***|      ...     |   ...    |    ...    |   ...    |  ...  |
| &emsp;R-Zero (Iter 3) |  **15.67**   |  **29.32**   |  **12.44**   |  **16.71**  |  4.20 |
| ***OctoThinker-8B***|      ...     |   ...    |    ...    |   ...    |  ...  |
| &emsp;R-Zero (Iter 3) |   **26.88**   |  **38.52**   |   **19.82**   |  40.92   |  **8.25** |

## Frequently Asked Questions

### Hardware Setup?

All experiments were conducted on an 8-GPU server using models that can run on a single GPU. If you need to run experiments under different conditions, such as with larger models or different hardware, you will need to modify the code accordingly.

### Environment Configuration Issues?

Refer to [EasyR1](https://github.com/hiyouga/EasyR1/tree/main) and consider their Docker environment as a reference.

### Saving Training Logs and Checkpoints?

The `STORAGE_PATH` environment variable specifies where all data, including logs, datasets, and model checkpoints, will be saved. Also dataset will be sent to huggingface via `HUGGINGFACENAME`.

### Code Gets Stuck?

Restart training from the last saved checkpoint if you encounter issues during the questioner training process.

## Acknowledgements

This work builds upon the foundation laid by [**EasyR1**](https://github.com/hiyouga/EasyR1/tree/main) and references the evaluation work from [**General-Reasoner**](https://github.com/TIGER-AI-Lab/General-Reasoner).

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