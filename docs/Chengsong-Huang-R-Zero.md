# R-Zero: Revolutionizing Reasoning in LLMs with Self-Evolving Intelligence

**R-Zero is a groundbreaking framework that empowers Large Language Models to autonomously improve their reasoning skills without any pre-existing data, achieving state-of-the-art performance through a novel self-evolutionary process.** Check out the original repo [here](https://github.com/Chengsong-Huang/R-Zero).

## Key Features

*   **Autonomous Learning:** R-Zero starts from scratch, eliminating the need for labeled datasets or human intervention.
*   **Co-Evolutionary Architecture:** A "Challenger" and "Solver" model engage in a dynamic loop, creating a tailored curriculum for continuous improvement.
*   **Significant Performance Gains:** Achieves notable performance boosts on various reasoning benchmarks.
*   **Strong Generalization:** Reasoning skills learned are transferable to diverse reasoning tasks.
*   **Model Agnostic:** R-Zero consistently enhances the performance of different LLM backbones.

## Updates

*   **2025-8-27:** Analysis added on iteration scaling and a model taking on multiple roles.
*   **2025-8-25:** Code updates improve training stability (using `stopit`).
*   **2025-8-8:** R-Zero featured as `#2 Paper of the day` on [Hugging Face Daily Paper](https://huggingface.co/papers/2508.05004).
*   **2025-8-7:** Released paper and code ([arXiv](https://arxiv.org/abs/2508.05004)).

## Overview

[![R-Zero Overview](./figs/abstract.png)](https://arxiv.org/abs/2508.05004)

Training robust reasoning models traditionally relies on extensive, human-curated datasets, which are costly and challenging to scale. R-Zero presents a novel framework enabling LLMs to autonomously enhance their reasoning capabilities, without requiring any pre-existing tasks or labels. This is a truly self-evolving system.

At its core, R-Zero establishes a dynamic co-evolutionary loop between two instances of the same base model:

1.  **The Challenger 🎯:** This model probes the Solver for weaknesses by generating challenging problems tailored to the Solver's capabilities.
2.  **The Solver 🧠:** This model focuses on continuous improvement by solving the increasingly complex tasks posed by the Challenger.

This process creates a perfectly tailored, adaptive curriculum. The Challenger refines its question-generation skills, and the Solver improves its answer-finding abilities. The entire process is self-contained and leverages techniques such as majority voting for pseudo-labels and relative policy optimization to guide learning.

## Quickstart Guide

Get started with R-Zero using these steps:

### 1.  Environment Setup and Directory Configuration

```bash
git clone https://github.com/Chengsong-Huang/R-Zero.git
cd R-Zero
pip install -r requirements.txt
export STORAGE_PATH="/path/to/your/storage" # Replace with your desired storage path
export HUGGINGFACENAME="yourhuggingfacename" # Replace with your Hugging Face username
mkdir -p "$STORAGE_PATH/evaluation" "$STORAGE_PATH/models" "$STORAGE_PATH/generated_question" "$STORAGE_PATH/temp_results"
```

### 2. API Key Configuration

*   Populate `tokens.json` with your Hugging Face and WandB (for logging) API keys.
*   Add your OpenAI GPT API key in `evaluation/results_recheck.py` for evaluation.

### 3. Run Experiments

Execute the following script to reproduce the experimental results:

```bash
# Format: bash scripts/main.sh [Base_Model_Name] [Abbreviation]
# Example:
bash scripts/main.sh Qwen/Qwen3-4B-Base qwen3-4b
```

## Results

The table below compares the performance of different models, with peak performance highlighted in **bold**:

| Model Name        | Overall AVG | MATH AVG | SuperGPQA | MMLU-Pro | BBEH   |
| :---------------- | :----------: | :------: | :-------: | :------: | :----: |
| ***Qwen3-4B-Base*** |             |          |           |          |        |
| &emsp;Base Model  |     27.10    |   42.58  |    20.88  |   37.38  |  7.57  |
| &emsp;Base Challenger  |     30.83    |   44.36  |    24.77  |   47.59  |  6.59  |
| &emsp;R-Zero (Iter 1)  |     34.27    |   48.06  |  **27.92**  |   51.69  |  9.42  |
| &emsp;R-Zero (Iter 2)  |  **34.92**   |   48.44  |    27.72  |  **53.75** |  9.76  |
| &emsp;R-Zero (Iter 3)  |     34.64    |  **49.07** |    27.55  |   51.53  | **10.42** |
| ***Qwen3-8B-Base*** |             |          |           |          |        |
| &emsp;Base Model  |     34.49    |   49.18  |    28.33  |   51.80  |  8.63  |
| &emsp;Base Challenger  |     36.43    |   51.87  |    30.12  |   54.14  |  9.60  |
| &emsp;R-Zero (Iter 1)  |     37.93    |   53.39  |    31.26  |   57.17  |  9.91  |
| &emsp;R-Zero (Iter 2)  |     38.45    |   53.84  |  **31.58** |   58.20  | 10.20  |
| &emsp;R-Zero (Iter 3)  |  **38.73**   |  **54.69** |    31.38  |  **58.23** | **10.60** |
| ***OctoThinker-3B*** |             |          |           |          |        |
| &emsp;Base Model  |     12.27    |   26.64  |    10.09  |   10.87  |  1.46  |
| &emsp;Base Challenger  |     14.41    |   27.51  |    11.19  |   14.53  |  **4.40**  |
| &emsp;R-Zero (Iter 1)  |     14.93    |   27.76  |    12.21  |   15.72  |  4.05  |
| &emsp;R-Zero (Iter 2)  |     15.11    |   28.20  |    12.43  |   16.08  |  3.74  |
| &emsp;R-Zero (Iter 3)  |  **15.67**   |  **29.32** |  **12.44** |  **16.71** |  4.20  |
| ***OctoThinker-8B*** |             |          |           |          |        |
| &emsp;Base Model  |     16.81    |   32.11  |    13.26  |   20.21  |  1.64  |
| &emsp;Base Challenger  |     25.08    |   36.41  |    16.99  |   41.46  |  5.46  |
| &emsp;R-Zero (Iter 1)  |     26.44    |   37.80  |    19.15  |  **42.05** |  6.77  |
| &emsp;R-Zero (Iter 2)  |     26.77    |   38.23  |    19.27  |   41.34  |  **8.25**  |
| &emsp;R-Zero (Iter 3)  |  **26.88**   |  **38.52** |  **19.82** |   40.92  |  **8.25**  |

## FAQ for Developers

**Q: What hardware is required?**

**A:** Experiments were conducted on an 8-GPU server. The models used can run on a single GPU (e.g., 4B or 8B). For larger models or different hardware, adapt the code accordingly.

**Q:  What if I encounter environment configuration issues?**

**A:** R-Zero's framework structure is inspired by [EasyR1](https://github.com/hiyouga/EasyR1/tree/main). Refer to their setup instructions or Docker environment for help.

**Q: Where are training logs and model checkpoints saved?**

**A:** All generated data is stored in the directory specified by the `STORAGE_PATH` environment variable. Datasets will also be sent to Hugging Face using your `HUGGINGFACENAME`.

**Q: What if the code gets stuck during questioner training?**

**A:** This is likely caused by an infinite loop within the `math_verify` lib.  We have implemented a timeout control. If the issue occurs, restart training from the last saved checkpoint.

## Acknowledgements

This project is based on the work of [**EasyR1**](https://github.com/hiyouga/EasyR1/tree/main). Evaluation process referenced work from [**General-Reasoner**](https://github.com/TIGER-AI-Lab/General-Reasoner).

## Citation

If this project is helpful, please cite our paper:

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