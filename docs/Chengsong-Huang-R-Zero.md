# R-Zero: Revolutionizing LLM Reasoning with Self-Evolution 

**R-Zero is a groundbreaking framework that empowers Large Language Models to autonomously improve their reasoning abilities from scratch, eliminating the need for any training data.** ([Original Repo](https://github.com/Chengsong-Huang/R-Zero))

R-Zero enables Large Language Models to develop their reasoning capabilities without any pre-existing datasets. This framework utilizes a co-evolutionary loop between a Challenger and Solver to create a dynamic, self-improving system.

## Key Features

*   **Autonomous Learning:** R-Zero starts with a base model and requires no external data or pre-existing tasks.
*   **Co-Evolutionary Architecture:** A Challenger-Solver dynamic creates an adaptive curriculum, constantly refining both models.
*   **Proven Performance Boosts:** Achieves significant performance gains on various reasoning benchmarks.
*   **Strong Generalization:** Reasoning skills learned are transferable to general reasoning tasks.
*   **Model-Agnostic:** Improves performance across different LLM backbones.

## Updates

*   **2025-08-27:** Analysis on iteration scaling and one model taking on two roles has been added.
*   **2025-08-25:** Code updates to improve training smoothness (via stopit).
*   **2025-08-08:** R-Zero received `#2 Paper of the day` in [huggingface daily paper](https://huggingface.co/papers/2508.05004).
*   **2025-08-07:** Paper and code released.

## Quickstart Guide

Get up and running with R-Zero in a few simple steps:

### 1.  Environment Setup
```bash
git clone https://github.com/Chengsong-Huang/R-Zero.git
cd R-Zero
pip install -r requirements.txt
export STORAGE_PATH="/path/to/your/storage"  # Set a path for storing checkpoints and data
export HUGGINGFACENAME="yourhuggingfacename" #Set a name to push your dataset
mkdir -p "$STORAGE_PATH/evaluation" "$STORAGE_PATH/models" "$STORAGE_PATH/generated_question" "$STORAGE_PATH/temp_results"
```

### 2. API Keys
*   Add your **Hugging Face** and **WandB** API keys to `tokens.json`.
*   Add your **OpenAI GPT** API key to `evaluation/results_recheck.py` for evaluation.

### 3. Run Experiments
```bash
# Run experiments using the base model name and an abbreviation
# Format: bash scripts/main.sh [Base_Model_Name] [Abbreviation]
# Example:
bash scripts/main.sh Qwen/Qwen3-4B-Base qwen3-4b
```

## Impressive Results

R-Zero demonstrates impressive performance improvements over base models and challenger baselines across several benchmarks.

| Model Name         | Overall AVG | MATH AVG | SuperGPQA | MMLU-Pro | BBEH  |
| :----------------- | :----------: | :------: | :-------: | :------: | :---: |
| ***Qwen3-4B-Base*** |              |          |           |          |       |
| &emsp;Base Model    |     27.10    |   42.58  |    20.88  |   37.38  |  7.57 |
| &emsp;Base Challenger|     30.83    |   44.36  |    24.77  |   47.59  |  6.59 |
| &emsp;R-Zero (Iter 1) |     34.27    |   48.06  |   **27.92** |   51.69  |  9.42 |
| &emsp;R-Zero (Iter 2) |   **34.92**  |   48.44  |    27.72  |  **53.75** |  9.76 |
| &emsp;R-Zero (Iter 3) |     34.64    |  **49.07** |    27.55  |   51.53  | **10.42** |
| ***Qwen3-8B-Base*** |              |          |           |          |       |
| &emsp;Base Model    |     34.49    |   49.18  |    28.33  |   51.80  |  8.63 |
| &emsp;Base Challenger|     36.43    |   51.87  |    30.12  |   54.14  |  9.60 |
| &emsp;R-Zero (Iter 1) |     37.93    |   53.39  |    31.26  |   57.17  |  9.91 |
| &emsp;R-Zero (Iter 2) |     38.45    |   53.84  |   **31.58** |   58.20  | 10.20 |
| &emsp;R-Zero (Iter 3) |   **38.73**  |  **54.69** |    31.38  |  **58.23** | **10.60** |
| ***OctoThinker-3B*** |              |          |           |          |       |
| &emsp;Base Model    |     12.27    |   26.64  |    10.09  |   10.87  |  1.46 |
| &emsp;Base Challenger|     14.41    |   27.51  |    11.19  |   14.53  |  **4.40** |
| &emsp;R-Zero (Iter 1) |     14.93    |   27.76  |    12.21  |   15.72  |  4.05 |
| &emsp;R-Zero (Iter 2) |     15.11    |   28.20  |    12.43  |   16.08  |  3.74 |
| &emsp;R-Zero (Iter 3) |   **15.67**  |  **29.32** |   **12.44** |  **16.71** |  4.20 |
| ***OctoThinker-8B*** |              |          |           |          |       |
| &emsp;Base Model    |     16.81    |   32.11  |    13.26  |   20.21  |  1.64 |
| &emsp;Base Challenger|     25.08    |   36.41  |    16.99  |   41.46  |  5.46 |
| &emsp;R-Zero (Iter 1) |     26.44    |   37.80  |    19.15  |  **42.05** |  6.77 |
| &emsp;R-Zero (Iter 2) |     26.77    |   38.23  |    19.27  |   41.34  |  **8.25** |
| &emsp;R-Zero (Iter 3) |   **26.88**  |  **38.52** |   **19.82** |   40.92  |  **8.25** |

## FAQ for Developers

### Hardware Requirements

Experiments were conducted on an 8-GPU server. You may need to modify code for larger models or different hardware.

### Environment Configuration

Refer to [EasyR1](https://github.com/hiyouga/EasyR1/tree/main) for environment setup guidance or use their Docker environment as a reference.

### Data and Checkpoint Location

All generated data, logs, and model checkpoints are saved in the directory specified by the `STORAGE_PATH` environment variable.  Datasets are also sent to Hugging Face via the `HUGGINGFACENAME` env variable.

### Troubleshooting

If the code gets stuck during questioner training, restart from the last saved checkpoint.

## Acknowledgements

R-Zero is built upon the foundational work of [**EasyR1**](https://github.com/hiyouga/EasyR1/tree/main) and references the evaluation methodology of [**General-Reasoner**](https://github.com/TIGER-AI-Lab/General-Reasoner).

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