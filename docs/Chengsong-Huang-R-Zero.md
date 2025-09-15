# R-Zero: Self-Evolving Reasoning LLMs from Zero Data

**R-Zero empowers Large Language Models (LLMs) to learn and evolve their reasoning abilities autonomously, entirely without the need for pre-existing data.** ([Original Repo](https://github.com/Chengsong-Huang/R-Zero))

## Key Features

*   **Autonomous Learning:** Trains LLMs from scratch, eliminating the need for curated datasets or labeled examples.
*   **Co-Evolutionary Loop:** Employs a Challenger-Solver dynamic for continuous improvement and adaptive curriculum generation.
*   **Performance Boost:** Achieves significant gains on various reasoning benchmarks.
*   **Generalization:** Demonstrated skills transfer across different reasoning tasks and domains.
*   **Model Agnostic:** Compatible with a variety of base LLMs, enhancing their reasoning performance.

## 🚀 Updates

*   **2025-08-27:** Added analysis on iteration scaling and one model taking on two roles.
*   **2025-08-25:** Code updates for smoother training (using stopit).
*   **2025-08-08:** Recognized as `#2 Paper of the day` on [Hugging Face Daily Paper](https://huggingface.co/papers/2508.05004).
*   **2025-08-07:** Released the [paper](https://arxiv.org/abs/2508.05004) and code.

## 💡 Overview

R-Zero is a groundbreaking framework that addresses the limitations of traditional LLM training methods by enabling LLMs to enhance their reasoning capabilities in a fully autonomous manner.

R-Zero establishes a co-evolutionary loop between two instances of the same base model:

1.  **Challenger 🎯:** This instance generates challenging reasoning problems, pushing the Solver's boundaries.
2.  **Solver 🧠:**  Continuously improves its reasoning skills by solving tasks presented by the Challenger.

This creates a dynamically tailored learning experience, with the Challenger learning to ask better questions and the Solver developing more robust answers. The entire process is self-contained, relying on techniques like majority voting for pseudo-labels and relative policy optimization to optimize learning.

## ⚡ Quickstart

Get started quickly by following these steps:

### 1. Configure Environment & Prepare Directories

```bash
git clone https://github.com/Chengsong-Huang/R-Zero.git
cd R-Zero
pip install -r requirements.txt
export STORAGE_PATH="/path/to/your/storage"  # Set your storage path
export HUGGINGFACENAME="yourhuggingfacename"  # Set your Hugging Face name
mkdir -p "$STORAGE_PATH/evaluation" "$STORAGE_PATH/models" "$STORAGE_PATH/generated_question" "$STORAGE_PATH/temp_results"
```

### 2. Add API Keys

You'll need to provide your API keys:

*   Add Hugging Face and WandB API keys to `tokens.json`.
*   Add your OpenAI GPT API key in `evaluation/results_recheck.py`.

### 3. Run Experiments

Replicate experimental results with a single script:

```bash
# Format: bash scripts/main.sh [Base_Model_Name] [Abbreviation]
# Example:
bash scripts/main.sh Qwen/Qwen3-4B-Base qwen3-4b
```

## 📊 Impressive Results

R-Zero delivers significant performance improvements, as shown in the table below.  The best results for each model are highlighted in **bold**.

| Model Name          | Overall AVG | MATH AVG | SuperGPQA | MMLU-Pro | BBEH    |
| :------------------ | :----------: | :------: | :-------: | :------: | :-----: |
| ***Qwen3-4B-Base*** |              |          |           |          |         |
| &emsp;Base Model   |    27.10     |   42.58  |    20.88  |   37.38  |  7.57   |
| &emsp;Base Challenger |    30.83     |   44.36  |    24.77  |   47.59  |  6.59   |
| &emsp;R-Zero (Iter 1) |    34.27     |   48.06  |  **27.92** |   51.69  |  9.42   |
| &emsp;R-Zero (Iter 2) |  **34.92**   |   48.44  |    27.72  |  **53.75** |  9.76   |
| &emsp;R-Zero (Iter 3) |    34.64     |  **49.07** |    27.55  |   51.53  | **10.42** |
| ***Qwen3-8B-Base*** |              |          |           |          |         |
| &emsp;Base Model   |    34.49     |   49.18  |    28.33  |   51.80  |  8.63   |
| &emsp;Base Challenger |    36.43     |   51.87  |    30.12  |   54.14  |  9.60   |
| &emsp;R-Zero (Iter 1) |    37.93     |   53.39  |    31.26  |   57.17  |  9.91   |
| &emsp;R-Zero (Iter 2) |    38.45     |   53.84  |  **31.58** |   58.20  | 10.20   |
| &emsp;R-Zero (Iter 3) |  **38.73**   |  **54.69** |    31.38  |  **58.23** | **10.60** |
| ***OctoThinker-3B*** |              |          |           |          |         |
| &emsp;Base Model   |    12.27     |   26.64  |    10.09  |   10.87  |  1.46   |
| &emsp;Base Challenger |    14.41     |   27.51  |    11.19  |   14.53  |  **4.40**   |
| &emsp;R-Zero (Iter 1) |    14.93     |   27.76  |    12.21  |   15.72  |  4.05   |
| &emsp;R-Zero (Iter 2) |    15.11     |   28.20  |    12.43  |   16.08  |  3.74   |
| &emsp;R-Zero (Iter 3) |  **15.67**   |  **29.32** |  **12.44** |  **16.71** |  4.20   |
| ***OctoThinker-8B*** |              |          |           |          |         |
| &emsp;Base Model   |    16.81     |   32.11  |    13.26  |   20.21  |  1.64   |
| &emsp;Base Challenger |    25.08     |   36.41  |    16.99  |   41.46  |  5.46   |
| &emsp;R-Zero (Iter 1) |    26.44     |   37.80  |    19.15  |  **42.05** |  6.77   |
| &emsp;R-Zero (Iter 2) |    26.77     |   38.23  |    19.27  |   41.34  |  **8.25**   |
| &emsp;R-Zero (Iter 3) |  **26.88**   |  **38.52** |  **19.82** |   40.92  |  **8.25**   |

## ❓ FAQ for Developers

### Q: What hardware is required?

**A:**  Experiments were conducted on an 8-GPU server, using models that can run on a single GPU (e.g., 4B or 8B).  Adjust the code for larger models or different hardware.

### Q: How to resolve environment configuration issues?

**A:** R-Zero is based on the work of [EasyR1](https://github.com/hiyouga/EasyR1/tree/main).  Refer to their setup instructions or use their Docker environment for environment issues.

### Q: Where are training artifacts saved?

**A:**  Logs, datasets, and model checkpoints are saved in the directory specified by the `STORAGE_PATH` environment variable.  Datasets are also sent to Hugging Face using your `HUGGINGFACENAME`.

### Q: What to do if code gets stuck during training?

**A:** A timeout mechanism is implemented, but the `math_verify` lib can cause an infinite loop. Restart training from the last checkpoint.
<!-- >> I suddenly find there is a lib named `timeout_decorator` which can solve this problem after I complete most of the experiments...... (not sure whether it will introduce new problems.) -->

## 🙏 Acknowledgements

This project is inspired by [**EasyR1**](https://github.com/hiyouga/EasyR1/tree/main), and the evaluation process referenced the work from [**General-Reasoner**](https://github.com/TIGER-AI-Lab/General-Reasoner).

## 💬 Citation

If you use this work, please cite our paper:

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