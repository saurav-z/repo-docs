# R-Zero: Revolutionizing LLM Reasoning with Self-Evolution 

**R-Zero empowers Large Language Models to autonomously learn and improve reasoning skills from scratch, eliminating the need for any pre-existing training data.** ([Original Repo](https://github.com/Chengsong-Huang/R-Zero))

## Key Features

*   **Zero-Shot Learning:** Train LLMs without any prior data or labeled examples.
*   **Self-Evolving Architecture:**  A Challenger-Solver dynamic generates a tailored curriculum for continuous improvement.
*   **Proven Performance:** Achieve significant performance boosts on reasoning benchmarks.
*   **Generalization:** Reasoning skills learned in specific domains transfer to broader reasoning tasks.
*   **Model Agnostic:** Compatible and effective across various base LLM architectures.

## 🚀 Updates

*   **[2025-8-27]** Analysis on iteration scaling and role assignment (one model taking on two roles) added.
*   **[2025-8-25]** Code updates for smoother training (using `stopit`).
*   **[2025-8-8]**  Recognized as `#2 Paper of the day` on [huggingface daily paper](https://huggingface.co/papers/2508.05004).
*   **[2025-8-7]** Paper and code released.

## 💡 Overview: How R-Zero Works

R-Zero's innovative approach eliminates the need for large, hand-curated datasets. It leverages a co-evolutionary loop between two components:

1.  **Challenger:**  Generates challenging problems to identify the Solver's weaknesses.
2.  **Solver:** Improves its reasoning abilities by solving the problems posed by the Challenger.

This iterative process creates an adaptive learning environment, allowing both the Challenger and Solver to refine their capabilities. The system employs techniques like majority voting for pseudo-labels and relative policy optimization to drive learning, resulting in a self-contained and efficient learning loop.

<img src="./figs/abstract.png" alt="R-Zero Overview" width="600"/>

## 🛠 Quickstart Guide

Follow these steps to get started:

### 1. Set Up Your Environment

```bash
git clone https://github.com/Chengsong-Huang/R-Zero.git
cd R-Zero
pip install -r requirements.txt

export STORAGE_PATH="/path/to/your/storage" # Set your storage path
export HUGGINGFACENAME="yourhuggingfacename" # Your Hugging Face username

mkdir -p "$STORAGE_PATH/evaluation" "$STORAGE_PATH/models" "$STORAGE_PATH/generated_question" "$STORAGE_PATH/temp_results"
```

### 2. Configure API Keys

*   Add your Hugging Face and WandB API keys to `tokens.json`.
*   Add your OpenAI GPT API key to `evaluation/results_recheck.py`.

### 3. Run Experiments

Replicate the results with a single command:

```bash
# Format: bash scripts/main.sh [Base_Model_Name] [Abbreviation]
bash scripts/main.sh Qwen/Qwen3-4B-Base qwen3-4b # Example
```

## 📊 Impressive Results

R-Zero demonstrates significant improvements across multiple reasoning benchmarks.

| Model Name           | Overall AVG | MATH AVG | SuperGPQA | MMLU-Pro | BBEH  |
| :------------------- | :----------: | :------: | :-------: | :------: | :---: |
| ***Qwen3-4B-Base***   |      ...     |   ...    |    ...    |   ...    |  ...  |
| &emsp;Base Model     |     27.10    |   42.58  |    20.88  |   37.38  |  7.57 |
| &emsp;Base Challenger|     30.83    |   44.36  |    24.77  |   47.59  |  6.59 |
| &emsp;R-Zero (Iter 1) |     34.27    |   48.06  | **27.92** |   51.69  |  9.42 |
| &emsp;R-Zero (Iter 2) | **34.92**    |   48.44  |    27.72  | **53.75** |  9.76 |
| &emsp;R-Zero (Iter 3) |     34.64    | **49.07**  |    27.55  |   51.53  | **10.42** |
| ...                  |      ...     |   ...    |    ...    |   ...    |  ...  |

*See original README for full table.*

## ❓ FAQ for Developers

### Q: Hardware Setup?
**A:** Experiments were conducted on an 8-GPU server. You may need to adjust code for larger models or different hardware.

### Q: Environment Configuration Issues?
**A:** Refer to [EasyR1](https://github.com/hiyouga/EasyR1/tree/main) for setup guidance.

### Q: Where are the Training Logs/Checkpoints Saved?
**A:**  Generated data is saved in the directory specified by the `STORAGE_PATH` environment variable and uploaded to Hugging Face via `HUGGINGFACENAME`.

### Q: Code Stuck During Questioner Training?
**A:**  A timeout has been added to address potential infinite loops in the `math_verify` library. Restart training from the last checkpoint if needed.

## 🙏 Acknowledgements

We are thankful for the foundational work of [EasyR1](https://github.com/hiyouga/EasyR1/tree/main) and the evaluation process from [General-Reasoner](https://github.com/TIGER-AI-Lab/General-Reasoner).

## 💬 Citation

If you use our work, please cite our paper:

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