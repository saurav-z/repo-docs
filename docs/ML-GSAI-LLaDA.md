# LLaDA: Revolutionizing Language Modeling with Diffusion Models

**LLaDA introduces a novel approach to language modeling, utilizing diffusion models to achieve state-of-the-art performance and unlock new capabilities, rivaling LLaMA3 8B.** Explore the future of language AI with this groundbreaking model. For more details, visit the original repository: [LLaDA GitHub](https://github.com/ML-GSAI/LLaDA).

## Key Features

*   **Diffusion-Based Language Modeling:** LLaDA utilizes diffusion models, a novel approach to language modeling, achieving impressive performance in the field.
*   **8 Billion Parameter Scale:** LLaDA boasts an impressive 8B parameter count, demonstrating the power of the model.
*   **Competitive Performance:** LLaDA rivals the performance of the LLaMA3 8B model, showcasing its efficiency and power.
*   **Open Source:** Explore the LLaDA-8B-Base and LLaDA-8B-Instruct models available on Hugging Face.
*   **Versatile Capabilities:** LLaDA excels at in-context learning and instruction following.
*   **Comprehensive Evaluation:** Utilize evaluation code based on the lm-evaluation-harness library.

## What's New

*   **[2025.09.11]** Introduced LLaDA-MoE-7B-A1B-Base and LLaDA-MoE-7B-A1B-Instruct, the first diffusion language model pretrained from scratch with MoE architecture.
*   **[2025.05.25]** Introduced LLaDA 1.5, incorporating VRPO.
*   **[2025.05.23]** Introduced LLaDA-V, a competitive diffusion-based vision-language model.
*   **[2025.05.04]** Evaluation code based on the lm-evaluation-harness has been provided for the LLaDA-Base model.
*   **[2025.02.14]** Paper published on arXiv and open-sourced LLaDA-8B-Base and LLaDA-8B-Instruct.

## Quick Start - Inference

Get started with LLaDA by installing the necessary dependencies and loading the models:

```bash
pip install transformers==4.38.2
```

```python
from transformers import AutoModel, AutoTokenizer
import torch # Add this line

tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Base', trust_remote_code=True)
model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Base', trust_remote_code=True, torch_dtype=torch.bfloat16)
```

Explore the `get_log_likelihood.py` and `generate.py` scripts for conditional likelihood evaluation and conditional generation.

## Gradio Demo

Experience LLaDA firsthand with the interactive Gradio demo.

*   Run the demo using `python app.py` after installing Gradio: `pip install gradio`.

## Training and Evaluation

*   **Pre-training and SFT:** Follow the guidelines in [GUIDELINES.md](GUIDELINES.md) for pre-training and SFT. You can also refer to [SMDM](https://github.com/ML-GSAI/SMDM).
*   **Evaluation:** Detailed evaluation information is available in the [EVAL.md](EVAL.md) file and our [paper](https://arxiv.org/abs/2502.09992).

## Frequently Asked Questions (FAQ)

Address common questions about LLaDA:

### 0. How do I train my own LLaDA?

*   Refer to [GUIDELINES.md](GUIDELINES.md) for training guidelines. Also, see [SMDM](https://github.com/ML-GSAI/SMDM).

### 1. What is the difference between LLaDA and BERT?

*   LLaDA uses a masking ratio that varies randomly (0-1), unlike BERT's fixed ratio.  LLaDA's training objective is an upper bound on the negative log-likelihood, making it a generative model with in-context learning and instruction-following abilities.

### 2. What is the relationship between LLaDA and Transformer?

*   LLaDA adopts the Transformer architecture. The key difference is in the probabilistic modeling approach: GPT uses autoregressive next-token prediction, while LLaDA employs a diffusion model.

### 3. What is the sampling efficiency of LLaDA?

*   LLaDA's sampling speed is currently slower than autoregressive models. The sampling speed is being optimized, and there is significant room for improvement.

### 4. What is the training stability of LLaDA?

*   During training on 2.3T tokens, a crash occurred once. It was resolved by resuming the checkpoint and decreasing the learning rate.

### 5. Why is the final answer "72" generated earlier than the intermediate calculation step (e.g., 12 × 4 = 48) in Tab4?

*   The mask predictor correctly predicted the reasoning process, but during the remasking, reasoning steps were masked out again.

### 6. Why does LLaDA answer 'Bailing' when asked 'Who are you'?

*   This is due to the design of the pre-training and SFT data, which were designed for autoregressive models.

### 7. Our journey in developing LLaDA?

*   LLaDA builds upon the authors' work on [RADD](https://arxiv.org/abs/2406.03736) and [SMDM](https://arxiv.org/abs/2410.18514).

## Citation

```bibtex
@article{nie2025large,
  title={Large Language Diffusion Models},
  author={Nie, Shen and Zhu, Fengqi and You, Zebin and Zhang, Xiaolu and Ou, Jingyang and Hu, Jun and Zhou, Jun and Lin, Yankai and Wen, Ji-Rong and Li, Chongxuan},
  journal={arXiv preprint arXiv:2502.09992},
  year={2025}
}
```

## Discussion

Join the conversation and stay updated on the latest LLaDA developments.  Scan the WeChat QR code (included in the original README) to participate.