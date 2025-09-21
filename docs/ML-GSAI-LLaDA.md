# LLaDA: Revolutionizing Language Modeling with Diffusion (Large Language Diffusion with Masking)

**Explore LLaDA, a groundbreaking 8B parameter diffusion model that challenges traditional autoregressive language models, offering impressive performance and innovative capabilities.**  [Explore the original repository here](https://github.com/ML-GSAI/LLaDA)!

## Key Features

*   **Cutting-Edge Architecture:** LLaDA is an 8B-parameter diffusion model trained from scratch, offering a novel approach to language modeling.
*   **Competitive Performance:**  LLaDA achieves performance on par with LLaMA3 8B, demonstrating the potential of diffusion models in the LLM space.
*   **Multiple Variants:**  Explore different versions like LLaDA-MoE-7B-A1B and LLaDA-V for specialized tasks.
*   **Open-Source & Accessible:**  Pre-trained models, including LLaDA-8B-Base and LLaDA-8B-Instruct, are available on Hugging Face for easy deployment and experimentation.
*   **Flexible Inference:** Utilize provided scripts for conditional likelihood evaluation and conditional generation.
*   **Gradio Demo:**  Interact with LLaDA through an intuitive Gradio demo.
*   **Comprehensive Evaluation:**  Evaluate LLaDA using both conditional likelihood estimation and conditional generation, with evaluation code based on the lm-evaluation-harness library.

## What's New

*   **[2025.09.11]** Introduction of LLaDA-MoE-7B-A1B-Base and LLaDA-MoE-7B-A1B-Instruct, the first diffusion language model pretrained from scratch with MoE architecture.
*   **[2025.05.25]** Introduction of LLaDA 1.5, incorporating VRPO for improved preference alignment.
*   **[2025.05.23]** Introduction of LLaDA-V, a diffusion-based vision-language model.
*   **[2025.05.04]** Evaluation code based on the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) for LLaDA-Base.
*   **[2025.02.14]** Paper on [arXiv](https://arxiv.org/abs/2502.09992) and open-sourced LLaDA-8B-Base and LLaDA-8B-Instruct.

## Quickstart

### Installation

```bash
pip install transformers==4.38.2 gradio
```

### Inference

```python
from transformers import AutoModel, AutoTokenizer
import torch  # Import torch

tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Base', trust_remote_code=True)
model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Base', trust_remote_code=True, torch_dtype=torch.bfloat16).to("cuda") # Move model to GPU
```

Explore `get_log_likelihood.py` and `generate.py` for conditional evaluation and generation.

Run `python chat.py` for multi-round conversations with LLaDA-8B-Instruct.

### Gradio Demo

Run `python app.py` after installing Gradio.

## Training and Fine-tuning

Guidance for pre-training and SFT can be found in [GUIDELINES.md](GUIDELINES.md). Also, consider looking at [SMDM](https://github.com/ML-GSAI/SMDM) for similar training procedures.

## Evaluation

*   Conditional likelihood estimation and conditional generation.
*   Refer to [EVAL.md](EVAL.md) for evaluation code usage and bug details.
*   Find comprehensive details in Appendix B.5 of the [paper](https://arxiv.org/abs/2502.09992).

## FAQ

*   **How do I train my own LLaDA?** Consult [GUIDELINES.md](GUIDELINES.md) and [SMDM](https://github.com/ML-GSAI/SMDM).

*   **What is the difference between LLaDA and BERT?** LLaDA employs a randomly varying masking ratio (0-1) which makes LLaDA a generative model. The training objective of LLaDA is an upper bound on the negative log-likelihood of the model distribution.

*   **What is the relationship between LLaDA and Transformer?** LLaDA uses the Transformer architecture; the key difference is the probabilistic modeling approach where LLaDA uses a diffusion model.

*   **What is the sampling efficiency of LLaDA?** LLaDA's sampling is slower than autoregressive models. Optimization is in progress.

*   **What is the training stability of LLaDA?** See Section 2.2 of the paper for details.

*   **Why is the final answer "72" generated earlier than the intermediate calculation step (e.g., 12 × 4 = 48) in Tab4?** This is due to the remasking strategy.

*   **Why does LLaDA answer 'Bailing' when asked 'Who are you'?** Training data included identity markers.

*   **Our journey in developing LLaDA?** LLaDA builds on the prior works of [RADD](https://arxiv.org/abs/2406.03736) and [SMDM](https://arxiv.org/abs/2410.18514).

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

Join the conversation and stay updated by scanning the WeChat QR code in the original README.