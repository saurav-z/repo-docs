# LLaDA: Revolutionizing Language Modeling with Diffusion Models

**LLaDA introduces a groundbreaking approach to language modeling, leveraging diffusion techniques to achieve state-of-the-art performance, challenging the dominance of autoregressive models. Check out the original repo for more details: [https://github.com/ML-GSAI/LLaDA](https://github.com/ML-GSAI/LLaDA)**

## Key Features

*   **Cutting-Edge Architecture:** LLaDA (Large Language Diffusion with Masking) utilizes a diffusion model with an 8B parameter scale, trained from scratch.
*   **Competitive Performance:** LLaDA rivals LLaMA3 8B in performance, demonstrating the power of diffusion models in the language domain.
*   **Open-Source Models:** Access to both [LLaDA-8B-Base](https://huggingface.co/GSAI-ML/LLaDA-8B-Base) and [LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) models available on Hugging Face.
*   **Gradio Demo:** Interact with LLaDA through a user-friendly Gradio demo.
*   **Evaluation Code:** Evaluation code based on the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) for the LLaDA-Base.
*   **Ongoing Development:** Continuous improvements with recent releases like LLaDA-MoE and LLaDA-V.
*   **Inference Made Easy:** Simple instructions for loading and using LLaDA with the `transformers` library.
*   **Training Guides:** Resources and guidelines for pre-training and supervised fine-tuning (SFT) are available.

## What's New
*   [2025.09.11] We introduce [LLaDA-MoE-7B-A1B-Base](https://huggingface.co/inclusionAI/LLaDA-MoE-7B-A1B-Base) and [LLaDA-MoE-7B-A1B-Instruct](https://huggingface.co/inclusionAI/LLaDA-MoE-7B-A1B-Instruct), the first diffusion language model pretrained from scratch with MoE architecture. LLaDA-MoE-7B-A1B-Instruct use only ~1B active parameters at inference while surpassing LLaDA 1.5(an 8B dense model), and comparable to Qwen2.5-3B-Instruct.
*   [2025.05.25] We introduce [LLaDA 1.5](https://ml-gsai.github.io/LLaDA-1.5-Demo/), which incorporates VRPO to reduce gradient variance and enhance preference alignment in LLaDA.
*   [2025.05.23] We introduce [LLaDA-V](https://ml-gsai.github.io/LLaDA-V-demo/), a competitive diffusion-based vision-language model, outperforming other diffusion MLLMs.
*   [2025.05.04] We have provided evaluation code based on the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) for the LLaDA-Base.
*   [2025.02.14] We have uploaded our paper to [arXiv](https://arxiv.org/abs/2502.09992) and open-sourced [LLaDA-8B-Base](https://huggingface.co/GSAI-ML/LLaDA-8B-Base) and [LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct).

## Getting Started

### Inference

*   Install the `transformers` library: `pip install transformers==4.38.2`
*   Load the model using the `transformers` library:

```python
from transformers import AutoModel, AutoTokenizer
import torch  # Import torch

tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Base', trust_remote_code=True)
model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Base', trust_remote_code=True, torch_dtype=torch.bfloat16).to('cuda')  # Move model to GPU
```

*   Use `get_log_likelihood.py` and `generate.py` for conditional likelihood evaluation and generation.
*   Run `python chat.py` for multi-round conversations with LLaDA-8B-Instruct.
*   Refer to the [GUIDELINES.md](GUIDELINES.md) for more details.

### Gradio Demo

*   Install Gradio: `pip install gradio`
*   Run the demo: `python app.py`

## Training

*   Guidelines for pre-training and SFT are provided in [GUIDELINES.md](GUIDELINES.md).
*   Refer to [SMDM](https://github.com/ML-GSAI/SMDM) for a similar training process.

## Evaluation

*   The project uses conditional likelihood estimation and conditional generation for evaluation.
*   Evaluation code for LLaDA-Base is available. Refer to [EVAL.md](EVAL.md) for usage and bug details.

## Frequently Asked Questions (FAQ)

*   **How do I train my own LLaDA?**
    *   See [GUIDELINES.md](GUIDELINES.md) and [SMDM](https://github.com/ML-GSAI/SMDM).
*   **What is the difference between LLaDA and BERT?**
    *   LLaDA is a generative model with a varying masking ratio, providing an upper bound on negative log-likelihood.
*   **What is the relationship between LLaDA and Transformer?**
    *   LLaDA uses the Transformer architecture with a diffusion model for probabilistic modeling.
*   **What is the sampling efficiency of LLaDA?**
    *   Currently slower than autoregressive models, with ongoing optimization efforts.
*   **What is the training stability of LLaDA?**
    *   Training stability details are available in Section 2.2 of the paper.
*   **Why is the final answer "72" generated earlier than the intermediate calculation step (e.g., 12 × 4 = 48) in Tab4?**
    *   The remasking process may lead to the reasoning steps being masked out again.
*   **Why does LLaDA answer 'Bailing' when asked 'Who are you'?**
    *   Due to the nature of the pre-training and SFT data.
*   **Our journey in developing LLaDA?**
    *   LLaDA builds upon prior work: [RADD](https://arxiv.org/abs/2406.03736) and [SMDM](https://arxiv.org/abs/2410.18514).

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

Engage with the LLaDA community and stay updated via the WeChat QR code:

<div style="display: flex; justify-content: center; flex-wrap: wrap;">
    <img src="./imgs/QR.jpg" style="width: 50%" />
</div>