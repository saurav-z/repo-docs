# LLaDA: Large Language Diffusion Models - Revolutionizing Language Modeling

**LLaDA introduces a groundbreaking approach to language modeling, achieving impressive performance with an 8B-parameter diffusion model, challenging the dominance of autoregressive models.** Learn more about LLaDA on [GitHub](https://github.com/ML-GSAI/LLaDA).

**Key Features:**

*   **Diffusion-Based Approach:** LLaDA leverages a diffusion model for language generation, offering a novel perspective on text modeling.
*   **8B Parameter Scale:**  Trained from scratch, LLaDA rivals the performance of LLaMA3 8B.
*   **Openly Available Models:** Explore [LLaDA-8B-Base](https://huggingface.co/GSAI-ML/LLaDA-8B-Base) and [LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) on Hugging Face.
*   **Inference and Generation:** Includes scripts for conditional likelihood evaluation and text generation.
*   **Interactive Demo:** Experiment with LLaDA through the Gradio demo, allowing multi-round conversations.
*   **Evaluation Code:** Evaluation code based on the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) is provided for LLaDA-Base.
*   **Versatile Applications:** Designed for in-context learning, instruction-following, and exhibits Fisher consistency for scalability.

## What is LLaDA?

LLaDA (<b>L</b>arge <b>La</b>nguage <b>D</b>iffusion with m<b>A</b>sking) is a state-of-the-art large language model that leverages a diffusion-based approach, pushing the boundaries of text generation. Unlike traditional autoregressive models, LLaDA explores a new paradigm for language modeling with an 8B parameter scale.

<div style="display: flex; justify-content: center; flex-wrap: wrap;">
    <img src="./imgs/LLaDA_vs_LLaMA.svg" style="width: 45%" />
    <img src="./imgs/LLaDA_vs_LLaMA_chat.svg" style="width: 46%" />
</div>

## Getting Started

### Inference

To use LLaDA-8B-Base or LLaDA-8B-Instruct, install `transformers==4.38.2`.

```bash
pip install transformers==4.38.2
```

Then, load the model and tokenizer:

```python
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Base', trust_remote_code=True)
model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Base', trust_remote_code=True, torch_dtype=torch.bfloat16)
```

Use `get_log_likelihood.py` and `generate.py` for conditional likelihood evaluation and generation. Run `python chat.py` for interactive conversations with LLaDA-8B-Instruct. For more details, consult the paper and [GUIDELINES.md](GUIDELINES.md).

### Gradio Demo

Run `python app.py` after installing Gradio (`pip install gradio`) to access the interactive demo.

<div style="display: flex; justify-content: center; flex-wrap: wrap;">
    <img src="./imgs/example_gradio.gif" style="width: 80%" />
</div>

## Training and Evaluation

### Pre-training and Supervised Fine-Tuning

While we do not provide the training framework directly, pre-training and Supervised Fine-Tuning of LLaDA are relatively straightforward. Adapt your existing autoregressive model training code with minimal changes. Consult [GUIDELINES.md](GUIDELINES.md) and [SMDM](https://github.com/ML-GSAI/SMDM) for detailed guidance.

### Evaluation

LLaDA is evaluated using conditional likelihood estimation and conditional generation. Evaluation code is provided for LLaDA-Base based on the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness). See [EVAL.md](EVAL.md) for usage and details.

## Frequently Asked Questions (FAQ)

### 0. How do I train my own LLaDA?

Refer to [GUIDELINES.md](GUIDELINES.md) and [SMDM](https://github.com/ML-GSAI/SMDM) for training guidance.

### 1. What is the difference between LLaDA and BERT?

LLaDA's masked diffusion model training objective is an upper bound on the negative log-likelihood, making it a generative model, unlike BERT's fixed masking ratio.

### 2. What is the relationship between LLaDA and Transformer?

LLaDA utilizes a Transformer architecture but employs a diffusion model for probabilistic modeling, diverging from the autoregressive approach of GPT.

### 3. What is the sampling efficiency of LLaDA?

Sampling is currently slower than autoregressive models due to context length, KV-Cache limitations, and the need for multiple sampling steps for optimal performance. Optimization efforts are ongoing.

### 4. What is the training stability of LLaDA?

Training stability is described in Section 2.2 of the paper.

### 5. Why is the final answer "72" generated earlier than the intermediate calculation step (e.g., 12 × 4 = 48) in Tab4?

The mask predictor predicts the reasoning steps, which may be masked out during remasking, influencing the generation order.

### 6. Why does LLaDA answer 'Bailing' when asked 'Who are you'?

The model was trained with data containing identity markers.

### 7. Our journey in developing LLaDA?

LLaDA builds upon [RADD](https://arxiv.org/abs/2406.03736) and [SMDM](https://arxiv.org/abs/2410.18514), which explored the theoretical foundations and scaling laws of masked diffusion models.

## Updates

*   **[2025.05.25]** Introduction of [LLaDA 1.5](https://ml-gsai.github.io/LLaDA-1.5-Demo/) to reduce gradient variance.
*   **[2025.05.23]** Introduction of [LLaDA-V](https://ml-gsai.github.io/LLaDA-V-demo/), a diffusion-based vision-language model.
*   **[2025.05.04]** Evaluation code based on the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) for LLaDA-Base.
*   **[2025.02.14]** Paper uploaded to [arXiv](https://arxiv.org/abs/2502.09992) and open-sourced [LLaDA-8B-Base](https://huggingface.co/GSAI-ML/LLaDA-8B-Base) and [LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct).

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

Join the discussion via the WeChat QR code to stay updated.

<div style="display: flex; justify-content: center; flex-wrap: wrap;">
    <img src="./imgs/QR.jpg" style="width: 50%" />
</div>