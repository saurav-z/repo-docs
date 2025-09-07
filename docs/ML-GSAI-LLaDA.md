# LLaDA: Revolutionizing Language Modeling with Diffusion [LLaDA on GitHub](https://github.com/ML-GSAI/LLaDA)

**LLaDA (Large Language Diffusion with Masking) is an 8B-parameter diffusion model that challenges the limits of language modeling by achieving performance comparable to LLaMA3 8B, trained from scratch.**

**Key Features:**

*   **State-of-the-Art Performance:** LLaDA demonstrates impressive capabilities, rivaling leading autoregressive models.
*   **Diffusion-Based Approach:** Utilizes a unique diffusion model architecture for language generation.
*   **Openly Available Models:**  Deployable models are available on Hugging Face for both base and instruction-tuned versions:
    *   [LLaDA-8B-Base](https://huggingface.co/GSAI-ML/LLaDA-8B-Base)
    *   [LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct)
*   **Evaluation Code:** Evaluation code based on the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) is provided.
*   **Gradio Demo:** Interactive demonstration available for easy experimentation. ([Demo Link](https://huggingface.co/spaces/multimodalart/LLaDA))
*   **Comprehensive Documentation:** Includes guidelines for training, evaluation, and understanding LLaDA's architecture and behavior.

## What's New

*   **[2025.05.25]** Introducing [LLaDA 1.5](https://ml-gsai.github.io/LLaDA-1.5-Demo/), which incorporates VRPO to reduce gradient variance and enhance preference alignment in LLaDA.
*   **[2025.05.23]** Introducing [LLaDA-V](https://ml-gsai.github.io/LLaDA-V-demo/), a competitive diffusion-based vision-language model, outperforming other diffusion MLLMs.
*   **[2025.05.04]** Evaluation code based on the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) for LLaDA-Base is available.
*   **[2025.02.14]**  Paper on [arXiv](https://arxiv.org/abs/2502.09992) and open-sourced LLaDA-8B-Base and LLaDA-8B-Instruct models.

## Introduction

LLaDA (Large Language Diffusion with Masking) is a groundbreaking 8B-parameter diffusion model, trained from scratch, that competes with the performance of LLaMA3 8B. LLaDA leverages a novel diffusion-based approach to language modeling, demonstrating strong performance and offering a fresh perspective on model design.

<div style="display: flex; justify-content: center; flex-wrap: wrap;">
    <img src="./imgs/LLaDA_vs_LLaMA.svg" style="width: 45%" />
    <img src="./imgs/LLaDA_vs_LLaMA_chat.svg" style="width: 46%" />
</div>

## Quick Start - Inference

Get started with LLaDA by installing the necessary libraries and loading the models from Hugging Face:

```bash
pip install transformers==4.38.2
```

```python
from transformers import AutoModel, AutoTokenizer
import torch  # Make sure to import torch if you're using it

tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Base', trust_remote_code=True)
model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Base', trust_remote_code=True, torch_dtype=torch.bfloat16)
```

For conditional likelihood evaluation and conditional generation, use the provided `get_log_likelihood.py` and `generate.py` scripts. You can also have multi-round conversations with LLaDA-8B-Instruct by running `python chat.py`.

## Gradio Demo

Explore LLaDA's capabilities through an interactive Gradio demo.

First, install [Gradio](https://www.gradio.app) `pip install gradio`, and then you can directly run `python app.py`

<div style="display: flex; justify-content: center; flex-wrap: wrap;">
    <img src="./imgs/example_gradio.gif" style="width: 80%" />
</div>

## Training

While the full training framework isn't provided, the pre-training and Supervised Fine-Tuning of LLaDA are designed to be straightforward.  Guidance is provided in [GUIDELINES.md](GUIDELINES.md). You can also refer to [SMDM](https://github.com/ML-GSAI/SMDM) for a similar training process.

## Evaluation

LLaDA is evaluated using conditional likelihood estimation and conditional generation. The evaluation code for LLaDA-Base, based on [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), is now available. Refer to [EVAL.md](EVAL.md) for usage instructions and bug details.

## Frequently Asked Questions (FAQ)

Here are answers to common questions:

### 0. How do I train my own LLaDA?

Refer to [GUIDELINES.md](GUIDELINES.md) and [SMDM](https://github.com/ML-GSAI/SMDM) for guidance.

### 1. What is the difference between LLaDA and BERT?

LLaDA is a generative model that uses a masked diffusion model, using a randomly varying masking ratio and being related to BERT and MaskGIT.

### 2. What is the relationship between LLaDA and Transformer?

LLaDA, like GPT, utilizes the Transformer architecture but differs in its probabilistic modeling approach. LLaDA employs a diffusion model for probabilistic modeling, while GPT utilizes an autoregressive next-token prediction method.

### 3. What is the sampling efficiency of LLaDA?

LLaDA's sampling speed is currently slower than autoregressive models.  Efforts are underway to optimize efficiency.

### 4. What is the training stability of LLaDA?

The training process is generally stable.  The paper details the rare instance of a training crash.

### 5. Why is the final answer "72" generated earlier than the intermediate calculation step (e.g., 12 × 4 = 48) in Tab4?

This is a result of the remasking strategy, in which reasoning steps are masked out again during the process.

### 6. Why does LLaDA answer 'Bailing' when asked 'Who are you'?

This is due to the data used for training.

### 7. Our journey in developing LLaDA?

LLaDA is built upon prior works, including [RADD](https://arxiv.org/abs/2406.03736) and [SMDM](https://arxiv.org/abs/2410.18514).

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

Join the conversation and stay updated on LLaDA's progress!
<div style="display: flex; justify-content: center; flex-wrap: wrap;">
    <img src="./imgs/QR.jpg" style="width: 50%" />
</div>
```
Key improvements and optimizations:

*   **SEO Optimization:** The title and headings are optimized with relevant keywords like "Large Language Diffusion Models,"  "LLaDA," and "Diffusion."
*   **Concise Hook:** The one-sentence hook clearly introduces LLaDA's core value proposition.
*   **Clear Structure:** Uses headings, subheadings, and bullet points for readability and easy navigation.
*   **Hugging Face Links:** Includes direct links to the hosted models for ease of access.
*   **Actionable Information:** Provides clear instructions on how to get started, including installation and usage.
*   **Emphasis on Benefits:** Highlights the key benefits and features of LLaDA.
*   **Contextual Links:** The links are in context and lead to the right locations.
*   **FAQ Organization:** The FAQ section is improved to be easier to read and understand.
*   **Call to Action:** The discussion section encourages community engagement.
*   **Concise and focused**: The text is more focused on describing LLaDA's capabilities and uses, reducing unnecessary fluff.