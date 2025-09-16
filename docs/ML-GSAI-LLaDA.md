# LLaDA: Pioneering Large Language Diffusion Models (LLMs)

**LLaDA revolutionizes language modeling with a novel masked diffusion approach, achieving performance that rivals leading autoregressive models. Learn more about LLaDA at the [original repository](https://github.com/ML-GSAI/LLaDA).**

[![arXiv](https://img.shields.io/badge/Paper-arXiv-red.svg)](https://arxiv.org/abs/2502.09992)
[![Hugging Face - Base](https://img.shields.io/badge/Hugging%20Face-LLaDA_Base-FFEB3B)](https://huggingface.co/GSAI-ML/LLaDA-8B-Base)
[![Hugging Face - Instruct](https://img.shields.io/badge/Hugging%20Face-LLaDA_Instruct-FFEB3B)](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct)
[![Hugging Face - Demo](https://img.shields.io/badge/Hugging%20Face-Demo-blue)](https://huggingface.co/spaces/multimodalart/LLaDA)
[![Zhihu 1](https://img.shields.io/badge/Zhihu1-%E7%9F%A5%E4%B9%8E1-blue)](https://zhuanlan.zhihu.com/p/24214732238)
[![Zhihu 2](https://img.shields.io/badge/Zhihu2-%E7%9F%A5%E4%B9%8E2-blue)](https://www.zhihu.com/question/1908479621466396378/answer/1910672718174589774?share_code=1kreOq5gzOtnM&utm_psn=1910708245535912148&utm_source=wechat_timeline&utm_medium=social&s_r=0)

## Key Features

*   **State-of-the-Art Performance:** LLaDA-8B rivals LLaMA3 8B in performance with a novel masked diffusion approach.
*   **Open-Source Models:** Access LLaDA-8B-Base and LLaDA-8B-Instruct models on Hugging Face.
*   **Innovative Architecture:** Employs a unique masked diffusion approach, offering a fresh perspective on language modeling.
*   **Competitive Vision-Language Model:** LLaDA-V, a diffusion-based vision-language model, outperforming other diffusion MLLMs.
*   **Easy to Use:** Inference scripts and Gradio demo provided for easy integration and experimentation.

## What's New

*   **LLaDA-MoE-7B-A1B-Base & LLaDA-MoE-7B-A1B-Instruct:** The first diffusion language models pretrained from scratch with MoE architecture.
*   **LLaDA 1.5:** Incorporates VRPO to reduce gradient variance and enhance preference alignment.
*   **Evaluation Code:** Evaluation code based on the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) is now available.

## Introduction

LLaDA (Large Language Diffusion with Masking) is a groundbreaking diffusion model with an 8B scale, trained entirely from scratch to deliver competitive performance.

<div style="display: flex; justify-content: center; flex-wrap: wrap;">
    <img src="./imgs/LLaDA_vs_LLaMA.svg" style="width: 45%" />
    <img src="./imgs/LLaDA_vs_LLaMA_chat.svg" style="width: 46%" />
</div>

## Inference

Easily run inference with the pre-trained LLaDA models. First, install `transformers==4.38.2`.

```bash
pip install transformers==4.38.2
```

Then use the `transformers` library to load the model:

```python
from transformers import AutoModel, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Base', trust_remote_code=True)
model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Base', trust_remote_code=True, torch_dtype=torch.bfloat16)
```

*   The `get_log_likelihood()` and `generate()` functions are available for conditional likelihood evaluation and conditional generation.
*   You can directly run `python chat.py` to have multi-round conversations with LLaDA-8B-Instruct.

Please consult the paper and [GUIDELINES.md](GUIDELINES.md) for more details.

## Gradio Demo

Experience LLaDA through an interactive Gradio demo, thanks to [apolinário](https://github.com/apolinario)!

*   Install Gradio: `pip install gradio`
*   Run the demo: `python app.py`

<div style="display: flex; justify-content: center; flex-wrap: wrap;">
    <img src="./imgs/example_gradio.gif" style="width: 80%" />
</div>

## Training and Fine-Tuning

While the training framework and data are not provided, training LLaDA is straightforward.

*   Adapt your existing autoregressive model training code.
*   Refer to [GUIDELINES.md](GUIDELINES.md) for guidance.
*   Explore [SMDM](https://github.com/ML-GSAI/SMDM) as a reference.

## Evaluation

LLaDA uses conditional likelihood estimation and conditional generation for evaluation.

*   See the [paper](https://arxiv.org/abs/2502.09992) for details and the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) library.
*   Evaluation code based on the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) is available for LLaDA-Base.
*   Refer to [EVAL.md](EVAL.md) for evaluation code usage and known bugs.

## FAQ

Address common questions about LLaDA.

### 0. How do I train my own LLaDA?

Refer to [GUIDELINES.md](GUIDELINES.md).  You can also see the [SMDM](https://github.com/ML-GSAI/SMDM) code.

### 1. What is the difference between LLaDA and BERT?

LLaDA is a generative model with a varied masking ratio (0 to 1), providing in-context learning and instruction-following capabilities, and ensuring Fisher consistency.

### 2. What is the relationship between LLaDA and Transformer?

LLaDA uses the Transformer architecture with a diffusion model for probabilistic modeling, unlike the autoregressive next-token prediction of GPT.

### 3. What is the sampling efficiency of LLaDA?

LLaDA's sampling speed is slower than autoregressive models.  The team will continue to optimize efficiency, believing there is significant room for improvement (e.g. Consistency model).

### 4. What is the training stability of LLaDA?

Training stability is discussed in Section 2.2 of the paper.

### 5. Why is the final answer "72" generated earlier than the intermediate calculation step (e.g., 12 × 4 = 48) in Tab4?

The mask predictor correctly predicts the reasoning process. However, the reasoning steps are masked out again during remasking, so the final answer is generated sooner.

### 6. Why does LLaDA answer 'Bailing' when asked 'Who are you'?

This is due to the design of the pre-training and SFT data.

### 7. Our journey in developing LLaDA?

LLaDA builds upon [RADD](https://arxiv.org/abs/2406.03736) and [SMDM](https://arxiv.org/abs/2410.18514) research.

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

Join the discussion and stay updated by scanning the QR code below:

<div style="display: flex; justify-content: center; flex-wrap: wrap;">
    <img src="./imgs/QR.jpg" style="width: 50%" />
</div>