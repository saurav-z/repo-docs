# LLaDA: Revolutionizing Language Modeling with Diffusion (Large Language Diffusion Models)

**LLaDA (Large Language Diffusion with Masking) is a groundbreaking diffusion model, demonstrating state-of-the-art performance at an 8B parameter scale, challenging conventional autoregressive language models.** Explore the original repository [here](https://github.com/ML-GSAI/LLaDA).

[![arXiv](https://img.shields.io/badge/Paper-arXiv-red.svg)](https://arxiv.org/abs/2502.09992)
[![Hugging Face - Base](https://img.shields.io/badge/Hugging%20Face-LLaDA_Base-FFEB3B)](https://huggingface.co/GSAI-ML/LLaDA-8B-Base)
[![Hugging Face - Instruct](https://img.shields.io/badge/Hugging%20Face-LLaDA_Instruct-FFEB3B)](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct)
[![Hugging Face - Demo](https://img.shields.io/badge/Hugging%20Face-Demo-blue)](https://huggingface.co/spaces/multimodalart/LLaDA)
[![Zhihu1](https://img.shields.io/badge/Zhihu1-知乎1-blue)](https://zhuanlan.zhihu.com/p/24214732238)
[![Zhihu2](https://img.shields.io/badge/Zhihu2-知乎2-blue)](https://www.zhihu.com/question/1908479621466396378/answer/1910672718174589774?share_code=1kreOq5gzOtnM&utm_psn=1910708245535912148&utm_source=wechat_timeline&utm_medium=social&s_r=0)

## Key Features of LLaDA:

*   **Diffusion-Based Language Modeling:** Utilizes a novel diffusion approach for language modeling.
*   **8B Parameter Scale:** Demonstrates impressive performance at a manageable model size.
*   **Competitive Performance:** Rivals the performance of LLaMA3 8B, a leading autoregressive model.
*   **Inference Ready:** Provides pre-trained models (Base & Instruct) on Hugging Face for easy use.
*   **Gradio Demo:** Includes an interactive Gradio demo for exploring LLaDA's capabilities.
*   **Open-Source & Accessible:** Offers evaluation code and guidelines for training and understanding.

## News

*   **[2025.05.25]** Introduced [LLaDA 1.5](https://ml-gsai.github.io/LLaDA-1.5-Demo/), incorporating VRPO to reduce gradient variance and enhance preference alignment.
*   **[2025.05.23]** Released [LLaDA-V](https://ml-gsai.github.io/LLaDA-V-demo/), a competitive diffusion-based vision-language model.
*   **[2025.05.04]** Provided evaluation code based on [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) for LLaDA-Base.
*   **[2025.02.14]** Published the research paper on [arXiv](https://arxiv.org/abs/2502.09992) and open-sourced [LLaDA-8B-Base](https://huggingface.co/GSAI-ML/LLaDA-8B-Base) and [LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct).

## Introduction

LLaDA (<b>L</b>arge <b>La</b>nguage <b>D</b>iffusion with m<b>A</b>sking) is a diffusion model with an unprecedented 8B scale, trained entirely from scratch, rivaling LLaMA3 8B in performance.

<div style="display: flex; justify-content: center; flex-wrap: wrap;">
    <img src="./imgs/LLaDA_vs_LLaMA.svg" style="width: 45%" />
    <img src="./imgs/LLaDA_vs_LLaMA_chat.svg" style="width: 46%" />
</div>

## Inference

The [LLaDA-8B-Base](https://huggingface.co/GSAI-ML/LLaDA-8B-Base) and [LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) models are available on Hugging Face.

1.  **Install Dependencies:** `transformers==4.38.2`
2.  **Load Model:**

    ```python
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Base', trust_remote_code=True)
    model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Base', trust_remote_code=True, torch_dtype=torch.bfloat16)
    ```
3.  **Use provided utility scripts:** `get_log_likelihood.py` and `generate.py`
4.  **Run chat:** `python chat.py`

Refer to the paper and [GUIDELINES.md](GUIDELINES.md) for details.

## Gradio Demo

Try out LLaDA with an interactive Gradio demo.

1.  **Install Gradio:** `pip install gradio`
2.  **Run Demo:** `python app.py`

<div style="display: flex; justify-content: center; flex-wrap: wrap;">
    <img src="./imgs/example_gradio.gif" style="width: 80%" />
</div>

## Pre-training and Supervised Fine-Tuning

Training framework and data are not provided.
Modify existing auto-regressive model code with a few lines of code.
Refer to [GUIDELINES.md](GUIDELINES.md) and [SMDM](https://github.com/ML-GSAI/SMDM)

## Evaluation

LLaDA uses conditional likelihood estimation and conditional generation for evaluation.
Implemented with [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) and custom methods.

*   **Evaluation Code:** Evaluation code is provided for LLaDA-Base (lm-evaluation-harness).
*   **Instruct Model:** Issue with Instruct model evaluation is being debugged.
*   **Details:** Refer to [EVAL.md](EVAL.md) for code usage and bug details.
*   **Paper:** See Appendix B.5. of the paper ([arxiv.org/abs/2502.09992](https://arxiv.org/abs/2502.09992)) for details.

## FAQ

### 0. How do I train my own LLaDA?

Refer to [GUIDELINES.md](GUIDELINES.md).  Also, see [SMDM](https://github.com/ML-GSAI/SMDM).

### 1. What is the difference between LLaDA and BERT?

LLaDA explores masked diffusion models. Its loss function is related to BERT and MaskGIT.
LLaDA employs a varying masking ratio, making it a generative model.
LLaDA can naturally perform in-context learning, instruction-following, and ensures Fisher consistency
for scalability with large datasets and models.

### 2. What is the relationship between LLaDA and Transformer?

LLaDA, like GPT, uses the Transformer architecture.
The key difference is the probabilistic modeling approach: GPT utilizes autoregressive next-token prediction, while LLaDA uses a diffusion model.

### 3. What is the sampling efficiency of LLaDA?

LLaDA's sampling is currently slower.
Reasons: fixed context length, lack of KV-Cache, and optimal performance at response length sampling steps.
Future Optimization: Semi-autoregressive sampling, consistency distillation, and cache methods (Fast-dllm, dllm-cache).

### 4. What is the training stability of LLaDA?

Pre-training: Encountered one crash at 1.2T tokens.  Solution: Resume checkpoint and reduce learning rate.

### 5. Why is the final answer "72" generated earlier than the intermediate calculation step (e.g., 12 × 4 = 48) in Tab4?

The mask predictor predicts the reasoning process, but remasking occurs during generation.

<div style="display: flex; justify-content: center; flex-wrap: wrap;">
    <img src="./imgs/diff_remask.gif" style="width: 80%" />
</div>

### 6. Why does LLaDA answer 'Bailing' when asked 'Who are you'?

Trained with data that contains identity markers.

### 7. Our journey in developing LLaDA?

LLaDA builds upon previous works: [RADD](https://arxiv.org/abs/2406.03736) and [SMDM](https://arxiv.org/abs/2410.18514).

*   **RADD:** Proved LLaDA's training objective, showed that time t is not required in Transformer, and proved that the training objective of masked diffusion models is equivalent to any-order autoregressive models.
*   **SMDM:** Introduced the scaling law for masked diffusion models, and demonstrated masked diffusion models can achieve downstream benchmark results. SMDM introduced a classifier-free guidance method.

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

Join the discussion and stay updated.
<div style="display: flex; justify-content: center; flex-wrap: wrap;">
    <img src="./imgs/QR.jpg" style="width: 50%" />
</div>