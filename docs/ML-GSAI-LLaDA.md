# LLaDA: Revolutionizing Language Modeling with Diffusion Models

**LLaDA introduces a novel approach to large language models (LLMs) by leveraging the power of diffusion models, achieving performance that rivals state-of-the-art autoregressive models like LLaMA3, and is available on [GitHub](https://github.com/ML-GSAI/LLaDA).**

## Key Features

*   **Diffusion-Based Architecture:** LLaDA utilizes a diffusion model, offering a fresh perspective on language modeling.
*   **8B Parameter Scale:** Trained from scratch, LLaDA boasts an impressive 8 billion parameters, demonstrating its potential.
*   **Competitive Performance:** LLaDA matches the performance of LLaMA3 8B, showcasing its capabilities.
*   **Openly Available Models:** Access LLaDA-8B-Base and LLaDA-8B-Instruct models on Hugging Face.
*   **Extensive Resources:** Includes evaluation code based on [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) and detailed guidelines for training and inference.
*   **Active Development:** The project is actively maintained and includes recent advancements such as LLaDA-MoE.

## What's New

*   **[2025.09.11]** Introduced [LLaDA-MoE-7B-A1B-Base](https://huggingface.co/inclusionAI/LLaDA-MoE-7B-A1B-Base) and [LLaDA-MoE-7B-A1B-Instruct](https://huggingface.co/inclusionAI/LLaDA-MoE-7B-A1B-Instruct), the first diffusion language model pretrained from scratch with MoE architecture.
*   **[2025.05.25]** Introduced [LLaDA 1.5](https://ml-gsai.github.io/LLaDA-1.5-Demo/), incorporating VRPO to enhance preference alignment.
*   **[2025.05.23]** Introduced [LLaDA-V](https://ml-gsai.github.io/LLaDA-V-demo/), a competitive diffusion-based vision-language model.
*   **[2025.05.04]** Provided evaluation code based on the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) for the LLaDA-Base.
*   **[2025.02.14]** Published the paper on [arXiv](https://arxiv.org/abs/2502.09992) and open-sourced the LLaDA models.

## Getting Started

### Inference

1.  **Install Dependencies:**
    ```bash
    pip install transformers==4.38.2
    ```

2.  **Load Model and Tokenizer:**
    ```python
    from transformers import AutoModel, AutoTokenizer
    import torch # Add this import

    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Base', trust_remote_code=True)
    model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Base', trust_remote_code=True, torch_dtype=torch.bfloat16)
    ```

3.  **Example Usage:**
    Use `get_log_likelihood.py` and `generate.py` scripts for conditional likelihood evaluation and generation.  Run `python chat.py` for interactive conversations with LLaDA-8B-Instruct.

### Gradio Demo

Run `python app.py` after installing Gradio:

```bash
pip install gradio
```

## Training and Evaluation

*   **Training:** Refer to [GUIDELINES.md](GUIDELINES.md) and [SMDM](https://github.com/ML-GSAI/SMDM) for guidance.  Pre-training and SFT processes are detailed.
*   **Evaluation:** LLaDA is evaluated using conditional likelihood estimation and conditional generation.  Refer to [EVAL.md](EVAL.md) and the paper for detailed methods and results.

## Frequently Asked Questions (FAQ)

A dedicated section provides insights into LLaDA, addressing topics such as model training, comparisons with BERT, the relationship with Transformers, sampling efficiency, training stability, and more.

### 0. How do I train my own LLaDA?
Please refer to [GUIDELINES.md](GUIDELINES.md) for the guidelines. 
You can also refer to [SMDM](https://github.com/ML-GSAI/SMDM), which follows the same training 
process as LLaDA and has open-sourced its code.


### 1. What is the difference between LLaDA and BERT?

Our motivation is not to improve BERT, nor to apply image generation methods like [MaskGIT](https://arxiv.org/abs/2202.04200) 
to text. **Our goal is to explore a theoretically complete language modeling approach — masked diffusion models.** 
During this process, we simplified the approach and discovered that the loss function of masked diffusion models 
is related to the loss functions of BERT and MaskGIT. You can find our theoretical research process in Question 7.

Specifically, LLaDA employs a masking ratio that varies randomly between 0 and 1, while BERT uses 
a fixed ratio. This subtle difference has significant implications. **The training
objective of LLaDA is an upper bound on the negative log-likelihood of the model 
distribution, making LLaDA a generative model.** This enables LLaDA to naturally 
perform in-context learning, instruction-following, and ensures Fisher consistency 
for scalability with large datasets and models. You can also find a direct answer 
to this question in Section 2.1 of our paper.


### 2. What is the relationship between LLaDA and Transformer?
Network structure and probabilistic modeling are two distinct approaches that collectively form the 
foundation of language models. LLaDA, like GPT, adopts the 
Transformer architecture. The key difference lies in the probabilistic modeling approach: GPT 
utilizes an autoregressive next-token prediction method, 
while LLaDA employs a diffusion model for probabilistic modeling.


### 3. What is the sampling efficiency of LLaDA?
Currently, LLaDA's sampling speed is slower than the autoregressive baseline for three reasons: 
1. LLaDA samples with a fixed context length;
2. LLaDA cannot yet leverage techniques like KV-Cache;
3. LLaDA achieves optimal performance when the number of sampling steps equals the response length.
Reducing the number of sampling steps leads to a decrease in performance, as detailed in Appendix B.4 
and Appendix B.6 of our paper.

In this work, we aim to explore the upper limits of LLaDA's capabilities, **challenging the assumption 
that the key LLM abilities are inherently tied to autoregressive models**. We will continue 
to optimize its efficiency in the future. We believe this research approach is reasonable, 
as verifying the upper limits of diffusion language models' capabilities will provide us with
more resources and sufficient motivation to optimize efficiency.

Recall the development of diffusion models for images, from [DDPM](https://arxiv.org/abs/2006.11239) 
to the [Consistency model](https://arxiv.org/pdf/2410.11081), where sampling speed accelerated nearly 
1000 times over the course of 4 years. **We believe there is significant room for optimization in LLaDA's 
sampling efficiency as well**. Current solutions, including semi-autoregressive sampling (as 
detailed in [GUIDELINES.md](GUIDELINES.md)), can mitigate the fixed context length issue, and 
[consistency distillation](https://arxiv.org/pdf/2502.05415) can reduce the number of sampling steps. In
addition, some cache methods (e.g., [Fast-dllm](https://github.com/NVlabs/Fast-dLLM), [dllm-cache](https://github.com/maomaocun/dLLM-cache))
can also be adapted by LLaDA.


### 4. What is the training stability of LLaDA?
For details on the pre-training process of LLaDA, please refer to Section 2.2 of our paper. 
During the total pre-training on 2.3T tokens, we encountered a training crash (loss becoming NaN) 
only once at 1.2T tokens. Our solution was to resume the checkpoint and reduce 
the learning rate from 4e-4 to 1e-4.


### 5. Why is the final answer "72" generated earlier than the intermediate calculation step (e.g., 12 × 4 = 48) in Tab4?

**The mask predictor has successfully predicted the reasoning process. However, during the 
remasking process, the reasoning steps are masked out again.** As shown in the figure 
below, the non-white background represents the model's generation process, while the 
white-background boxes indicate the predictions made by the mask predictor at each step. 
We adopt a randomly remasking strategy.

<div style="display: flex; justify-content: center; flex-wrap: wrap;">
    <img src="./imgs/diff_remask.gif" style="width: 80%" />
</div>

### 6. Why does LLaDA answer 'Bailing' when asked 'Who are you'?
This is because our pre-training and SFT data were designed for training an autoregressive model, 
whereas LLaDA directly utilizes data that contains identity markers.


### 7. Our journey in developing LLaDA?
LLaDA is built upon our two prior works, [RADD](https://arxiv.org/abs/2406.03736) and 
[SMDM](https://arxiv.org/abs/2410.18514). 

RADD demonstrated that the **training objective of LLaDA serves as an upper bound on the negative 
log-likelihood** of the model’s distribution, a conclusion also supported by [MD4](https://arxiv.org/abs/2406.04329) 
and [MDLM](https://arxiv.org/abs/2406.07524). 
Furthermore, RADD was the first to theoretically prove that **masked diffusion models do not require time t 
as an input to Transformer**. This insight provides the theoretical 
justification for LLaDA’s unmodified use of the Transformer architecture. Lastly, 
RADD showed that **the training objective of masked diffusion models is equivalent to that of 
any-order autoregressive models**, offering valuable insights into how masked diffusion models can 
overcome the reversal curse.

SMDM introduces the first **scaling law** for masked diffusion models and demonstrates that, with the 
same model size and training data, masked diffusion models can achieve downstream benchmark results 
on par with those of autoregressive models. Additionally, SMDM presents a simple, **unsupervised 
classifier-free guidance** method that greatly improves downstream benchmark performance, which has 
been adopted by LLaDA.

## Citation

```bibtex
@article{nie2025large,
  title={Large Language Diffusion Models},
  author={Nie, Shen and Zhu, Fengqi and You, Zebin and Zhang, Xiaolu and Ou, Jingyang and Hu, Jun and Zhou, Jun and Lin, Yankai and Wen, Ji-Rong and Li, Chongxuan},
  journal={arXiv preprint arXiv:2502.09992},
  year={2025}
}
```

## Discussion and Support

Join the conversation and stay updated via the WeChat QR code (provided in the original README).