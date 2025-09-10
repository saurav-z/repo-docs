# LLaDA: Revolutionizing Language Modeling with Diffusion [LLaDA on GitHub](https://github.com/ML-GSAI/LLaDA)

LLaDA (Large Language Diffusion with Masking) is a groundbreaking diffusion model that rivals the performance of the LLaMA3 8B model, offering a new approach to language understanding and generation.

**Key Features:**

*   **State-of-the-Art Performance:** LLaDA achieves competitive results compared to models like LLaMA3 8B.
*   **Diffusion-Based Approach:** Explores a novel generative approach using masked diffusion models.
*   **Open-Source Models:**  Freely available base and instruction-tuned models (LLaDA-8B-Base and LLaDA-8B-Instruct) on Hugging Face.
*   **Comprehensive Evaluation:** Provides evaluation code based on the lm-evaluation-harness library.
*   **Easy Integration:** Straightforward inference and training, with clear guidelines and examples.
*   **Active Development:**  Ongoing updates and improvements, including LLaDA 1.5 and LLaDA-V.

## News & Updates

*   **[2025.05.25]** Introduced [LLaDA 1.5](https://ml-gsai.github.io/LLaDA-1.5-Demo/) with VRPO for enhanced preference alignment.
*   **[2025.05.23]** Released [LLaDA-V](https://ml-gsai.github.io/LLaDA-V-demo/), a competitive diffusion-based vision-language model.
*   **[2025.05.04]**  Provided evaluation code based on the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) for LLaDA-Base.
*   **[2025.02.14]** Published the paper on [arXiv](https://arxiv.org/abs/2502.09992) and open-sourced LLaDA-8B-Base and LLaDA-8B-Instruct.

## Introduction

LLaDA introduces a fresh perspective on language modeling using a masked diffusion approach.  Trained from scratch with an 8B parameter scale, LLaDA rivals the performance of LLaMA3 8B.

<div style="display: flex; justify-content: center; flex-wrap: wrap;">
    <img src="./imgs/LLaDA_vs_LLaMA.svg" style="width: 45%" />
    <img src="./imgs/LLaDA_vs_LLaMA_chat.svg" style="width: 46%" />
</div>

## Inference

Deploy LLaDA-8B-Base and LLaDA-8B-Instruct for your projects:

*   **Installation:** Install `transformers==4.38.2`.
*   **Model Loading:** Use `transformers` library.
    ```python
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Base', trust_remote_code=True)
    model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Base', trust_remote_code=True, torch_dtype=torch.bfloat16)
    ```
*   **Utilities:** Utilize `get_log_likelihood.py` and `generate.py` for evaluation and generation.
*   **Chat Demo:** Run `python chat.py` for interactive conversations with LLaDA-8B-Instruct.
*   **Further Information:** Consult the [GUIDELINES.md](GUIDELINES.md) and the paper.

## Gradio Demo

Experience LLaDA through an interactive Gradio demo:

*   **Installation:** Install Gradio: `pip install gradio`
*   **Run:** Execute `python app.py`.

<div style="display: flex; justify-content: center; flex-wrap: wrap;">
    <img src="./imgs/example_gradio.gif" style="width: 80%" />
</div>

## Training Guide

While the training framework and data aren't provided, training LLaDA is straightforward:

*   **Adapt Existing Code:** Modify your autoregressive model training code with a few lines.
*   **Reference:** Refer to [GUIDELINES.md](GUIDELINES.md) for pre-training and SFT guidance.
*   **Example:** Explore [SMDM](https://github.com/ML-GSAI/SMDM) for a similar training process.

## Evaluation

LLaDA's performance is assessed using:

*   **Conditional Likelihood Estimation:** Applied to specific metrics for the base model using lm-evaluation-harness.
*   **Conditional Generation:** Used for the Instruct model and the remaining metrics.
*   **Details:** Refer to Appendix B.5 of the [paper](https://arxiv.org/abs/2502.09992) for comprehensive details.
*   **Evaluation Code:** Code provided for LLaDA-Base.  Seeking assistance with Instruct model bug fixes (details in [EVAL.md](EVAL.md)).

## FAQ

Addressing common inquiries regarding LLaDA:

*   **How to Train:** Refer to [GUIDELINES.md](GUIDELINES.md) and [SMDM](https://github.com/ML-GSAI/SMDM).
*   **LLaDA vs. BERT:** LLaDA explores masked diffusion models with a generative approach. It uses a random masking ratio (0-1), differing from BERT's fixed ratio.
*   **LLaDA vs. Transformer:** LLaDA utilizes the Transformer architecture with a diffusion model for probabilistic modeling.
*   **Sampling Efficiency:** LLaDA's sampling is currently slower than autoregressive models, but there is significant optimization potential.
*   **Training Stability:** During pre-training, LLaDA experienced one training crash at 1.2T tokens, which was resolved by resuming checkpoint and reducing learning rate.
*   **Reasoning Process:** Mask predictor predictions during remasking may lead to the final answer appearing earlier than intermediate steps.
*   **Identity Question:** LLaDA may respond with "Bailing" due to the design of its pre-training and SFT data.
*   **LLaDA's Development Journey:** Built upon [RADD](https://arxiv.org/abs/2406.03736) and [SMDM](https://arxiv.org/abs/2410.18514).

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

Connect with the LLaDA community:  [QR Code for WeChat Discussion](https://github.com/ML-GSAI/LLaDA/blob/main/imgs/QR.jpg)