# LLaDA: Large Language Diffusion Models - Unleashing the Power of Diffusion for Language Understanding

**LLaDA revolutionizes language modeling with an 8B-scale diffusion model, offering performance that rivals state-of-the-art autoregressive models.**  Learn more and access the models at the [original LLaDA repository](https://github.com/ML-GSAI/LLaDA).

## Key Features

*   **High Performance:** LLaDA achieves competitive results compared to other leading 8B parameter models like LLaMA3.
*   **Diffusion-Based Architecture:** Utilizes a masked diffusion model, a novel approach to language modeling.
*   **Open-Source:**  LLaDA-8B-Base and LLaDA-8B-Instruct are available on Hugging Face.
*   **Inference Ready:**  Easy-to-use inference code and a Gradio demo are available for exploration.
*   **Evaluation Code:**  Evaluation code is provided for LLaDA-Base, based on lm-evaluation-harness.
*   **Ongoing Development:**  Stay up-to-date with the latest innovations including LLaDA 1.5 and LLaDA-V.

## What's New
*   **LLaDA 1.5:** Introduced VRPO to reduce gradient variance and enhance preference alignment. (2025.05.25)
*   **LLaDA-V:** Introduced a competitive diffusion-based vision-language model. (2025.05.23)
*   **Evaluation Code:** Evaluation code based on the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) for LLaDA-Base is now available. (2025.05.04)
*   **Paper and Models Released:**  Paper available on arXiv and models released on Hugging Face. (2025.02.14)

## Quick Start - Inference

1.  **Install Dependencies:**

    ```bash
    pip install transformers==4.38.2 torch
    ```

2.  **Load the Model:**

    ```python
    from transformers import AutoModel, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Base', trust_remote_code=True)
    model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Base', trust_remote_code=True, torch_dtype=torch.bfloat16)
    ```
3.  **Use Provided Scripts:** Utilize `get_log_likelihood.py` and `generate.py` for conditional likelihood evaluation and conditional generation.

4.  **Run the Chat:** Interact with LLaDA-8B-Instruct using `python chat.py`.

5.  **Explore the Demo:** Run `python app.py` after installing Gradio (`pip install gradio`) for a user-friendly interface.

## Training, Guidelines & Evaluation

*   **Training:** Guidance for pre-training and Supervised Fine-Tuning is available in [GUIDELINES.md](GUIDELINES.md).  Refer to [SMDM](https://github.com/ML-GSAI/SMDM) for code examples.
*   **Evaluation:**  Evaluation code and details on conditional likelihood estimation and conditional generation can be found in [EVAL.md](EVAL.md) and the [paper](https://arxiv.org/abs/2502.09992).

## Frequently Asked Questions (FAQ)

Find answers to common questions including:

*   Training your own LLaDA.
*   Differences between LLaDA and BERT.
*   The relationship between LLaDA and the Transformer architecture.
*   Sampling efficiency.
*   Training stability.
*   Reasoning process and example generations.
*   The answer 'Bailing' when asked "Who are you?".
*   The research journey behind LLaDA.

## Citation

```bibtex
@article{nie2025large,
  title={Large Language Diffusion Models},
  author={Nie, Shen and Zhu, Fengqi and You, Zebin and Zhang, Xiaolu and Ou, Jingyang and Hu, Jun and Zhou, Jun and Lin, Yankai and Wen, Ji-Rong and Li, Chongxuan},
  journal={arXiv preprint arXiv:2502.09992},
  year={2025}
}
```

## Join the Discussion

Stay updated and join the conversation by scanning the WeChat QR code in the original repository.