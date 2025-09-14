# LLaDA: Revolutionizing Language Modeling with Diffusion (Explore the Future!)

**LLaDA (Large Language Diffusion with Masking) is a groundbreaking 8B parameter diffusion model that challenges traditional autoregressive models, offering a new approach to language understanding and generation.** Explore the cutting edge of language models, where diffusion meets the power of the Transformer architecture.  [Explore the original LLaDA repo](https://github.com/ML-GSAI/LLaDA).

**Key Features:**

*   **State-of-the-Art Performance:** LLaDA competes with, and in some cases, surpasses, models like LLaMA3 8B, showcasing the power of diffusion models.
*   **Innovative Architecture:** LLaDA utilizes a masked diffusion approach, offering a fresh perspective on language modeling.
*   **Open Source:** Access and experiment with LLaDA-8B-Base and LLaDA-8B-Instruct models on Hugging Face.
*   **Inference Made Easy:**  Simple installation and inference instructions, along with readily available code for conditional likelihood evaluation and generation.
*   **Interactive Demo:**  Try out the LLaDA demo using Gradio to experience its capabilities firsthand.
*   **Extensive Evaluation:**  Detailed evaluation methodologies, including conditional likelihood estimation and generation, provide comprehensive insights.
*   **Active Development:** LLaDA is continuously evolving, with new models and improvements being released regularly. (LLaDA-MoE-7B-A1B-Base and LLaDA-MoE-7B-A1B-Instruct)
*   **Training Insights:** Access to training guidelines and related resources for understanding and potentially adapting LLaDA for your own projects.
*   **Detailed FAQ:** Address common questions and provide in-depth explanations about LLaDA's architecture, training, and performance.

## Key Highlights:

*   **LLaDA-MoE-7B-A1B-Instruct**  uses ~1B active parameters at inference while surpassing LLaDA 1.5(an 8B dense model), and comparable to Qwen2.5-3B-Instruct.
*   **LLaDA 1.5**  incorporates VRPO to reduce gradient variance and enhance preference alignment in LLaDA.
*   **LLaDA-V** is a competitive diffusion-based vision-language model.

## Quickstart

### Inference

1.  **Install Dependencies:**
    ```bash
    pip install transformers==4.38.2 torch
    ```
2.  **Load the Model:**
    ```python
    from transformers import AutoModel, AutoTokenizer
    import torch # Add this line

    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Base', trust_remote_code=True)
    model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Base', trust_remote_code=True, torch_dtype=torch.bfloat16)
    ```
3.  **Explore Code:** Use `get_log_likelihood.py` and `generate.py` for conditional evaluation and generation.
4.  **Chat:** Run `python chat.py` to converse with the LLaDA-8B-Instruct model.

### Demo

1.  **Install Gradio:** `pip install gradio`
2.  **Run Demo:** `python app.py`

## Learn More

*   **Paper:**  [arXiv](https://arxiv.org/abs/2502.09992)
*   **Models:**
    *   [LLaDA-8B-Base](https://huggingface.co/GSAI-ML/LLaDA-8B-Base)
    *   [LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct)
    *   [LLaDA-MoE-7B-A1B-Base](https://huggingface.co/inclusionAI/LLaDA-MoE-7B-A1B-Base)
    *   [LLaDA-MoE-7B-A1B-Instruct](https://huggingface.co/inclusionAI/LLaDA-MoE-7B-A1B-Instruct)
*   **Demo:** [Hugging Face Demo](https://huggingface.co/spaces/multimodalart/LLaDA)
*   **Evaluation:** [EVAL.md](EVAL.md)
*   **Training:** [GUIDELINES.md](GUIDELINES.md)
*   **Related Work:** [SMDM](https://github.com/ML-GSAI/SMDM)

## Citation

```bibtex
@article{nie2025large,
  title={Large Language Diffusion Models},
  author={Nie, Shen and Zhu, Fengqi and You, Zebin and Zhang, Xiaolu and Ou, Jingyang and Hu, Jun and Zhou, Jun and Lin, Yankai and Wen, Ji-Rong and Li, Chongxuan},
  journal={arXiv preprint arXiv:2502.09992},
  year={2025}
}
```

## Join the Community

Stay up-to-date with the latest LLaDA developments and engage in discussions by scanning the WeChat QR code provided in the original repo.