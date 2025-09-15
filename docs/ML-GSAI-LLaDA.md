# LLaDA: Pioneering Large Language Diffusion Models

**LLaDA pushes the boundaries of language modeling with an innovative diffusion-based approach, offering competitive performance to established autoregressive models.**.  [Explore the LLaDA Repository](https://github.com/ML-GSAI/LLaDA)

## Key Features

*   **Groundbreaking Architecture:** LLaDA is a diffusion model, trained from scratch on an 8B parameter scale, showcasing a novel approach to language modeling.
*   **Competitive Performance:**  Achieves performance comparable to LLaMA3 8B, demonstrating the potential of diffusion models in the LLM landscape.
*   **Openly Available Models:**  Includes [LLaDA-8B-Base](https://huggingface.co/GSAI-ML/LLaDA-8B-Base) and [LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) readily available on Hugging Face.
*   **Versatile Applications:**  Demonstrates capabilities in various tasks with pre-trained models and easy to adapt for fine-tuning.
*   **Gradio Demo:** Interactive demonstration available for LLaDA-Instruct through a user-friendly Gradio interface.
*   **Modular Evaluation:**  Evaluation code provided based on the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) for conditional likelihood estimation and conditional generation.
*   **Active Development:** The LLaDA team is actively developing the model, including recent updates with LLaDA-MoE-7B-A1B-Base and LLaDA-V (vision-language).

## Quick Start

### Inference
1.  **Installation:**  Install the necessary transformers library using `pip install transformers==4.38.2`.
2.  **Load Models:** Access and use the models from Hugging Face.

    ```python
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Base', trust_remote_code=True)
    model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Base', trust_remote_code=True, torch_dtype=torch.bfloat16)
    ```
3.  **Run Scripts:** Use the provided scripts (`get_log_likelihood.py`, `generate.py`, and `chat.py`) to evaluate and interact with the model.

### Gradio Demo
1.  **Installation:** Install Gradio with `pip install gradio`.
2.  **Run Demo:** Execute `python app.py` to launch the interactive demonstration.

## Key Updates

*   **LLaDA-MoE-7B-A1B-Base & Instruct:** The first diffusion language model pre-trained from scratch with MoE architecture.
*   **LLaDA 1.5:**  Incorporates VRPO for improved gradient variance reduction and preference alignment.
*   **LLaDA-V:** A diffusion-based vision-language model.
*   **Evaluation Code:**  Evaluation code available using the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

## Training & Evaluation

*   **Training:** Follow the guidelines provided in [GUIDELINES.md](GUIDELINES.md) for pre-training and supervised fine-tuning, or refer to the code in [SMDM](https://github.com/ML-GSAI/SMDM) for a similar approach.
*   **Evaluation:** Evaluate the model using conditional likelihood estimation and conditional generation methods as outlined in the paper and [EVAL.md](EVAL.md).

## Frequently Asked Questions (FAQ)

*   **How do I train my own LLaDA?**  See [GUIDELINES.md](GUIDELINES.md) and the [SMDM](https://github.com/ML-GSAI/SMDM) repository.
*   **What is the difference between LLaDA and BERT?** LLaDA is a masked diffusion model and a generative model, using a randomly varying masking ratio, making it suitable for in-context learning and instruction following.
*   **What is the relationship between LLaDA and the Transformer?** LLaDA utilizes the Transformer architecture while employing a diffusion model for probabilistic modeling.
*   **What is the sampling efficiency of LLaDA?** Sampling is currently slower than autoregressive models, but optimization efforts are ongoing.
*   **What is the training stability of LLaDA?** The model experienced a training crash only once during its 2.3T token training and was resolved by reducing the learning rate.
*   **Why is the final answer "72" generated earlier than the intermediate calculation step?**  The reasoning process is masked during remasking.
*   **Why does LLaDA answer 'Bailing' when asked 'Who are you'?** LLaDA uses data designed for autoregressive model training and includes identity markers directly.
*   **What is the journey in developing LLaDA?**  LLaDA is built upon prior work in diffusion models and explores masked diffusion models' training objectives and scaling laws.

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

Stay updated on the project's progress and engage in discussions through the provided WeChat QR code.

---

**Original Repository:** [https://github.com/ML-GSAI/LLaDA](https://github.com/ML-GSAI/LLaDA)