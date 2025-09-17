# LLaDA: Revolutionizing Language Modeling with Diffusion (Large Language Diffusion with Masking)

**Explore the future of language AI with LLaDA, an 8B-scale diffusion model that challenges the status quo, achieving performance comparable to LLaMA3.**  [Explore the LLaDA Repository](https://github.com/ML-GSAI/LLaDA)

LLaDA (Large Language Diffusion with Masking) represents a significant leap forward in language modeling, utilizing a novel diffusion-based approach.  This README provides a comprehensive overview of LLaDA, its key features, and how to get started.

**Key Features:**

*   **State-of-the-Art Performance:** LLaDA-8B models rival LLaMA3 8B in performance, demonstrating the power of diffusion models in the language domain.
*   **Diffusion-Based Architecture:** Employs a unique masked diffusion approach, offering a fresh perspective on language modeling compared to traditional autoregressive methods.
*   **Open-Source Availability:**  Pre-trained models, including LLaDA-8B-Base and LLaDA-8B-Instruct, are readily available for inference and experimentation on Hugging Face.
*   **Modular Design:** The architecture is built using the Transformer framework, providing flexibility for customization and integration.
*   **Ongoing Development:**  The project is actively developed with the introduction of new models (e.g., LLaDA-MoE-7B-A1B) and improvements, demonstrating the team's commitment to innovation.
*   **Competitive Evaluation:** Provides evaluation code, including conditional likelihood estimation and conditional generation using `lm-evaluation-harness`.
*   **Clear Guidelines:**  Offers guidelines and resources, including [GUIDELINES.md](GUIDELINES.md), for training and adapting LLaDA.

## Key Highlights

*   **Model Releases:**  Explore models like LLaDA-MoE-7B-A1B, LLaDA 1.5 (with VRPO), and LLaDA-V, pushing boundaries in both language and vision-language tasks.
*   **Interactive Demo:** Interact with the LLaDA-Instruct model via a user-friendly Gradio demo.
*   **Detailed Documentation:** Provides comprehensive information on inference, training, and evaluation to aid in your research and development endeavors.

## Getting Started

### Inference

1.  **Install Dependencies:** Ensure you have the required libraries.  
    ```bash
    pip install transformers==4.38.2
    ```
2.  **Load the Model:** Utilize the `transformers` library to load the LLaDA models from Hugging Face.

    ```python
    from transformers import AutoModel, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Base', trust_remote_code=True)
    model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Base', trust_remote_code=True, torch_dtype=torch.bfloat16)
    ```

3.  **Explore Example Scripts:** Use provided scripts for conditional likelihood evaluation (`get_log_likelihood.py`) and conditional generation (`generate.py`).
4.  **Chat Interface:** Run `python chat.py` to engage in multi-round conversations with the LLaDA-8B-Instruct model.

### Gradio Demo

1.  **Install Gradio:**
    ```bash
    pip install gradio
    ```
2.  **Run the Demo:** Execute `python app.py` to launch the interactive Gradio demo.

## Training & Evaluation

*   **Training:**  While the training framework isn't directly provided, clear guidelines are available in [GUIDELINES.md](GUIDELINES.md) and [SMDM](https://github.com/ML-GSAI/SMDM) provides code samples.
*   **Evaluation:** Use the provided evaluation code based on `lm-evaluation-harness`. Details are in [EVAL.md](EVAL.md) and the paper ([https://arxiv.org/abs/2502.09992](https://arxiv.org/abs/2502.09992)).

## Frequently Asked Questions (FAQ)

Address common questions about LLaDA, including training, model differences (e.g., BERT, Transformer), sampling efficiency, training stability, and understanding of the diffusion-based architecture.

## Citation

```bibtex
@article{nie2025large,
  title={Large Language Diffusion Models},
  author={Nie, Shen and Zhu, Fengqi and You, Zebin and Zhang, Xiaolu and Ou, Jingyang and Hu, Jun and Zhou, Jun and Lin, Yankai and Wen, Ji-Rong and Li, Chongxuan},
  journal={arXiv preprint arXiv:2502.09992},
  year={2025}
}
```

## Discussion & Updates

Stay updated on the latest developments by joining the discussion via the provided WeChat QR code.