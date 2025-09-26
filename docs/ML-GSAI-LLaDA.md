# LLaDA: Revolutionizing Language Modeling with Diffusion [Link to Original Repo](https://github.com/ML-GSAI/LLaDA)

LLaDA introduces a novel approach to language modeling, utilizing a diffusion-based architecture to achieve impressive performance and explore the potential of masked diffusion models.

**Key Features:**

*   **Cutting-Edge Architecture:** LLaDA leverages a diffusion model with an 8B scale, trained from scratch, pushing the boundaries of language model capabilities.
*   **Competitive Performance:** LLaDA rivals the performance of LLaMA3 8B, demonstrating the effectiveness of its diffusion-based approach.
*   **Open-Source Models:** Access and experiment with LLaDA-8B-Base and LLaDA-8B-Instruct models available on Hugging Face.
*   **Modular Design:** Easily adapt existing autoregressive model training code for LLaDA, promoting research and innovation.
*   **Comprehensive Evaluation:** Evaluate LLaDA using conditional likelihood estimation and conditional generation.
*   **Active Development:** Stay up-to-date with the latest advancements, including LLaDA-MoE models and vision-language integration.

## Models and Deployment

*   [LLaDA-8B-Base](https://huggingface.co/GSAI-ML/LLaDA-8B-Base)
*   [LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct)
*   [LLaDA-MoE-7B-A1B-Base](https://huggingface.co/inclusionAI/LLaDA-MoE-7B-A1B-Base)
*   [LLaDA-MoE-7B-A1B-Instruct](https://huggingface.co/inclusionAI/LLaDA-MoE-7B-A1B-Instruct)
*   [LLaDA-V](https://ml-gsai.github.io/LLaDA-V-demo/)
*   [LLaDA Demo](https://huggingface.co/spaces/multimodalart/LLaDA)

## Getting Started

### Inference

1.  **Install Dependencies:** `pip install transformers==4.38.2`
2.  **Load Models:** Utilize the `transformers` library to load and use the models.

    ```python
    from transformers import AutoModel, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Base', trust_remote_code=True)
    model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Base', trust_remote_code=True, torch_dtype=torch.bfloat16)
    ```

3.  **Utilize Inference Scripts:** Leverage `get_log_likelihood.py` and `generate.py` for conditional likelihood evaluation and generation.

### Gradio Demo

1.  **Install Gradio:** `pip install gradio`
2.  **Run Demo:** Execute `python app.py` to interact with LLaDA through the Gradio interface.

## Training, Evaluation, and Guidelines

*   **Training:** See `GUIDELINES.md` for guidance on pre-training and SFT.
*   **Evaluation:** Detailed methods in the paper and `EVAL.md`.

## Frequently Asked Questions (FAQ)

**0. How do I train my own LLaDA?**
Refer to [GUIDELINES.md](GUIDELINES.md) and consider [SMDM](https://github.com/ML-GSAI/SMDM) for training process information.

**1. What is the difference between LLaDA and BERT?**
LLaDA employs a masked diffusion model with a varying masking ratio, making it a generative model that enables in-context learning and instruction-following.

**2. What is the relationship between LLaDA and Transformer?**
LLaDA adopts the Transformer architecture, with the key difference being the use of a diffusion model for probabilistic modeling, unlike GPT's autoregressive approach.

**3. What is the sampling efficiency of LLaDA?**
LLaDA's sampling speed is currently slower than autoregressive models, but there is significant room for optimization. See paper for details.

**4. What is the training stability of LLaDA?**
Refer to Section 2.2 of the paper for information on the pre-training process.

**5. Why is the final answer "72" generated earlier than the intermediate calculation step (e.g., 12 × 4 = 48) in Tab4?**
The mask predictor remasks reasoning steps.

**6. Why does LLaDA answer 'Bailing' when asked 'Who are you'?**
Due to the pre-training/SFT data, it contains identity markers.

**7. Our journey in developing LLaDA?**
LLaDA builds on RADD and SMDM, with key research in masked diffusion models.

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

Engage in discussions and stay informed about the latest updates by scanning the WeChat QR code (image not included).