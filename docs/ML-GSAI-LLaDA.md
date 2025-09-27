# LLaDA: Revolutionizing Language Modeling with Diffusion Models

**LLaDA, a groundbreaking large language diffusion model, challenges the status quo, demonstrating impressive performance and paving the way for a new era in generative AI.  Explore the future of language modeling:  [Original Repo](https://github.com/ML-GSAI/LLaDA)**

**Key Features:**

*   **State-of-the-Art Performance:** LLaDA achieves performance comparable to LLaMA3 8B, showcasing the power of diffusion models in language generation.
*   **8B Parameter Scale:**  Trained from scratch, LLaDA boasts an impressive 8 billion parameter scale, demonstrating significant advancements in model size.
*   **Diverse Model Variations:** Explore LLaDA-Base, LLaDA-Instruct, LLaDA-MoE, and LLaDA-V, offering different architectures and functionalities.
*   **Comprehensive Evaluation:**  Evaluation code based on `lm-evaluation-harness` is provided for the LLaDA-Base, enabling in-depth performance analysis.
*   **Easy Inference & Deployment:** Ready-to-use models are available on Hugging Face, with straightforward inference instructions using `transformers`.
*   **Gradio Demo:**  Interactive Gradio demo lets you experience the power of LLaDA firsthand.
*   **Open-Source & Accessible:**  Includes detailed guidelines for pre-training, supervised fine-tuning, and evaluation.

## Introduction

LLaDA (Large Language Diffusion with Masking) is a diffusion model trained from scratch with 8B parameters.  LLaDA models achieves a high level of performance while utilizing the diffusion model architecture.

<div style="display: flex; justify-content: center; flex-wrap: wrap;">
    <img src="./imgs/LLaDA_vs_LLaMA.svg" style="width: 45%" />
    <img src="./imgs/LLaDA_vs_LLaMA_chat.svg" style="width: 46%" />
</div>

## Models and Availability

*   **LLaDA-8B-Base:** [Hugging Face](https://huggingface.co/GSAI-ML/LLaDA-8B-Base)
*   **LLaDA-8B-Instruct:** [Hugging Face](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct)
*   **LLaDA-MoE-7B-A1B-Base:** [Hugging Face](https://huggingface.co/inclusionAI/LLaDA-MoE-7B-A1B-Base)
*   **LLaDA-MoE-7B-A1B-Instruct:** [Hugging Face](https://huggingface.co/inclusionAI/LLaDA-MoE-7B-A1B-Instruct)
*   **LLaDA-V:** [Demo](https://ml-gsai.github.io/LLaDA-V-demo/)
*   **LLaDA 1.5:** [Demo](https://ml-gsai.github.io/LLaDA-1.5-Demo/)
*   **Demo:** [Hugging Face](https://huggingface.co/spaces/multimodalart/LLaDA)

## Inference

Get started with LLaDA using the following code snippet:

```python
from transformers import AutoModel, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Base', trust_remote_code=True)
model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Base', trust_remote_code=True, torch_dtype=torch.bfloat16)
```

Use `get_log_likelihood.py` and `generate.py` for conditional likelihood evaluation and conditional generation, respectively.  Run `python chat.py` to interact with LLaDA-8B-Instruct.

## Gradio Demo

Interact with LLaDA through the user-friendly Gradio demo:

```bash
pip install gradio
python app.py
```

<div style="display: flex; justify-content: center; flex-wrap: wrap;">
    <img src="./imgs/example_gradio.gif" style="width: 80%" />
</div>

## Training and Evaluation

Detailed guidelines for pre-training and Supervised Fine-Tuning (SFT) can be found in [GUIDELINES.md](GUIDELINES.md).  Evaluation code based on the `lm-evaluation-harness` is provided; refer to [EVAL.md](EVAL.md) for usage and details.

## Frequently Asked Questions (FAQ)

*   **How do I train my own LLaDA?** Refer to [GUIDELINES.md](GUIDELINES.md) and the [SMDM](https://github.com/ML-GSAI/SMDM) for guidance.
*   **What are the key differences between LLaDA and BERT?**
    LLaDA employs a masking ratio that varies randomly between 0 and 1, while BERT uses a fixed ratio, making it a generative model.
*   **What is the relationship between LLaDA and Transformer?**
    LLaDA, like GPT, adopts the Transformer architecture.  The key difference lies in the probabilistic modeling approach: GPT utilizes an autoregressive next-token prediction method, while LLaDA employs a diffusion model for probabilistic modeling.
*   **What is the sampling efficiency of LLaDA?**
    LLaDA's sampling is slower than the autoregressive baseline due to the fixed context length and the lack of techniques like KV-Cache.
*   **What is the training stability of LLaDA?**
    During the total pre-training on 2.3T tokens, we encountered a training crash (loss becoming NaN) only once at 1.2T tokens. Our solution was to resume the checkpoint and reduce the learning rate from 4e-4 to 1e-4.
*   **Why is the final answer "72" generated earlier than the intermediate calculation step (e.g., 12 × 4 = 48) in Tab4?**
    The mask predictor has successfully predicted the reasoning process, but during the remasking process, the reasoning steps are masked out again.
*   **Why does LLaDA answer 'Bailing' when asked 'Who are you'?**
    Because the pre-training and SFT data were designed for training an autoregressive model, whereas LLaDA directly utilizes data that contains identity markers.
*   **What is the journey in developing LLaDA?**
    LLaDA builds on previous works RADD and SMDM. RADD demonstrated the training objective of LLaDA serves as an upper bound on the negative log-likelihood. SMDM introduced the first scaling law for masked diffusion models.

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

Join the conversation and stay updated via the WeChat QR code:

<div style="display: flex; justify-content: center; flex-wrap: wrap;">
    <img src="./imgs/QR.JPG" style="width: 50%" />
</div>
```
Key improvements and SEO optimizations:

*   **Concise Hook:** Starts with a compelling one-sentence introduction that immediately grabs attention and highlights the core value proposition.
*   **Clear Headings:** Uses descriptive and SEO-friendly headings (e.g., "Introduction," "Models and Availability," "Inference") to structure the content logically.
*   **Bulleted Key Features:**  Emphasizes the model's strengths and benefits using bullet points, making it easy for readers to scan and understand.
*   **Keyword Integration:** Naturally incorporates relevant keywords like "Large Language Diffusion Model," "Generative AI," and "Language Modeling."
*   **Hugging Face Links:** Provides direct links to Hugging Face model repositories for easy access and deployment.
*   **Strong Call to Action:** Encourages users to explore the demo and engage with the project.
*   **FAQ Section:** Addresses common questions to provide clarity and improve user experience.
*   **Citation Section:**  Includes a citation for the paper.
*   **Mobile-Friendly Formatting:**  Uses `<div style="display: flex; justify-content: center; flex-wrap: wrap;">` for better responsiveness and display on different devices.
*   **Concise Language:**  Removes unnecessary jargon and focuses on clarity and conciseness.
*   **Summarized Content:** Condenses the original README while retaining essential information.
*   **SEO-Friendly Titles:**  Uses clear and descriptive titles for the sections.
*   **Backlink:**  Includes a prominent link back to the original GitHub repository.