# LLaDA: Large Language Diffusion Models - Revolutionizing Language Generation

**LLaDA introduces a novel approach to language modeling, leveraging diffusion models to achieve state-of-the-art performance and challenging the dominance of autoregressive models.** Explore LLaDA's capabilities at [https://github.com/ML-GSAI/LLaDA](https://github.com/ML-GSAI/LLaDA).

## Key Features

*   **Diffusion-Based Language Modeling:** LLaDA utilizes a diffusion model architecture, offering a fresh perspective on language generation compared to traditional autoregressive models.
*   **Competitive Performance:**  LLaDA demonstrates performance that rivals that of LLaMA3 8B models.
*   **Openly Available Models:**  Access pre-trained models, including [LLaDA-8B-Base](https://huggingface.co/GSAI-ML/LLaDA-8B-Base) and [LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct), on Hugging Face.
*   **Evaluation Framework:** Evaluation code based on the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) is provided.
*   **Inference and Demo:**  Easy-to-use inference scripts and a Gradio demo ([Hugging Face Demo](https://huggingface.co/spaces/multimodalart/LLaDA)) are available for interaction and experimentation.
*   **Guidance for Training:** While pre-training data and the training framework are not provided, the repository includes guidance for pre-training and supervised fine-tuning (SFT) processes, including details in `GUIDELINES.md`.
*   **Detailed Documentation:** The repository contains FAQs, guidelines, and a paper ([arXiv](https://arxiv.org/abs/2502.09992)) for the model and its development.
*   **Recent Updates:**
    *   **LLaDA 1.5:** Incorporates VRPO to reduce gradient variance.
    *   **LLaDA-V:**  A competitive diffusion-based vision-language model.

## Getting Started

### Installation

To use LLaDA models, install the necessary libraries:

```bash
pip install transformers==4.38.2 gradio
```

### Inference

Load and use the LLaDA models using the provided code snippets.  Example code is provided in the original README.

## Resources

*   **Hugging Face Models:**
    *   [LLaDA-8B-Base](https://huggingface.co/GSAI-ML/LLaDA-8B-Base)
    *   [LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct)
*   **Gradio Demo:** [Hugging Face Demo](https://huggingface.co/spaces/multimodalart/LLaDA)
*   **Paper:** [arXiv:2502.09992](https://arxiv.org/abs/2502.09992)
*   **Evaluation Code:** Evaluation code based on the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) is provided.
*   **Guidelines:**  [GUIDELINES.md](GUIDELINES.md)

## Frequently Asked Questions (FAQ)

A comprehensive FAQ section is included in the original README, which offers insight into LLaDA's architecture, training, and performance.  Key topics covered include:
*   Training LLaDA
*   Differences between LLaDA and BERT
*   LLaDA's relationship with the Transformer architecture
*   Sampling efficiency
*   Training stability
*   Reasoning process
*   Answering "Who are you?"
*   Development Journey
  
## Discussion

Engage with the LLaDA community and stay updated by scanning the WeChat QR code provided in the original README.

## Citation

```bibtex
@article{nie2025large,
  title={Large Language Diffusion Models},
  author={Nie, Shen and Zhu, Fengqi and You, Zebin and Zhang, Xiaolu and Ou, Jingyang and Hu, Jun and Zhou, Jun and Lin, Yankai and Wen, Ji-Rong and Li, Chongxuan},
  journal={arXiv preprint arXiv:2502.09992},
  year={2025}
}