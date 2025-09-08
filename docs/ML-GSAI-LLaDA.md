# LLaDA: Revolutionizing Language Modeling with Large Language Diffusion Models

**LLaDA (Large Language Diffusion with Masking) is a groundbreaking 8B-scale diffusion model that challenges the dominance of autoregressive models, offering a new approach to language understanding and generation.**  [Explore the original LLaDA repository](https://github.com/ML-GSAI/LLaDA) for more details.

## Key Features

*   **State-of-the-Art Performance:**  LLaDA rivals the performance of LLaMA3 8B, demonstrating the potential of diffusion models in the language domain.
*   **Diffusion-Based Architecture:** Employs a novel diffusion-based approach to language modeling, offering a fresh perspective compared to traditional autoregressive models.
*   **Openly Available Models:** Pre-trained and fine-tuned models (LLaDA-8B-Base and LLaDA-8B-Instruct) are readily accessible on Hugging Face, enabling easy experimentation and integration.
*   **Comprehensive Evaluation:** Offers thorough evaluation using both conditional likelihood estimation and conditional generation, providing a robust assessment of its capabilities.
*   **Versatile Applications:**  Supports various language tasks, including in-context learning and instruction following.
*   **Community Support:**  Active development and discussion channels, including a WeChat QR code for community interaction, ensuring continuous improvements and user engagement.
*   **Open Source Resources:** The repository contains a demo, a model, and evaluation code.

## Key Updates
*   **LLaDA 1.5:** Incorporates VRPO to improve the performance of LLaDA.
*   **LLaDA-V:** A diffusion-based vision-language model that outperforms other diffusion MLLMs.
*   **Evaluation Code:** Evaluation code based on the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) is now available for LLaDA-Base.

## Quick Start

*   **Inference:** Utilize the provided `transformers==4.38.2` with `AutoModel` and `AutoTokenizer` to load the LLaDA models from Hugging Face. Example code is provided in the original README.
*   **Conversation:** Run `python chat.py` to engage in multi-round conversations with the LLaDA-8B-Instruct model.
*   **Demo:** Experiment with the Gradio demo by running `python app.py` after installing Gradio (`pip install gradio`).

## Training and Evaluation

*   **Training Guidelines:**  Refer to [GUIDELINES.md](GUIDELINES.md) for detailed instructions on pre-training and supervised fine-tuning.  The codebase for [SMDM](https://github.com/ML-GSAI/SMDM) can be used as a reference.
*   **Evaluation:**  Detailed evaluation methods and results can be found in the [paper](https://arxiv.org/abs/2502.09992) and [EVAL.md](EVAL.md).

## Frequently Asked Questions (FAQ)

A curated FAQ addresses common queries about LLaDA, including:

*   Training your own LLaDA model.
*   The differences between LLaDA and BERT.
*   The relationship between LLaDA and Transformers.
*   Sampling efficiency and its optimization.
*   Training stability and failure mitigation.
*   Specific model behaviors.
*   The research journey behind LLaDA.

## Citation

```bibtex
@article{nie2025large,
  title={Large Language Diffusion Models},
  author={Nie, Shen and Zhu, Fengqi and You, Zebin and Zhang, Xiaolu and Ou, Jingyang and Hu, Jun and Zhou, Jun and Lin, Yankai and Wen, Ji-Rong and Li, Chongxuan},
  journal={arXiv preprint arXiv:2502.09992},
  year={2025}
}