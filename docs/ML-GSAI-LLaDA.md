# LLaDA: Revolutionizing Language Modeling with Diffusion (8B Parameter Model)

**LLaDA (Large Language Diffusion with Masking) is an 8B-parameter diffusion model that rivals LLaMA3 8B in performance, pushing the boundaries of language understanding and generation.**  [Explore the LLaDA Repository](https://github.com/ML-GSAI/LLaDA)

## Key Features & Highlights:

*   **Cutting-Edge Performance:** LLaDA achieves competitive results against state-of-the-art models like LLaMA3 8B, demonstrating the power of diffusion-based language modeling.
*   **Innovative Diffusion Approach:** LLaDA employs a novel masked diffusion model, offering a fresh perspective on language model training and inference.
*   **Open-Source Models:** Access and experiment with both the LLaDA-8B-Base and LLaDA-8B-Instruct models available on Hugging Face.
*   **Comprehensive Evaluation:** Detailed evaluation code based on the `lm-evaluation-harness` for the base model is provided.
*   **Gradio Demo:** Interactive Gradio demo available for easy experimentation and exploration of LLaDA's capabilities.
*   **Model Variants:** Includes MoE architecture variants for improved performance.

## What's New

*   **LLaDA-MoE-7B:** Introducing the LLaDA-MoE-7B family, the first diffusion language models pre-trained with a Mixture-of-Experts (MoE) architecture, offering improved performance with fewer parameters.
*   **LLaDA 1.5:** Incorporates VRPO for enhanced preference alignment.
*   **LLaDA-V:** A competitive diffusion-based vision-language model.

## Getting Started with LLaDA

*   **Inference:** Load and use the LLaDA-8B-Base and LLaDA-8B-Instruct models via Hugging Face using the `transformers` library. Code snippets are provided for easy integration.
*   **Gradio Demo:** Experience LLaDA through an interactive web demo built with Gradio. Simply run `python app.py` after installing Gradio.
*   **Pre-training & SFT:** Detailed guidelines for pre-training and supervised fine-tuning (SFT) are available in [GUIDELINES.md](GUIDELINES.md).

## Evaluation

LLaDA's performance is rigorously evaluated using both conditional likelihood estimation and conditional generation, as detailed in the research paper and  [EVAL.md](EVAL.md).

## Frequently Asked Questions (FAQ)

*   **How do I train my own LLaDA?** Refer to [GUIDELINES.md](GUIDELINES.md) for comprehensive training guidance.
*   **What is the difference between LLaDA and BERT?** LLaDA is a generative model based on masked diffusion, while BERT uses a fixed masking ratio for a different training objective.
*   **What is the relationship between LLaDA and Transformer?**  LLaDA uses the Transformer architecture but employs a diffusion model for probabilistic modeling, unlike the autoregressive approach of GPT.
*   **What is the sampling efficiency of LLaDA?** LLaDA is slower than autoregressive models. The authors are exploring efficiency optimizations.
*   **What is the training stability of LLaDA?**  LLaDA's training is generally stable with measures taken to address potential crashes.
*   **Why is the final answer generated earlier than the intermediate calculation step in Tab4?** The remasking strategy explains this behavior.
*   **Why does LLaDA answer 'Bailing' when asked 'Who are you'?** This is due to the pre-training data focusing on language modeling tasks.
*   **What is the journey in developing LLaDA?** LLaDA is based on the research described in RADD and SMDM.

## Citation

```bibtex
@article{nie2025large,
  title={Large Language Diffusion Models},
  author={Nie, Shen and Zhu, Fengqi and You, Zebin and Zhang, Xiaolu and Ou, Jingyang and Hu, Jun and Zhou, Jun and Lin, Yankai and Wen, Ji-Rong and Li, Chongxuan},
  journal={arXiv preprint arXiv:2502.09992},
  year={2025}
}
```

## Join the Conversation

Stay up-to-date with the latest LLaDA developments and discuss with the community!

<div style="display: flex; justify-content: center; flex-wrap: wrap;">
    <img src="./imgs/QR.JPG" style="width: 50%" />
</div>