# LLaDA: Revolutionizing Language Modeling with Diffusion (Large Language Diffusion Models)

**LLaDA introduces a novel approach to language modeling, offering an 8B-parameter diffusion model that rivals the performance of LLaMA3 8B, opening new possibilities in natural language processing!** Check out the original repo for more details: [LLaDA GitHub](https://github.com/ML-GSAI/LLaDA)

## Key Features

*   **State-of-the-Art Performance:** LLaDA's 8B model achieves competitive results, matching LLaMA3 8B.
*   **Diffusion-Based Approach:** Utilizing a masked diffusion model for a generative approach to language modeling.
*   **Openly Available Models:** Explore and experiment with pre-trained models like LLaDA-8B-Base and LLaDA-8B-Instruct on Hugging Face.
*   **Flexible Inference:** Leverage provided code and utilize the Hugging Face `transformers` library for easy integration.
*   **Interactive Demo:** Experience LLaDA firsthand with an interactive Gradio demo.
*   **Straightforward Training:** Follow provided guidelines for pre-training and supervised fine-tuning (SFT).
*   **Evaluation Code:** Evaluation code is available based on [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) for the LLaDA-Base.

## Model Details

*   **Introduction:** LLaDA (<b>L</b>arge <b>La</b>nguage <b>D</b>iffusion with m<b>A</b>sking) is a large language diffusion model trained from scratch with an 8B scale.
*   **Model Variants:** Explore base and instruction-tuned models available on Hugging Face, including LLaDA-8B-Base and LLaDA-8B-Instruct.
*   **Architecture:** LLaDA uses the Transformer architecture, combined with a diffusion model for probabilistic modeling.

## Getting Started

*   **Installation:** Install `transformers==4.38.2` to load and use the models.
*   **Inference:** Utilize the provided code examples for conditional likelihood evaluation and generation. Run `python chat.py` to converse with LLaDA-8B-Instruct.
*   **Demos:** Explore the Gradio demo for an interactive experience, and check out the demo hosted on Hugging Face Spaces.

## Training and Evaluation

*   **Training Guidance:** Guidelines for pre-training and SFT are available in [GUIDELINES.md](GUIDELINES.md), and you can refer to [SMDM](https://github.com/ML-GSAI/SMDM) for code.
*   **Evaluation Methods:** Learn about conditional likelihood estimation and conditional generation approaches.
*   **Evaluation Code:**  Evaluation code is available for LLaDA-Base, refer to [EVAL.md](EVAL.md) for details.

## Frequently Asked Questions (FAQ)

*   **Training Your Own LLaDA:** Refer to [GUIDELINES.md](GUIDELINES.md).
*   **LLaDA vs. BERT:** LLaDA is a generative model using a masked diffusion approach, offering advantages in in-context learning and Fisher consistency.
*   **LLaDA and Transformers:** LLaDA uses the Transformer architecture but employs a diffusion model for probabilistic modeling, unlike autoregressive models like GPT.
*   **Sampling Efficiency:** While sampling speed is currently slower than autoregressive models, optimization efforts are underway.
*   **Training Stability:** The training process has been relatively stable.
*   **Reasoning and Masking:** Learn about the remasking strategy of LLaDA in complex scenarios.
*   **Answer "Bailing":** Understanding the potential limitations regarding pre-training and SFT data.
*   **LLaDA's Development:** The project is built upon RADD and SMDM.

## Latest News

*   **[2025.09.11]** Introduced LLaDA-MoE-7B-A1B-Base and LLaDA-MoE-7B-A1B-Instruct, a MoE architecture model.
*   **[2025.05.25]** Introduced LLaDA 1.5.
*   **[2025.05.23]** Introduced LLaDA-V, a competitive diffusion-based vision-language model.
*   **[2025.05.04]** Evaluation code based on the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) provided for the LLaDA-Base.
*   **[2025.02.14]** Paper on [arXiv](https://arxiv.org/abs/2502.09992) and open-sourced LLaDA-8B-Base and LLaDA-8B-Instruct.

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

Join the discussion and stay updated via the WeChat QR code in the original README (included at the end of the original).