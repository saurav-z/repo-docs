# LLaDA: Revolutionizing Language Modeling with Diffusion 
**Explore LLaDA, an 8B-scale Large Language Diffusion Model (LLDM) that challenges the boundaries of text generation by achieving performance comparable to LLaMA3 8B, with a novel diffusion approach.**  [Learn more at the original repo](https://github.com/ML-GSAI/LLaDA).

## Key Features

*   **State-of-the-Art Performance:** LLaDA achieves performance comparable to LLaMA3 8B, demonstrating the potential of diffusion models in language generation.
*   **Innovative Diffusion Approach:** Utilizes a diffusion model with masking for probabilistic modeling, offering a unique perspective on language model architecture.
*   **Openly Available Models:** Access both LLaDA-8B-Base and LLaDA-8B-Instruct models on Hugging Face for easy integration and experimentation.
*   **Comprehensive Evaluation:** Rigorous evaluation based on conditional likelihood estimation and conditional generation. Evaluation code is available for LLaDA-Base.
*   **Clear Guidelines:** Provides guidelines and resources to train your own LLaDA-style models, fostering community contribution.
*   **Active Development:** Ongoing research with recent releases, including LLaDA 1.5, and LLaDA-V, pushing the boundaries of the technology.

## What's New

*   **LLaDA 1.5:** Enhances preference alignment using VRPO to reduce gradient variance.
*   **LLaDA-V:** A competitive diffusion-based vision-language model.
*   **Evaluation Code:** Evaluation code based on the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) is available for LLaDA-Base.

## Quickstart

*   **Inference:** Easily load and run LLaDA models using the Transformers library. Example code is provided in the README.
*   **Interactive Demo:** Explore LLaDA's capabilities via a Gradio demo (installation instructions included).
*   **Training Resources:** Access guidelines and examples for pre-training and Supervised Fine-Tuning (SFT)

## FAQ

*   **How can I train my own LLaDA?** Refer to the [GUIDELINES.md](GUIDELINES.md) for instructions.
*   **What is the key difference between LLaDA and BERT?** LLaDA's training objective is an upper bound on the negative log-likelihood, making it a generative model.
*   **What is the relationship between LLaDA and Transformer?** LLaDA adopts the Transformer architecture but utilizes a diffusion model for probabilistic modeling.
*   **What is the sampling efficiency of LLaDA?** LLaDA's sampling speed is slower than autoregressive models but has significant room for optimization.
*   **What is the training stability of LLaDA?** Training stability is generally good, with one instance of a NaN loss encountered during training.
*   **Why is the final answer generated earlier than the intermediate calculation step?** The model's mask predictor can successfully predict reasoning steps, however these steps may be re-masked.

## Citation

If you use LLaDA in your research, please cite our paper:

```bibtex
@article{nie2025large,
  title={Large Language Diffusion Models},
  author={Nie, Shen and Zhu, Fengqi and You, Zebin and Zhang, Xiaolu and Ou, Jingyang and Hu, Jun and Zhou, Jun and Lin, Yankai and Wen, Ji-Rong and Li, Chongxuan},
  journal={arXiv preprint arXiv:2502.09992},
  year={2025}
}
```

## Join the Community

Stay updated on the latest progress by scanning the WeChat QR code in the original README.