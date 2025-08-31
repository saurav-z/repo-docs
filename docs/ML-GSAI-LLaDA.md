# LLaDA: Revolutionizing Language Modeling with Diffusion Models

**LLaDA (Large Language Diffusion with Masking) is a groundbreaking 8B-parameter diffusion model, trained from scratch, that achieves performance comparable to LLaMA3 8B.** Explore the cutting-edge advancements in language modeling and discover how LLaDA is pushing the boundaries of what's possible. Find the original repo [here](https://github.com/ML-GSAI/LLaDA).

## Key Features

*   **State-of-the-Art Performance:** LLaDA rivals the performance of LLaMA3 8B, demonstrating the potential of diffusion models in the language domain.
*   **Diffusion-Based Architecture:** LLaDA utilizes a novel masked diffusion approach for probabilistic modeling, offering a fresh perspective on language model design.
*   **Open-Source Models:** Access and experiment with pre-trained LLaDA models, including LLaDA-8B-Base and LLaDA-8B-Instruct, available on Hugging Face.
*   **Inference and Generation:** Utilize provided code for conditional likelihood estimation, conditional generation, and interactive chat capabilities.
*   **Comprehensive Documentation:** Access detailed guidelines, evaluation code, and frequently asked questions to facilitate your understanding and experimentation.
*   **Active Development:** The LLaDA project is continuously evolving, with regular updates and new features like LLaDA 1.5 and LLaDA-V.

## Latest Updates

*   **LLaDA 1.5:** Introduction of VRPO to enhance preference alignment.
*   **LLaDA-V:** A competitive diffusion-based vision-language model.
*   **Evaluation Code:** Evaluation code based on lm-evaluation-harness for LLaDA-Base.

## Getting Started

*   **Model Access:** Download and experiment with [LLaDA-8B-Base](https://huggingface.co/GSAI-ML/LLaDA-8B-Base) and [LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) from Hugging Face.
*   **Installation:** Install `transformers==4.38.2`.
*   **Inference:** Utilize provided Python scripts (`get_log_likelihood.py`, `generate.py`, and `chat.py`) for evaluation and interaction.
*   **Gradio Demo:**  Run a user-friendly demo with `python app.py` after installing Gradio (`pip install gradio`).

## Training and Evaluation

*   **Training Guidelines:** Explore [GUIDELINES.md](GUIDELINES.md) and [SMDM](https://github.com/ML-GSAI/SMDM) to understand the pre-training and supervised fine-tuning processes.
*   **Evaluation:** Evaluate LLaDA using conditional likelihood estimation and conditional generation methods.  Refer to [EVAL.md](EVAL.md) for evaluation code and details.

## Frequently Asked Questions (FAQ)

Find answers to common questions regarding:

*   Training your own LLaDA
*   Differences between LLaDA and BERT
*   Relationship between LLaDA and Transformers
*   Sampling efficiency
*   Training stability
*   Reasoning and generation processes
*   And more!

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

Stay up-to-date on the latest progress. Scan the WeChat QR code for discussions and updates!
<div style="display: flex; justify-content: center; flex-wrap: wrap;">
    <img src="./imgs/QR.jpg" style="width: 50%" />
</div>