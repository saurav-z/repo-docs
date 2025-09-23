# LLaDA: Revolutionizing Language Modeling with Diffusion (and Outperforming LLAMA3 8B!)

**Discover LLaDA, a groundbreaking Large Language Diffusion Model (LLM) that leverages the power of diffusion to achieve state-of-the-art performance, rivaling even the powerful LLaMA3 8B model.** Explore the official [LLaDA GitHub Repository](https://github.com/ML-GSAI/LLaDA) for more details and resources.

## Key Features of LLaDA

*   **Diffusion-Based Architecture:** Employs a novel diffusion-based approach to language modeling, offering a fresh perspective on generating and understanding text.
*   **8B Parameter Scale:**  A large-scale model trained from scratch, demonstrating the potential of diffusion models at scale.
*   **Competitive Performance:**  Achieves performance comparable to or exceeding models like LLaMA3 8B.
*   **Openly Available Models:** Easily accessible LLaDA-8B-Base and LLaDA-8B-Instruct models are available on Hugging Face for inference.
*   **Multi-faceted Capabilities:** Supports both conditional likelihood estimation and conditional generation for comprehensive evaluation.
*   **Active Development & Advancements:** Constantly evolving with new releases like LLaDA-MoE-7B-A1B and LLaDA-V, demonstrating continued innovation.
*   **Gradio Demo:**  Interactive demo available for testing and experiencing LLaDA's capabilities (see examples below).

## What's New

*   **[2025.09.11]** Introducing [LLaDA-MoE-7B-A1B-Base](https://huggingface.co/inclusionAI/LLaDA-MoE-7B-A1B-Base) and [LLaDA-MoE-7B-A1B-Instruct](https://huggingface.co/inclusionAI/LLaDA-MoE-7B-A1B-Instruct), the first diffusion language model pretrained from scratch with MoE architecture.
*   **[2025.05.25]** Introducing [LLaDA 1.5](https://ml-gsai.github.io/LLaDA-1.5-Demo/), which incorporates VRPO to reduce gradient variance and enhance preference alignment in LLaDA.
*   **[2025.05.23]** Introducing [LLaDA-V](https://ml-gsai.github.io/LLaDA-V-demo/), a competitive diffusion-based vision-language model, outperforming other diffusion MLLMs.
*   **[2025.05.04]** Evaluation code based on the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) for the LLaDA-Base.
*   **[2025.02.14]** Paper released on [arXiv](https://arxiv.org/abs/2502.09992) and open-sourced [LLaDA-8B-Base](https://huggingface.co/GSAI-ML/LLaDA-8B-Base) and [LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct).

## Inference

**Get started with LLaDA!**  Load and use the [LLaDA-8B-Base](https://huggingface.co/GSAI-ML/LLaDA-8B-Base) and [LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) models easily with Transformers:

```python
from transformers import AutoModel, AutoTokenizer
import torch  # Import torch

tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Base', trust_remote_code=True)
model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Base', trust_remote_code=True, torch_dtype=torch.bfloat16)
```

You can use functions like `get_log_likelihood()` and `generate()` from provided scripts to evaluate and generate text. For instructions, see the original [GUIDELINES.md](GUIDELINES.md) file.

## Gradio Demo

Experience LLaDA interactively through the Gradio demo, which allows you to have multi-round conversations with LLaDA-8B-Instruct.

<div style="display: flex; justify-content: center; flex-wrap: wrap;">
    <img src="./imgs/example_gradio.gif" style="width: 80%" />
</div>

## Training and Evaluation

*   **Training:** Refer to [GUIDELINES.md](GUIDELINES.md) and the [SMDM](https://github.com/ML-GSAI/SMDM) repository for guidance on training your own LLaDA models.
*   **Evaluation:**  LLaDA is evaluated using conditional likelihood estimation and conditional generation. The evaluation code for the LLaDA-Base model, based on the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), is provided. Refer to [EVAL.md](EVAL.md) for details and bug reports.

## Frequently Asked Questions (FAQ)

*   **Training Your Own LLaDA:** See [GUIDELINES.md](GUIDELINES.md) and [SMDM](https://github.com/ML-GSAI/SMDM) for training instructions.
*   **LLaDA vs. BERT:** LLaDA is a generative diffusion model with a unique masking strategy that allows for in-context learning and Fisher consistency, differentiating it from BERT's approach.
*   **LLaDA and Transformers:** LLaDA uses the Transformer architecture but employs a diffusion model for probabilistic modeling, unlike the autoregressive approach of GPT.
*   **Sampling Efficiency:** The sampling speed is currently slower than autoregressive models, with optimization on the horizon.
*   **Training Stability:** During pre-training, training stability was maintained, with a minor adjustment at the 1.2T token mark.
*   **Reasoning Process:** LLaDA's reasoning process, while successful, can sometimes lead to the final answer appearing before intermediate steps due to remasking during diffusion.
*   **"Who are you" Response:** The response "Bailing" is due to the training data.
*   **LLaDA's Evolution:** LLaDA builds on the foundations laid by RADD and SMDM, offering theoretical and practical insights into masked diffusion models.

## Citation

```bibtex
@article{nie2025large,
  title={Large Language Diffusion Models},
  author={Nie, Shen and Zhu, Fengqi and You, Zebin and Zhang, Xiaolu and Ou, Jingyang and Hu, Jun and Zhou, Jun and Lin, Yankai and Wen, Ji-Rong and Li, Chongxuan},
  journal={arXiv preprint arXiv:2502.09992},
  year={2025}
}
```

## Discussion and Updates

Stay informed and join the conversation! Scan the QR code below to connect with the LLaDA community on WeChat for discussions and the latest updates.
<div style="display: flex; justify-content: center; flex-wrap: wrap;">
    <img src="./imgs/QR.JPG" style="width: 50%" />
</div>