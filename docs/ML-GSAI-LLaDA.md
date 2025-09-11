# LLaDA: Large Language Diffusion Models - Revolutionizing Language Modeling

**LLaDA introduces a novel approach to language modeling by using diffusion models, achieving performance that rivals state-of-the-art LLMs like LLaMA3-8B.**  For more details, check out the original repository [here](https://github.com/ML-GSAI/LLaDA).

## Key Features:

*   **Diffusion-Based Language Modeling:** LLaDA utilizes a masked diffusion process for language modeling, offering a fresh perspective on generating and understanding text.
*   **Competitive Performance:**  LLaDA matches the performance of the LLaMA3-8B model, demonstrating the potential of diffusion models in the LLM space.
*   **Open-Source Models:** Access and experiment with LLaDA-8B-Base and LLaDA-8B-Instruct models available on Hugging Face.
*   **Inference and Demo:** Easily load the models with `transformers` and try out the Gradio demo to experience LLaDA's capabilities.
*   **In-Context Learning and Instruction Following:** LLaDA's architecture allows for effective in-context learning and following instructions.
*   **Evaluation Code and Comprehensive Paper:** Evaluate the models with provided evaluation code and explore the methods in the arXiv paper.
*   **Continuous Updates:** Stay updated with new versions and demo releases like LLaDA 1.5 and LLaDA-V.

## What's New:

*   **[2025.05.25]** Introducing [LLaDA 1.5](https://ml-gsai.github.io/LLaDA-1.5-Demo/), which incorporates VRPO.
*   **[2025.05.23]** Introducing [LLaDA-V](https://ml-gsai.github.io/LLaDA-V-demo/), a diffusion-based vision-language model.
*   **[2025.05.04]** Evaluation code based on the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) for LLaDA-Base.
*   **[2025.02.14]** Paper published on [arXiv](https://arxiv.org/abs/2502.09992) and open-sourced [LLaDA-8B-Base](https://huggingface.co/GSAI-ML/LLaDA-8B-Base) and [LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct).

## Inference
The [LLaDA-8B-Base](https://huggingface.co/GSAI-ML/LLaDA-8B-Base) and [LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) are uploaded
in Huggingface. Please first install `transformers==4.38.2` and employ the [transformers](https://huggingface.co/docs/transformers/index) to load.

```angular2html
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Base', trust_remote_code=True)
model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Base', trust_remote_code=True, torch_dtype=torch.bfloat16)
```

## Try the Demo
Interact with LLaDA-8B-Instruct by running `python chat.py`. Experience LLaDA's capabilities firsthand with our interactive [Gradio](https://www.gradio.app) demo (run `python app.py` after installing Gradio with `pip install gradio`).

## Training and Evaluation

While we do not provide the training framework, we offer guidelines and insights for training and evaluating LLaDA models through [GUIDELINES.md](GUIDELINES.md) and [EVAL.md](EVAL.md).

## Frequently Asked Questions (FAQ)

Explore answers to common questions about LLaDA:

1.  **How do I train my own LLaDA?** See [GUIDELINES.md](GUIDELINES.md) and [SMDM](https://github.com/ML-GSAI/SMDM).
2.  **What is the difference between LLaDA and BERT?** LLaDA is a generative model using a diffusion process, unlike BERT.
3.  **What is the relationship between LLaDA and Transformer?** LLaDA uses the Transformer architecture but employs diffusion for probabilistic modeling.
4.  **What is the sampling efficiency of LLaDA?** Sampling speed is currently slower, but optimization is ongoing.
5.  **What is the training stability of LLaDA?** The training stability is discussed in Section 2.2 of the paper.
6.  **Why is the final answer generated before the intermediate calculation step in Tab4?** Reasoning steps are remasked during the remasking process.
7.  **Why does LLaDA answer 'Bailing' when asked 'Who are you'?** This is due to data design and the use of identity markers.
8.  **Our journey in developing LLaDA?** LLaDA is built upon our two prior works, [RADD](https://arxiv.org/abs/2406.03736) and [SMDM](https://arxiv.org/abs/2410.18514). 

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

Stay updated on the latest progress by scanning the WeChat QR code.