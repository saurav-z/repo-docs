# LLaDA: Large Language Diffusion Models 

**LLaDA: Revolutionizing language modeling with a novel diffusion approach that rivals LLaMA3 8B performance, built entirely from scratch!**

[<img src="https://img.shields.io/badge/GitHub-LLaDA-blue?logo=github" alt="GitHub Repo">](https://github.com/ML-GSAI/LLaDA)
[arXiv Paper](https://arxiv.org/abs/2502.09992) | [LLaDA-8B-Base on Hugging Face](https://huggingface.co/GSAI-ML/LLaDA-8B-Base) | [LLaDA-8B-Instruct on Hugging Face](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) | [LLaDA Demo](https://huggingface.co/spaces/multimodalart/LLaDA) | [Zhihu 1](https://zhuanlan.zhihu.com/p/24214732238) | [Zhihu 2](https://www.zhihu.com/question/1908479621466396378/answer/1910672718174589774?share_code=1kreOq5gzOtnM&utm_psn=1910708245535912148&utm_source=wechat_timeline&utm_medium=social&s_r=0)

LLaDA (Large Language Diffusion with Masking) introduces a groundbreaking approach to language modeling, pushing the boundaries of what's possible with diffusion models. Built from the ground up with an 8B parameter scale, LLaDA achieves performance comparable to LLaMA3 8B. Explore the cutting-edge technology behind LLaDA and its potential to transform the field of NLP.

**Key Features:**

*   **Diffusion-Based Language Model:** LLaDA utilizes a novel diffusion model for probabilistic modeling.
*   **8B Parameter Scale:** Trained from scratch, achieving strong performance.
*   **Comparable Performance:** Rivals the performance of LLaMA3 8B.
*   **Open-Source Models:**  Available on Hugging Face for both Base and Instruct models.
*   **Evaluation Code:** Provides evaluation code based on the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).
*   **Gradio Demo:** Interactive demo for easy exploration of LLaDA's capabilities.
*   **Detailed Guidelines:**  Guidance for pre-training and supervised fine-tuning available in [GUIDELINES.md](GUIDELINES.md).

## What's New

*   **[2025.05.25]** LLaDA 1.5, incorporating VRPO for enhanced preference alignment. [LLaDA 1.5 Demo](https://ml-gsai.github.io/LLaDA-1.5-Demo/)
*   **[2025.05.23]** LLaDA-V, a competitive diffusion-based vision-language model. [LLaDA-V Demo](https://ml-gsai.github.io/LLaDA-V-demo/)
*   **[2025.05.04]** Evaluation code based on the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) released for the LLaDA-Base.
*   **[2025.02.14]** Paper released on [arXiv](https://arxiv.org/abs/2502.09992) and open-sourced [LLaDA-8B-Base](https://huggingface.co/GSAI-ML/LLaDA-8B-Base) and [LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct).

## Inference

Easily use LLaDA with the `transformers` library:

```python
from transformers import AutoModel, AutoTokenizer
import torch # Added to ensure it imports
tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Base', trust_remote_code=True)
model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Base', trust_remote_code=True, torch_dtype=torch.bfloat16)
```

Use the provided scripts `get_log_likelihood.py` and `generate.py` for conditional likelihood evaluation and generation. Run `python chat.py` to chat with LLaDA-8B-Instruct.  Refer to the paper and [GUIDELINES.md](GUIDELINES.md) for detailed inference methods.

## Gradio Demo

Interact with LLaDA via a user-friendly Gradio demo:

```bash
pip install gradio
python app.py
```

<div style="display: flex; justify-content: center; flex-wrap: wrap;">
    <img src="./imgs/example_gradio.gif" style="width: 80%" />
</div>

## Training Your Own LLaDA

While the training framework and data are not provided, pre-training and Supervised Fine-Tuning are straightforward. Modify your existing autoregressive model training codebase for LLaDA. Refer to [GUIDELINES.md](GUIDELINES.md) and [SMDM](https://github.com/ML-GSAI/SMDM) for details.

## Evaluation

LLaDA's performance is evaluated using conditional likelihood estimation and conditional generation. The evaluation code is available for the LLaDA-Base. See the paper and [EVAL.md](EVAL.md) for more details, including any current bugs.

## Frequently Asked Questions (FAQ)

**Q: How do I train my own LLaDA?**

A: Refer to [GUIDELINES.md](GUIDELINES.md) and [SMDM](https://github.com/ML-GSAI/SMDM).

**Q: What is the difference between LLaDA and BERT?**

A: LLaDA is a generative model, using a variable masking ratio and an upper bound on the negative log-likelihood of the model distribution, setting it apart from BERT.

**Q: What is the relationship between LLaDA and Transformer?**

A: LLaDA uses the Transformer architecture but employs a diffusion model for probabilistic modeling.

**Q: What is the sampling efficiency of LLaDA?**

A: LLaDA's sampling is currently slower than autoregressive models, but there is significant room for optimization.  See Appendix B.4 and B.6 in the paper.

**Q: What is the training stability of LLaDA?**

A: During pre-training on 2.3T tokens, there was one training crash at 1.2T tokens. The solution was to resume from the checkpoint and reduce the learning rate.

**Q: Why is the final answer generated before intermediate steps?**

A: The mask predictor successfully predicts the reasoning process, but remasking can lead to steps being masked out again.

**Q: Why does LLaDA answer 'Bailing' when asked 'Who are you'?**

A: The pre-training and SFT data contained identity markers, causing this response.

**Q: Our journey in developing LLaDA?**

A: LLaDA builds upon the work of [RADD](https://arxiv.org/abs/2406.03736) and [SMDM](https://arxiv.org/abs/2410.18514).

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

Join the conversation! Scan the WeChat QR code (provided in the original README) to stay updated on LLaDA's progress.