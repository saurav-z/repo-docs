# LLaDA: Large Language Diffusion Models

**LLaDA revolutionizes language modeling with a novel diffusion-based approach, offering performance that rivals LLaMA3 8B.** Explore the power of LLaDA and unlock new possibilities in natural language processing. For more details, see the original repository: [https://github.com/ML-GSAI/LLaDA](https://github.com/ML-GSAI/LLaDA).

## Key Features:

*   **State-of-the-Art Performance:** LLaDA (8B scale) achieves performance comparable to LLaMA3 8B.
*   **Diffusion-Based Architecture:** Explores a new approach to language modeling, offering unique advantages.
*   **Open-Source Models:** Access LLaDA-8B-Base and LLaDA-8B-Instruct models on Hugging Face.
*   **Inference Code & Demos:** Includes example inference scripts and a Gradio demo for easy experimentation.
*   **Evaluation Code:** Evaluation code based on the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) is available for the LLaDA-Base model.
*   **Comprehensive Documentation:** Guidelines, FAQ, and insights into the LLaDA model.

## What's New:

*   **LLaDA 1.5:** Incorporates VRPO for improved gradient variance and preference alignment.
*   **LLaDA-V:** A competitive diffusion-based vision-language model that outperforms other diffusion MLLMs.
*   **Evaluation Code:** Evaluation code based on the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) for the LLaDA-Base model.

## Quick Start

### 1. Installation:

```bash
pip install transformers==4.38.2 gradio
```

### 2. Inference:

```python
from transformers import AutoModel, AutoTokenizer
import torch # Import torch

tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Base', trust_remote_code=True)
model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Base', trust_remote_code=True, torch_dtype=torch.bfloat16)

#Example
input_text = "Translate this to French: Hello, how are you?"
input_ids = tokenizer(input_text, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(**input_ids, max_length=50)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### 3. Run Demo:

```bash
python app.py
```

## Further Exploration:

*   **Paper:** [arXiv:2502.09992](https://arxiv.org/abs/2502.09992)
*   **Hugging Face:** [LLaDA-8B-Base](https://huggingface.co/GSAI-ML/LLaDA-8B-Base) and [LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct)
*   **Gradio Demo:** [LLaDA Demo](https://huggingface.co/spaces/multimodalart/LLaDA)
*   **Guidelines:** [GUIDELINES.md](GUIDELINES.md)
*   **Evaluation Details:** [EVAL.md](EVAL.md)

## Citation

```bibtex
@article{nie2025large,
  title={Large Language Diffusion Models},
  author={Nie, Shen and Zhu, Fengqi and You, Zebin and Zhang, Xiaolu and Ou, Jingyang and Hu, Jun and Zhou, Jun and Lin, Yankai and Wen, Ji-Rong and Li, Chongxuan},
  journal={arXiv preprint arXiv:2502.09992},
  year={2025}
}