# LLaDA: Revolutionizing Language Modeling with Diffusion (LLaDA)

**LLaDA (Large Language Diffusion with Masking) is an 8B parameter diffusion model that challenges the capabilities of traditional autoregressive models, offering comparable performance to LLaMA3-8B.** Explore the cutting edge of language modeling and discover the potential of masked diffusion models.  [Explore the LLaDA Repository](https://github.com/ML-GSAI/LLaDA)

**Key Features:**

*   **State-of-the-Art Performance:** LLaDA demonstrates competitive results, rivalling models like LLaMA3-8B.
*   **Diffusion-Based Approach:** Explores a novel approach to language modeling utilizing masked diffusion.
*   **Open-Source Models:** Access and experiment with pre-trained models including LLaDA-8B-Base and LLaDA-8B-Instruct, available on Hugging Face.
*   **Comprehensive Evaluation:** Provides evaluation code based on lm-evaluation-harness.
*   **Easy Integration:** Compatible with existing Transformers library.
*   **Ongoing Development:**  Stay tuned for new models such as LLaDA 1.5 and LLaDA-V.

## What's New

*   **LLaDA 1.5:**  Released to incorporate VRPO for improved gradient variance and preference alignment.
*   **LLaDA-V:** Introduced as a competitive diffusion-based vision-language model.
*   **Evaluation Code:** Provided evaluation code utilizing the lm-evaluation-harness.
*   **Open-Sourced Models:** Released LLaDA-8B-Base and LLaDA-8B-Instruct.

## Quick Start

### Installation

Install necessary libraries:

```bash
pip install transformers==4.38.2 gradio
```

### Inference

Load the model and tokenizer:

```python
from transformers import AutoModel, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Base', trust_remote_code=True)
model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Base', trust_remote_code=True, torch_dtype=torch.bfloat16)
```

### Interactive Demo

Run the Gradio demo:

```bash
python app.py
```

## Key Concepts & FAQs

### **1. LLaDA vs. BERT:**

LLaDA's goal is to explore masked diffusion models as a language modeling approach, with a focus on in-context learning, instruction-following, and Fisher consistency. The masking ratio varies randomly between 0 and 1, unlike BERT's fixed ratio, which makes it a generative model.

### **2. LLaDA and Transformers:**

LLaDA uses the Transformer architecture, but employs a diffusion model for probabilistic modeling, unlike GPT's autoregressive next-token prediction.

### **3. Sampling Efficiency:**

LLaDA's sampling speed is currently slower than autoregressive models due to fixed context length, lack of KV-Cache, and the need for sampling steps equal to the response length. The team is actively working on optimizations, including semi-autoregressive sampling and consistency distillation.

### **4. Training Stability:**

LLaDA's pre-training process experienced a single crash at 1.2T tokens, which was resolved by resuming the checkpoint and reducing the learning rate.

### **5. Why is the final answer "72" generated earlier than the intermediate calculation step?**

This is due to the mask predictor successfully predicting the reasoning process but, during the remasking process, the reasoning steps get masked out again.

### **6. Why does LLaDA answer 'Bailing' when asked 'Who are you'?**

This is because the pre-training and SFT data were designed for training an autoregressive model, and LLaDA directly utilizes data that contains identity markers.

### **7. What is the journey in developing LLaDA?**

LLaDA builds upon prior works, including RADD and SMDM, exploring the theoretical underpinnings of masked diffusion models and developing scaling laws.

## Further Exploration

*   **[Paper](https://arxiv.org/abs/2502.09992)**:  Find detailed information about LLaDA.
*   **[GUIDELINES.md](GUIDELINES.md)**:  Refer to the guidelines for pre-training and SFT.
*   **[EVAL.md](EVAL.md)**: Learn how to use the evaluation code.
*   **[SMDM](https://github.com/ML-GSAI/SMDM)**:  Explore a similar training process.
*   **Hugging Face:** Access the models [LLaDA-8B-Base](https://huggingface.co/GSAI-ML/LLaDA-8B-Base) and [LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct).
*   **Demo:** Test the demo at [Hugging Face](https://huggingface.co/spaces/multimodalart/LLaDA).

## Citation

```bibtex
@article{nie2025large,
  title={Large Language Diffusion Models},
  author={Nie, Shen and Zhu, Fengqi and You, Zebin and Zhang, Xiaolu and Ou, Jingyang and Hu, Jun and Zhou, Jun and Lin, Yankai and Wen, Ji-Rong and Li, Chongxuan},
  journal={arXiv preprint arXiv:2502.09992},
  year={2025}
}
```

## Community

Stay up-to-date on the latest progress and participate in discussions by scanning the WeChat QR code.
<div style="display: flex; justify-content: center; flex-wrap: wrap;">
    <img src="./imgs/QR.jpg" style="width: 50%" />
</div>