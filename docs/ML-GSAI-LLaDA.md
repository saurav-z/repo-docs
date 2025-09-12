# LLaDA: Revolutionizing Language Modeling with Diffusion Models

**LLaDA (Large Language Diffusion with Masking) introduces a novel approach to language modeling, offering performance comparable to LLaMA3 8B.**  Explore the power of diffusion models in language generation! For more details, visit the original repository: [https://github.com/ML-GSAI/LLaDA](https://github.com/ML-GSAI/LLaDA).

**Key Features:**

*   🚀 **High Performance:** Achieves competitive results with LLaMA3 8B, demonstrating the potential of diffusion models in the LLM domain.
*   💡 **Innovative Architecture:** Employs a masked diffusion model trained from scratch, exploring a new frontier in language modeling.
*   🛠️ **Open Source:**  LLaDA-8B-Base and LLaDA-8B-Instruct are available on Hugging Face for easy access and experimentation.
*   🧠 **Instruction Following:** LLaDA-Instruct models are available, demonstrating enhanced capabilities for following instructions.
*   📚 **Comprehensive Evaluation:**  Includes evaluation code based on the `lm-evaluation-harness`, providing a detailed understanding of performance metrics.
*   🖼️ **Multimodal Capabilities:** LLaDA-V, a diffusion-based vision-language model, showcases its potential in multimodal applications.
*   ⚙️ **MoE Architecture:** LLaDA-MoE-7B-A1B models using a Mixture of Experts architecture have been introduced, offering improved performance with fewer active parameters.

**Key Resources:**

*   **Paper:**  [arXiv](https://arxiv.org/abs/2502.09992)
*   **Hugging Face Models:**
    *   [LLaDA-8B-Base](https://huggingface.co/GSAI-ML/LLaDA-8B-Base)
    *   [LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct)
    *   [LLaDA-MoE-7B-A1B-Base](https://huggingface.co/inclusionAI/LLaDA-MoE-7B-A1B-Base)
    *   [LLaDA-MoE-7B-A1B-Instruct](https://huggingface.co/inclusionAI/LLaDA-MoE-7B-A1B-Instruct)
*   **Demo:** [Hugging Face Demo](https://huggingface.co/spaces/multimodalart/LLaDA)
*   **Additional Resources:** Zhihu articles (links provided in the original README).
*   **Evaluation Code:** Provided, based on the `lm-evaluation-harness` for the LLaDA-Base.

## Introduction

LLaDA (<b>L</b>arge <b>La</b>nguage <b>D</b>iffusion with m<b>A</b>sking) is a cutting-edge diffusion model, built from scratch and boasting an impressive 8 billion parameters, competing with LLaMA3 8B in its capabilities.

## Inference

Easily integrate and utilize LLaDA-8B-Base and LLaDA-8B-Instruct models with the provided code and instructions using the Transformers library.

*   **Installation:** Install `transformers==4.38.2`
*   **Loading Models:** Use the provided code snippet:

    ```python
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Base', trust_remote_code=True)
    model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Base', trust_remote_code=True, torch_dtype=torch.bfloat16)
    ```

*   **Evaluation and Generation:** Utilize the provided functions `get_log_likelihood()` and `generate()` in `get_log_likelihood.py` and `generate.py` for evaluations.
*   **Interactive Chat:** Directly run `python chat.py` to engage in multi-round conversations with the LLaDA-8B-Instruct model.
*   **Detailed instructions:** Please refer to the [GUIDELINES.md](GUIDELINES.md) file for further instructions.

## Gradio Demo

Experiment with LLaDA through an interactive Gradio demo, made possible by [apolinário](https://github.com/apolinario).

*   **Installation:** Install Gradio with `pip install gradio`
*   **Run:** Execute `python app.py` to launch the demo and experience LLaDA's capabilities firsthand.

## Training and Fine-Tuning

Though the training framework and data are not provided, LLaDA is straightforward to train and fine-tune.

*   **Guidelines:** For comprehensive instructions, consult the [GUIDELINES.md](GUIDELINES.md) file.
*   **Related Work:** Refer to [SMDM](https://github.com/ML-GSAI/SMDM), which employs a similar training process and has an open-sourced training framework.

## Evaluation

Understand LLaDA's performance through two key evaluation methods:

*   **Conditional Likelihood Estimation:** Used for specific metrics with the base model.
*   **Conditional Generation:** Applied to all metrics for the Instruct model.

**Resources:**

*   **Paper:** [arXiv](https://arxiv.org/abs/2502.09992)
*   **Evaluation Code:** The evaluation code is provided for the LLaDA-Base, leveraging the `lm-evaluation-harness` library, with details found in [EVAL.md](EVAL.md).
*   **Bug Details:** For the Instruct model, refer to the [EVAL.md](EVAL.md) for specific information about ongoing debugging efforts.

## FAQ

This section addresses common questions about LLaDA:

### 0. How do I train my own LLaDA?

*   Refer to [GUIDELINES.md](GUIDELINES.md) for detailed instructions.
*   Also, explore [SMDM](https://github.com/ML-GSAI/SMDM), which uses the same training process as LLaDA and has an open-sourced code.

### 1. What is the difference between LLaDA and BERT?

*   LLaDA is a masked diffusion model aiming for a generative approach.
*   LLaDA employs a randomly varying masking ratio (0-1), unlike BERT's fixed ratio.
*   LLaDA’s objective is an upper bound on the model distribution's negative log-likelihood, enabling in-context learning, instruction-following, and scalability.

### 2. What is the relationship between LLaDA and Transformer?

*   LLaDA utilizes the Transformer architecture, like GPT.
*   The core distinction is in the probabilistic modeling: GPT uses autoregressive next-token prediction, while LLaDA employs a diffusion model.

### 3. What is the sampling efficiency of LLaDA?

*   LLaDA's current sampling speed is slower than autoregressive models due to fixed context length, lack of KV-Cache, and performance depending on the number of sampling steps.
*   Future optimizations are expected.
*   Semi-autoregressive sampling, consistency distillation, and cache methods can be adapted by LLaDA for better results.

### 4. What is the training stability of LLaDA?

*   During pre-training on 2.3T tokens, a single training crash (loss becoming NaN) occurred at 1.2T tokens.
*   The solution: Resume the checkpoint and reduce the learning rate from 4e-4 to 1e-4.

### 5. Why is the final answer "72" generated earlier than the intermediate calculation step (e.g., 12 × 4 = 48) in Tab4?

*   The mask predictor identifies the reasoning process.
*   During remasking, reasoning steps can be masked out again.
*   LLaDA uses a random remasking strategy.

### 6. Why does LLaDA answer 'Bailing' when asked 'Who are you'?

*   The data used was designed for training an autoregressive model.
*   LLaDA directly utilizes data with identity markers.

### 7. Our journey in developing LLaDA?

*   LLaDA builds upon the works [RADD](https://arxiv.org/abs/2406.03736) and [SMDM](https://arxiv.org/abs/2410.18514).
*   RADD proved that masked diffusion models don't need time t as input.
*   SMDM introduced the scaling law for masked diffusion models.

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

Join the conversation and stay updated on the latest progress by scanning the WeChat QR code:
<div style="display: flex; justify-content: center; flex-wrap: wrap;">
    <img src="./imgs/QR.jpg" style="width: 50%" />
</div>