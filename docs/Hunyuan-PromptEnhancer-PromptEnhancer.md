# PromptEnhancer: Enhance Your Text-to-Image Prompts with Chain-of-Thought Rewriting

**Transform your text prompts into detailed and effective instructions for image generation using PromptEnhancer, a simple yet powerful tool.**  Learn more about the project at the original repository: [PromptEnhancer](https://github.com/Hunyuan-PromptEnhancer/PromptEnhancer).

---

<p align="center">
  <img src="assets/teaser-1.png" alt="PromptEnhancer Teaser"/>
</p>

## Key Features

*   **Intent Preservation:** Maintains the core meaning across key elements like subject, action, style, and relationships.
*   **Structured Output:** Employs a "global-details-summary" structure for clear and organized prompts.
*   **Robust Parsing:** Handles output effectively, prioritizing the `<answer>...</answer>` tag for optimal results and gracefully falling back to cleaner text extraction.
*   **Configurable Inference:** Allows control over determinism and diversity through adjustable parameters like temperature and `top_p`.
*   **Supports Chinese and English:** Process prompts in both languages.

## Overview

PromptEnhancer is a utility designed to rewrite text prompts, making them more effective for image generation models. By restructuring input prompts while preserving the original intent, it produces clearer, layered, and logically consistent prompts. This ensures better performance when using text-to-image models.

## Recent Updates

*   **[2025-09-16]** Released the [T2I-Keypoints-Eval dataset](https://huggingface.co/datasets/PromptEnhancer/T2I-Keypoints-Eval).
*   **[2025-09-07]** Released the [PromptEnhancer-7B model](https://huggingface.co/tencent/HunyuanImage-2.1/tree/main/reprompt).
*   **[2025-09-07]** Published the [technical report](https://arxiv.org/abs/2509.04545).

## Installation

Install the required packages using:

```bash
pip install -r requirements.txt
```

## Model Download

Download the PromptEnhancer-7B model:

```bash
huggingface-cli download tencent/HunyuanImage-2.1/reprompt --local-dir ./models/promptenhancer-7b
```

## Quickstart

Here's how to use HunyuanPromptEnhancer:

```python
from inference.prompt_enhancer import HunyuanPromptEnhancer

models_root_path = "./models/promptenhancer-7b"

enhancer = HunyuanPromptEnhancer(models_root_path=models_root_path, device_map="auto")

# Enhance a prompt (Chinese or English)
user_prompt = "Third-person view, a race car speeding on a city track..."
new_prompt = enhancer.predict(
    prompt_cot=user_prompt,
    # Default system prompt is tailored for image prompt rewriting; override if needed
    temperature=0.7,   # >0 enables sampling; 0 uses deterministic generation
    top_p=0.9,
    max_new_tokens=256,
)

print("Enhanced:", new_prompt)
```

## Parameters

*   `models_root_path`:  Specify the local path or repo ID for the model. Supports `trust_remote_code` models.
*   `device_map`:  Define the device mapping (defaults to `auto`).
*   `predict(...)`:
    *   `prompt_cot` (str):  The input prompt that you want to rewrite.
    *   `sys_prompt` (str):  An optional system prompt. A default is provided, optimized for image prompt rewriting.
    *   `temperature` (float): Use a value greater than `0` to enable sampling; use `0` for deterministic generation.
    *   `top_p` (float):  Set the nucleus sampling threshold (effective when sampling).
    *   `max_new_tokens` (int):  Set the maximum number of new tokens to generate.

## Citation

If you use this project in your research, please cite it using:

```bibtex
@article{promptenhancer,
  title={PromptEnhancer: A Simple Approach to Enhance Text-to-Image Models via Chain-of-Thought Prompt Rewriting},
  author={Wang, Linqing and Xing, Ximing and Cheng, Yiji and Zhao, Zhiyuan and Tao, Jiale and Wang, QiXun and Li, Ruihuang and Chen, Comi and Li, Xin and Wu, Mingrui and Deng, Xinchi and Wang, Chunyu and Lu, Qinglin},
  journal={arXiv preprint arXiv:2509.04545},
  year={2025}
}
```

## Acknowledgements

Thanks to the contributions of open-source projects and communities like [Transformers](https://huggingface.co/transformers) and [HuggingFace](https://huggingface.co).

## Contact

For inquiries and feedback, contact the open-source team or send an email to hunyuan_opensource@tencent.com.