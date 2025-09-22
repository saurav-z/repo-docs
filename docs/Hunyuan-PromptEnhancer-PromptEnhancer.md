<!-- SEO-optimized README -->

# PromptEnhancer: Enhance Text-to-Image Models with Advanced Prompt Rewriting

**PromptEnhancer revolutionizes image generation by rewriting prompts, leading to more detailed and accurate results.**  [Explore the original repo](https://github.com/Hunyuan-PromptEnhancer/PromptEnhancer).

---

[![arXiv](https://img.shields.io/badge/Paper-arXiv:2509.04545-red?logo=arxiv)](https://www.arxiv.org/abs/2509.04545)
[![Zhihu](https://img.shields.io/badge/知乎-技术解读-0084ff?logo=zhihu)](https://zhuanlan.zhihu.com/p/1949013083109459515)
[![Hugging Face Model](https://img.shields.io/badge/Model-PromptEnhancer_7B-blue?logo=huggingface)](https://huggingface.co/tencent/HunyuanImage-2.1/tree/main/reprompt)
[![T2I-Keypoints-Eval Dataset](https://img.shields.io/badge/Benchmark-T2I_Keypoints_Eval-blue?logo=huggingface)](https://huggingface.co/datasets/PromptEnhancer/T2I-Keypoints-Eval)
[![Homepage](https://img.shields.io/badge/Homepage-PromptEnhancer-1abc9c?logo=homeassistant&logoColor=white)](https://hunyuan-promptenhancer.github.io/)
[![HunyuanImage2.1 Code](https://img.shields.io/badge/Code-HunyuanImage2.1-2ecc71?logo=github)](https://github.com/Tencent-Hunyuan/HunyuanImage-2.1)
[![HunyuanImage2.1 Model](https://img.shields.io/badge/Model-HunyuanImage2.1-3498db?logo=huggingface)](https://huggingface.co/tencent/HunyuanImage-2.1)
[![Hunyuan](https://img.shields.io/badge/Hunyuan-black.svg?logo=x)](https://x.com/TencentHunyuan)

---

<p align="center">
  <img src="assets/teaser-1.png" alt="PromptEnhancer Teaser"/>
</p>

## Key Features of PromptEnhancer

*   **Enhanced Prompt Quality:** Rewrites prompts to be clearer, more detailed, and logically structured, resulting in improved image generation output.
*   **Intent Preservation:**  Maintains the core meaning of the original prompt across key elements like subject, action, style, and attributes.
*   **"Global to Detail" Structure:** Organizes prompts using a "global-details-summary" narrative, leading to a more coherent prompt structure.
*   **Robust Parsing:** Offers reliable output parsing, handling various scenarios with graceful fallback mechanisms.
*   **Customizable Parameters:** Allows fine-tuning of generation through configurable parameters such as temperature, top_p, and max_new_tokens.

## What's New

*   **[2025-09-18]** ✨ Check out the [PromptEnhancer-32B](https://huggingface.co/PromptEnhancer/PromptEnhancer-32B) model for superior prompt enhancement!
*   **[2025-09-16]** Released the [T2I-Keypoints-Eval dataset](https://huggingface.co/datasets/PromptEnhancer/T2I-Keypoints-Eval).
*   **[2025-09-07]** Released the [PromptEnhancer-7B model](https://huggingface.co/tencent/HunyuanImage-2.1/tree/main/reprompt).
*   **[2025-09-07]** Published the [technical report](https://arxiv.org/abs/2509.04545).

## Installation

```bash
pip install -r requirements.txt
```

## Model Download

```bash
# for PromptEnhancer-7B model
huggingface-cli download tencent/HunyuanImage-2.1/reprompt --local-dir ./models/promptenhancer-7b
```

## Quickstart Guide

### Using HunyuanPromptEnhancer

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

## Parameters Explained

*   `models_root_path`: Specifies the local path or repository ID for the model. Supports `trust_remote_code` models.
*   `device_map`:  Defines device mapping, defaulting to "auto".
*   `predict(...)`:
    *   `prompt_cot` (str): The original prompt you want to rewrite.
    *   `sys_prompt` (str): An optional system prompt. A default is provided for image prompt rewriting.
    *   `temperature` (float):  Set `>0` for sampling; `0` for deterministic generation.
    *   `top_p` (float):  Nucleus sampling threshold (relevant when sampling).
    *   `max_new_tokens` (int): Limits the number of new tokens to generate.

## Citation

If you use PromptEnhancer in your research, please cite the following:

```bibtex
@article{promptenhancer,
  title={PromptEnhancer: A Simple Approach to Enhance Text-to-Image Models via Chain-of-Thought Prompt Rewriting},
  author={Wang, Linqing and Xing, Ximing and Cheng, Yiji and Zhao, Zhiyuan and Tao, Jiale and Wang, QiXun and Li, Ruihuang and Chen, Comi and Li, Xin and Wu, Mingrui and Deng, Xinchi and Wang, Chunyu and Lu, Qinglin},
  journal={arXiv preprint arXiv:2509.04545},
  year={2025}
}
```

## Acknowledgements

We are grateful to the following open-source projects and communities: [Transformers](https://huggingface.co/transformers) and [HuggingFace](https://huggingface.co).

## Contact

For inquiries or feedback, please contact our open-source team or email us at [hunyuan_opensource@tencent.com](mailto:hunyuan_opensource@tencent.com).

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Hunyuan-PromptEnhancer/PromptEnhancer&type=Date)](https://www.star-history.com/#Hunyuan-PromptEnhancer/PromptEnhancer&Date)