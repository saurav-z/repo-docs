# Enhance Your Image Generation Prompts with PromptEnhancer

**PromptEnhancer** by Tencent Hunyuan uses chain-of-thought rewriting to significantly improve the quality and clarity of text prompts, leading to better image generation results.  Learn more at the original repository: [https://github.com/Hunyuan-PromptEnhancer/PromptEnhancer](https://github.com/Hunyuan-PromptEnhancer/PromptEnhancer).

[<img src="https://img.shields.io/badge/Paper-arXiv:2509.04545-red?logo=arxiv" alt="arXiv">](https://www.arxiv.org/abs/2509.04545)
[<img src="https://img.shields.io/badge/知乎-技术解读-0084ff?logo=zhihu" alt="Zhihu">](https://zhuanlan.zhihu.com/p/1949013083109459515)
[<img src="https://img.shields.io/badge/Model-PromptEnhancer_7B-blue?logo=huggingface" alt="HuggingFace Model">](https://huggingface.co/tencent/HunyuanImage-2.1/tree/main/reprompt)
[<img src="https://img.shields.io/badge/Benchmark-T2I_Keypoints_Eval-blue?logo=huggingface" alt="T2I-Keypoints-Eval Dataset">](https://huggingface.co/datasets/PromptEnhancer/T2I-Keypoints-Eval)
[<img src="https://img.shields.io/badge/Homepage-PromptEnhancer-1abc9c?logo=homeassistant&logoColor=white" alt="Homepage">](https://hunyuan-promptenhancer.github.io/)
[<img src="https://img.shields.io/badge/Code-HunyuanImage2.1-2ecc71?logo=github" alt="HunyuanImage2.1 Code">](https://github.com/Tencent-Hunyuan/HunyuanImage-2.1)
[<img src="https://img.shields.io/badge/Model-HunyuanImage2.1-3498db?logo=huggingface" alt="HunyuanImage2.1 Model">](https://huggingface.co/tencent/HunyuanImage-2.1)
[<img src="https://img.shields.io/badge/Hunyuan-black.svg?logo=x height=22px" alt="Hunyuan Twitter">](https://x.com/TencentHunyuan)

---

<p align="center">
  <img src="assets/teaser-1.png" alt="PromptEnhancer Teaser"/>
</p>

## Key Features

*   **Enhanced Prompt Structure:** Rewrites prompts to be clearer, layered, and logically consistent for superior image generation.
*   **Intent Preservation:**  Maintains the core meaning across key elements like subject, action, style, and layout.
*   **"Global-to-Details-to-Summary" Approach:** Organizes prompts to describe primary elements first, then secondary details, and finally a concise style summary.
*   **Robust Output Parsing:** Includes graceful fallback mechanisms to handle variations in model output.
*   **Configurable Parameters:** Allows users to fine-tune generation with parameters like temperature and top\_p for diverse results.

## What's New

*   **[2025-09-18]** ✨ Try the [PromptEnhancer-32B](https://huggingface.co/PromptEnhancer/PromptEnhancer-32B) for higher-quality prompt enhancement!
*   **[2025-09-16]** Released the [T2I-Keypoints-Eval dataset](https://huggingface.co/datasets/PromptEnhancer/T2I-Keypoints-Eval).
*   **[2025-09-07]** Released the [PromptEnhancer-7B model](https://huggingface.co/tencent/HunyuanImage-2.1/tree/main/reprompt).
*   **[2025-09-07]** Released the [technical report](https://arxiv.org/abs/2509.04545).

## Installation

Install the required packages using pip:

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

*   `models_root_path`:  Local path or repo ID for the model.  Supports `trust_remote_code` models.
*   `device_map`: Device mapping (defaults to `auto`).
*   `predict(...)`:
    *   `prompt_cot` (str):  The input prompt to be rewritten.
    *   `sys_prompt` (str):  Optional system prompt. A default is provided for image prompt rewriting.
    *   `temperature` (float):  Enables sampling when `>0`; uses deterministic generation when `0`.
    *   `top_p` (float):  Nucleus sampling threshold (used when sampling).
    *   `max_new_tokens` (int):  The maximum number of new tokens to generate.

## Citation

If you use PromptEnhancer in your work, please cite the following:

```bibtex
@article{promptenhancer,
  title={PromptEnhancer: A Simple Approach to Enhance Text-to-Image Models via Chain-of-Thought Prompt Rewriting},
  author={Wang, Linqing and Xing, Ximing and Cheng, Yiji and Zhao, Zhiyuan and Tao, Jiale and Wang, QiXun and Li, Ruihuang and Chen, Comi and Li, Xin and Wu, Mingrui and Deng, Xinchi and Wang, Chunyu and Lu, Qinglin},
  journal={arXiv preprint arXiv:2509.04545},
  year={2025}
}
```

## Acknowledgements

We thank the following open-source projects and communities: [Transformers](https://huggingface.co/transformers) and [HuggingFace](https://huggingface.co).

## Contact

For inquiries, contact the R&D and product teams or the open-source team at [hunyuan_opensource@tencent.com](mailto:hunyuan_opensource@tencent.com).

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Hunyuan-PromptEnhancer/PromptEnhancer&type=Date)](https://www.star-history.com/#Hunyuan-PromptEnhancer/PromptEnhancer&Date)