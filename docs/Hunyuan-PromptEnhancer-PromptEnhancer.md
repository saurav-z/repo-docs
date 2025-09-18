# PromptEnhancer: Enhance Text-to-Image Models with Chain-of-Thought Prompt Rewriting

PromptEnhancer is a powerful tool that refines text prompts to improve image generation quality by rewriting them with a chain-of-thought approach. [Explore the original repository](https://github.com/Hunyuan-PromptEnhancer/PromptEnhancer).

**Key Features:**

*   **Intent Preservation:** Maintains the original meaning of prompts while restructuring them for clarity.
*   **Structured Output:** Generates prompts in a "global–details–summary" format, improving logical consistency.
*   **Robust Parsing:** Uses a reliable parsing system with fallback mechanisms to ensure usable output.
*   **Configurable Parameters:** Allows you to control the generation process with parameters like temperature and top\_p.
*   **Open-Source Models:** Offers pre-trained models, including PromptEnhancer-7B and PromptEnhancer-32B.

## Overview

PromptEnhancer is designed to enhance text prompts for text-to-image models. By rewriting input prompts using a chain-of-thought approach, it produces clearer, more structured, and logically consistent prompts, leading to improved image generation results. The system focuses on preserving the intent of key elements such as the subject, action, and style.

## What's New

*   **September 18, 2025:** Released [PromptEnhancer-32B model](https://huggingface.co/PromptEnhancer/PromptEnhancer-32B).
*   **September 16, 2025:** Released [T2I-Keypoints-Eval dataset](https://huggingface.co/datasets/PromptEnhancer/T2I-Keypoints-Eval).
*   **September 07, 2025:** Released [PromptEnhancer-7B model](https://huggingface.co/tencent/HunyuanImage-2.1/tree/main/reprompt).
*   **September 07, 2025:** Released [technical report](https://arxiv.org/abs/2509.04545).

## Installation

To get started, install the necessary dependencies:

```bash
pip install -r requirements.txt
```

## Model Download

Download the desired model:

```bash
# For PromptEnhancer-7B
huggingface-cli download tencent/HunyuanImage-2.1/reprompt --local-dir ./models/promptenhancer-7b

# For PromptEnhancer-32B
huggingface-cli download PromptEnhancer/PromptEnhancer-32B --local-dir ./models/promptenhancer-32b
```

## Quickstart

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

### Using PromptEnhancerV2

```python
from inference.prompt_enhancer_v2 import PromptEnhancerV2

models_root_path = "./models/promptenhancer-32b"

enhancer = PromptEnhancerV2(models_root_path=models_root_path, device_map="auto")

# Enhance a prompt (Chinese or English)
user_prompt = "韩系插画风女生头像，粉紫色短发+透明感腮红，侧光渲染。"
new_prompt = enhancer.predict(
    prompt_cot=user_prompt,
    device="cuda"
)

print("Enhanced:", new_prompt)
```

## Parameters

*   `models_root_path`: The local path or repo ID for the model. Supports models with `trust_remote_code`.
*   `device_map`: Device mapping (default `auto`).
*   `predict(...)`:
    *   `prompt_cot` (str): The input prompt to be rewritten.
    *   `sys_prompt` (str): An optional system prompt. A default is provided for image prompt rewriting.
    *   `temperature` (float): Enables sampling if `>0`. Use `0` for deterministic generation.
    *   `top_p` (float): Nucleus sampling threshold (effective when sampling is enabled).
    *   `max_new_tokens` (int): The maximum number of new tokens to generate.

## TODO

*   [ ] Open source AlignEvaluator model.
*   [x] Open source PromptEnhancer-32B model.

## Citation

If you find this project helpful, please cite it using the following BibTeX entry:

```bibtex
@article{promptenhancer,
  title={PromptEnhancer: A Simple Approach to Enhance Text-to-Image Models via Chain-of-Thought Prompt Rewriting},
  author={Wang, Linqing and Xing, Ximing and Cheng, Yiji and Zhao, Zhiyuan and Tao, Jiale and Wang, QiXun and Li, Ruihuang and Chen, Comi and Li, Xin and Wu, Mingrui and Deng, Xinchi and Wang, Chunyu and Lu, Qinglin},
  journal={arXiv preprint arXiv:2509.04545},
  year={2025}
}
```

## Acknowledgements

This project acknowledges the contributions of the following open-source projects and communities: [Transformers](https://huggingface.co/transformers) and [HuggingFace](https://huggingface.co).

## Contact

For questions, feedback, or to reach the R&D and product teams, please contact the open-source team at hunyuan\_opensource@tencent.com.