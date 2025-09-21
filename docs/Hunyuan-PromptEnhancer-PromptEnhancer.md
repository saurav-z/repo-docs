# PromptEnhancer: Revolutionizing Text-to-Image Generation with Chain-of-Thought Prompt Rewriting

**PromptEnhancer** enhances text prompts for improved image generation by re-writing and restructuring them using a Chain-of-Thought approach.  [Explore the original repository](https://github.com/Hunyuan-PromptEnhancer/PromptEnhancer) for more details.

## Key Features

*   **Enhanced Prompt Structure:**  Transforms input prompts into clearer, layered, and logically consistent prompts tailored for text-to-image models.
*   **Intent Preservation:**  Maintains the original meaning across key elements like subjects, actions, styles, and relationships.
*   **"Global-to-Details-to-Summary" Narrative:**  Restructures prompts to describe the main elements first, then details, concluding with a style summary.
*   **Robust Output Parsing:**  Prioritizes the `<answer>...</answer>` tag for clean extraction; gracefully falls back to removing `<think>...</think>` if needed or returns the original prompt.
*   **Configurable Inference:**  Offers adjustable parameters (temperature, top\_p, max\_new\_tokens) to fine-tune determinism and creativity.

## Updates

*   **[2025-09-18]** ✨  Try the [PromptEnhancer-32B](https://huggingface.co/PromptEnhancer/PromptEnhancer-32B) for improved prompt enhancement!
*   **[2025-09-16]** Release of the [T2I-Keypoints-Eval dataset](https://huggingface.co/datasets/PromptEnhancer/T2I-Keypoints-Eval).
*   **[2025-09-07]** Release of the [PromptEnhancer-7B model](https://huggingface.co/tencent/HunyuanImage-2.1/tree/main/reprompt).
*   **[2025-09-07]** Release of the [technical report](https://arxiv.org/abs/2509.04545).

## Installation

```bash
pip install -r requirements.txt
```

## Model Download

```bash
# for PromptEnhancer-7B model
huggingface-cli download tencent/HunyuanImage-2.1/reprompt --local-dir ./models/promptenhancer-7b
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

## Parameters

*   `models_root_path`: Local path or repo id; supports `trust_remote_code` models.
*   `device_map`: Device mapping (default `auto`).
*   `predict(...)`:
    *   `prompt_cot` (str): Input prompt to rewrite.
    *   `sys_prompt` (str): Optional system prompt; a default is provided for image prompt rewriting.
    *   `temperature` (float): `>0` enables sampling; `0` for deterministic generation.
    *   `top_p` (float): Nucleus sampling threshold (effective when sampling).
    *   `max_new_tokens` (int): Maximum number of new tokens to generate.

## Citation

```bibtex
@article{promptenhancer,
  title={PromptEnhancer: A Simple Approach to Enhance Text-to-Image Models via Chain-of-Thought Prompt Rewriting},
  author={Wang, Linqing and Xing, Ximing and Cheng, Yiji and Zhao, Zhiyuan and Tao, Jiale and Wang, QiXun and Li, Ruihuang and Chen, Comi and Li, Xin and Wu, Mingrui and Deng, Xinchi and Wang, Chunyu and Lu, Qinglin},
  journal={arXiv preprint arXiv:2509.04545},
  year={2025}
}
```

## Acknowledgements

*   [Transformers](https://huggingface.co/transformers)
*   [HuggingFace](https://huggingface.co)

## Contact

For inquiries, please reach out to our open-source team or email us at [hunyuan\_opensource@tencent.com](mailto:hunyuan_opensource@tencent.com).

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Hunyuan-PromptEnhancer/PromptEnhancer&type=Date)](https://www.star-history.com/#Hunyuan-PromptEnhancer/PromptEnhancer&Date)