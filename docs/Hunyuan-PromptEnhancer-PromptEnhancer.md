# PromptEnhancer: Supercharge Your Text-to-Image Generation with Smarter Prompts

**Enhance your text-to-image generation with PromptEnhancer, a cutting-edge tool that rewrites your prompts for more detailed and accurate results.**  [Check out the original repo here](https://github.com/Hunyuan-PromptEnhancer/PromptEnhancer).

---

[![arXiv](https://img.shields.io/badge/Paper-arXiv:2509.04545-red?logo=arxiv)](https://www.arxiv.org/abs/2509.04545)
[![Zhihu](https://img.shields.io/badge/%E7%9F%A5%E4%B9%8E-%E6%8A%80%E6%9C%AF%E8%A7%A3%E8%AF%BB-0084ff?logo=zhihu)](https://zhuanlan.zhihu.com/p/1949013083109459515)
[![Hugging Face Model](https://img.shields.io/badge/Model-PromptEnhancer_7B-blue?logo=huggingface)](https://huggingface.co/tencent/HunyuanImage-2.1/tree/main/reprompt)
[![T2I Eval Dataset](https://img.shields.io/badge/Benchmark-T2I_Keypoints_Eval-blue?logo=huggingface)](https://huggingface.co/datasets/PromptEnhancer/T2I-Keypoints-Eval)
[![Homepage](https://img.shields.io/badge/Homepage-PromptEnhancer-1abc9c?logo=homeassistant&logoColor=white)](https://hunyuan-promptenhancer.github.io/)
[![HunyuanImage2.1 Code](https://img.shields.io/badge/Code-HunyuanImage2.1-2ecc71?logo=github)](https://github.com/Tencent-Hunyuan/HunyuanImage-2.1)
[![HunyuanImage2.1 Model](https://img.shields.io/badge/Model-HunyuanImage2.1-3498db?logo=huggingface)](https://huggingface.co/tencent/HunyuanImage-2.1)
[![Hunyuan](https://img.shields.io/badge/Hunyuan-black.svg?logo=x)](https://x.com/TencentHunyuan)

---
<p align="center">
  <img src="assets/teaser-1.png" alt="PromptEnhancer Teaser"/>
</p>
## Key Features

*   **Enhanced Prompts:** Rewrites input prompts into clearer, more detailed, and logically structured descriptions.
*   **Intent Preservation:** Maintains the original meaning of your prompts, including subject, action, style, and more.
*   **"Global-to-Details-to-Summary" Structure:** Organizes prompts for optimal image generation, describing primary elements first, followed by details and style.
*   **Robust Parsing:** Handles various outputs gracefully, prioritizing `<answer>` tags and providing fallbacks.
*   **Flexible Parameters:** Customize generation with temperature, top_p, and max_new_tokens settings.
*   **GGUF Support:** Optimized for efficient inference with quantized models, including GPU acceleration.

## What's New

*   **[2025-09-22]** 🚀 Added **GGUF model support** for efficient inference with quantized models! Thanks @mradermacher!
*   **[2025-09-18]** ✨  Try the [PromptEnhancer-32B](https://huggingface.co/PromptEnhancer/PromptEnhancer-32B) model for higher quality results.
*   **[2025-09-16]** Released the [T2I-Keypoints-Eval dataset](https://huggingface.co/datasets/PromptEnhancer/T2I-Keypoints-Eval).
*   **[2025-09-07]** Released the [PromptEnhancer-7B model](https://huggingface.co/tencent/HunyuanImage-2.1/tree/main/reprompt) and [technical report](https://arxiv.org/abs/2509.04545).

## Getting Started

### Prerequisites

*   **Python:** 3.8 or higher
*   **CUDA:** 11.8+ (Recommended for GPU acceleration)
*   **Storage:** At least 20GB free space (for models)
*   **Memory:** 8GB+ RAM (16GB+ recommended for 32B models)

### Installation

**Option 1: Standard Installation** (Recommended)

```bash
pip install -r requirements.txt
```

**Option 2: GGUF Installation** (For Quantized Models - Faster & Lower Memory)

```bash
chmod +x script/install_gguf.sh && ./script/install_gguf.sh
```

### Model Download

**Quick Start (Recommended):**

```bash
# Download PromptEnhancer-7B (13GB) - Best balance of quality and efficiency
huggingface-cli download tencent/HunyuanImage-2.1/reprompt --local-dir ./models/promptenhancer-7b
```

**Model Selection Guide:**

| Model                   | Size  | Quality     | Memory Required | Best For                                    |
| ----------------------- | ----- | ----------- | --------------- | ------------------------------------------- |
| **PromptEnhancer-7B**   | 13GB  | High        | 8GB+            | Most users, balanced performance            |
| **PromptEnhancer-32B**  | 64GB  | Highest     | 32GB+           | Research, highest quality needs            |
| **32B-Q8_0 (GGUF)**     | 35GB  | Highest     | ~35GB           | High-end GPUs (H100, A100)                 |
| **32B-Q6_K (GGUF)**     | 27GB  | Excellent   | ~27GB           | RTX 4090, RTX 5090                         |
| **32B-Q4_K_M (GGUF)**   | 20GB  | Good        | ~20GB           | RTX 3090, RTX 4080                         |

**Standard Models:**

```bash
# PromptEnhancer-7B (recommended)
huggingface-cli download tencent/HunyuanImage-2.1/reprompt --local-dir ./models/promptenhancer-7b

# PromptEnhancer-32B (highest quality)
huggingface-cli download PromptEnhancer/PromptEnhancer-32B --local-dir ./models/promptenhancer-32b
```

**GGUF Models (Quantized - Memory Efficient):**

```bash
# Create models directory
mkdir -p ./models

# Choose a model based on your GPU memory:
# Q8_0: Highest quality (35GB)
huggingface-cli download mradermacher/PromptEnhancer-32B-GGUF PromptEnhancer-32B.Q8_0.gguf --local-dir ./models

# Q6_K: Excellent quality (27GB) - Recommended for RTX 4090
huggingface-cli download mradermacher/PromptEnhancer-32B-GGUF PromptEnhancer-32B.Q6_K.gguf --local-dir ./models

# Q4_K_M: Good quality (20GB) - Recommended for RTX 3090/4080
huggingface-cli download mradermacher/PromptEnhancer-32B-GGUF PromptEnhancer-32B.Q4_K_M.gguf --local-dir ./models
```

**🚀 Performance Tip**:  GGUF models offer significant memory reduction (50-75%) with minimal quality loss. Q6\_K provides the best balance of quality and memory usage.

## Quickstart Examples

### Using Standard Models (Transformers)

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

### Using GGUF Models (Quantized)

```python
from inference.prompt_enhancer_gguf import PromptEnhancerGGUF

# Auto-detects Q8_0 model in models/ folder
enhancer = PromptEnhancerGGUF(
    model_path="./models/PromptEnhancer-32B.Q8_0.gguf",  # Optional: auto-detected
    n_ctx=1024,        # Context window size
    n_gpu_layers=-1,   # Use all GPU layers
)

# Enhance a prompt
user_prompt = "woman in jungle"
enhanced_prompt = enhancer.predict(
    user_prompt,
    temperature=0.3,
    top_p=0.9,
    max_new_tokens=512,
)

print("Enhanced:", enhanced_prompt)
```

### Command Line Usage (GGUF)

```bash
# Simple usage - auto-detects model in models/ folder
python inference/prompt_enhancer_gguf.py

# Or specify model path
GGUF_MODEL_PATH="./models/PromptEnhancer-32B.Q8_0.gguf" python inference/prompt_enhancer_gguf.py
```

## GGUF Model Advantages

*   **Memory Efficiency:** Reduced VRAM usage (50-75% less).
*   **Faster Inference:** Optimized for both CPU and GPU with llama.cpp.
*   **Quality Preservation:** Q8\_0 and Q6\_K models maintain excellent output.
*   **Easy Deployment:** Simple single-file format.
*   **GPU Acceleration:** Full CUDA support for faster results.

## Parameters

### Standard Models

*   `models_root_path`: Local path or repo ID.
*   `device_map`: Device mapping (defaults to `auto`).
*   `predict(...)`:
    *   `prompt_cot` (str): Input prompt.
    *   `sys_prompt` (str): Optional system prompt.
    *   `temperature` (float):  Controls sampling.
    *   `top_p` (float): Nucleus sampling threshold.
    *   `max_new_tokens` (int):  Maximum generated tokens.

### GGUF Models

*   `model_path` (str):  Path to the GGUF model.
*   `n_ctx` (int): Context window size (default: 8192, recommended 1024 for short prompts).
*   `n_gpu_layers` (int): Layers to offload to GPU (-1 for all).
*   `verbose` (bool): Enable verbose logging.

## Citation

If you use this project, please cite our work:

```bibtex
@article{promptenhancer,
  title={PromptEnhancer: A Simple Approach to Enhance Text-to-Image Models via Chain-of-Thought Prompt Rewriting},
  author={Wang, Linqing and Xing, Ximing and Cheng, Yiji and Zhao, Zhiyuan and Tao, Jiale and Wang, QiXun and Li, Ruihuang and Chen, Comi and Li, Xin and Wu, Mingrui and Deng, Xinchi and Wang, Chunyu and Lu, Qinglin},
  journal={arXiv preprint arXiv:2509.04545},
  year={2025}
}
```

## Acknowledgements

We thank the [Hugging Face Transformers](https://huggingface.co/transformers) and [Hugging Face](https://huggingface.co/) communities for their valuable contributions.

## Contact

For inquiries, contact our open-source team or email hunyuan_opensource@tencent.com.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Hunyuan-PromptEnhancer/PromptEnhancer&type=Date)](https://www.star-history.com/#Hunyuan-PromptEnhancer/PromptEnhancer&Date)