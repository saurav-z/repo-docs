# PromptEnhancer: Elevating Text-to-Image Models with Intelligent Prompt Rewriting

**PromptEnhancer enhances text-to-image models by rewriting prompts to be clearer, more structured, and logically consistent, leading to superior image generation.** Explore the original repository on [GitHub](https://github.com/Hunyuan-PromptEnhancer/PromptEnhancer).

## Key Features

*   **Intent Preservation:** Maintains the original meaning of your prompt, covering subject, action, quantity, style, and more.
*   **Structured Rewriting:** Employs a "global–details–summary" approach, creating organized prompts for better results.
*   **Robust Parsing:** Offers flexible output handling, prioritizing the `<answer>...</answer>` tag and providing fallback options for clean text extraction.
*   **Configurable Inference:** Fine-tune results with adjustable parameters like temperature, top\_p, and max\_new\_tokens.
*   **GGUF Model Support:** Utilize quantized models for efficient inference on various hardware configurations.

## What's New

*   **[2025-09-22]**: GGUF model support added by @mradermacher, increasing efficient inference with quantized models!
*   **[2025-09-18]**: PromptEnhancer-32B model released, enabling higher-quality prompt enhancement!
*   **[2025-09-16]**: T2I-Keypoints-Eval dataset released.
*   **[2025-09-07]**: PromptEnhancer-7B model and technical report released.

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

### Using GGUF Models (Quantized, Faster)

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

## Installation

### Prerequisites

*   **Python:** 3.8 or higher
*   **CUDA:** 11.8+ (recommended for GPU acceleration)
*   **Storage:** At least 20GB free space for models
*   **Memory:** 8GB+ RAM (16GB+ recommended for 32B models)

### Installation Options

#### Option 1: Standard Installation (Recommended)

```bash
pip install -r requirements.txt
```

#### Option 2: GGUF Installation (For Quantized Models)

```bash
chmod +x script/install_gguf.sh && ./script/install_gguf.sh
```

## Model Download

### Model Comparison & Selection Guide

| Model | Size   | Quality | Memory | Best For                             |
| :---- | :----- | :------ | :----- | :----------------------------------- |
| 7B    | 13GB   | High    | 8GB+   | Most users, balanced performance    |
| 32B   | 64GB   | Highest | 32GB+  | Research, highest quality needs       |
| Q8\_0 | 35GB   | Highest | ~35GB  | High-end GPUs (H100, A100)           |
| Q6\_K | 27GB   | Excellent | ~27GB | RTX 4090, RTX 5090                  |
| Q4\_K\_M | 20GB   | Good    | ~20GB  | RTX 3090, RTX 4080                  |

### Standard Models (Full Precision)

```bash
# PromptEnhancer-7B (recommended)
huggingface-cli download tencent/HunyuanImage-2.1/reprompt --local-dir ./models/promptenhancer-7b

# PromptEnhancer-32B (for highest quality)
huggingface-cli download PromptEnhancer/PromptEnhancer-32B --local-dir ./models/promptenhancer-32b
```

### GGUF Models (Quantized)

```bash
# Create models directory
mkdir -p ./models

# Choose one based on your GPU memory:
# Q8_0: Highest quality (35GB)
huggingface-cli download mradermacher/PromptEnhancer-32B-GGUF PromptEnhancer-32B.Q8_0.gguf --local-dir ./models

# Q6_K: Excellent quality (27GB) - Recommended for RTX 4090
huggingface-cli download mradermacher/PromptEnhancer-32B-GGUF PromptEnhancer-32B.Q6_K.gguf --local-dir ./models

# Q4_K_M: Good quality (20GB) - Recommended for RTX 3090/4080
huggingface-cli download mradermacher/PromptEnhancer-32B-GGUF PromptEnhancer-32B.Q4_K_M.gguf --local-dir ./models
```

## GGUF Model Benefits

🚀 **Why use GGUF models?**

*   **Memory Efficient:** 50-75% less VRAM usage.
*   **Faster Inference:** Optimized for CPU and GPU acceleration.
*   **Quality Preserved:** Q8\_0 and Q6\_K maintain excellent output quality.
*   **Easy Deployment:** Single file format, no complex dependencies.
*   **GPU Acceleration:** Full CUDA support for high-performance inference.

## Parameters

### Standard Models (Transformers)

*   `models_root_path`: Local path or repo id; supports `trust_remote_code` models.
*   `device_map`: Device mapping (default `auto`).
*   `predict(...)`:
    *   `prompt_cot` (str): Input prompt to rewrite.
    *   `sys_prompt` (str): Optional system prompt; a default is provided.
    *   `temperature` (float): `>0` enables sampling; `0` for deterministic generation.
    *   `top_p` (float): Nucleus sampling threshold.
    *   `max_new_tokens` (int): Maximum number of new tokens to generate.

### GGUF Models

*   `model_path` (str): Path to GGUF model file.
*   `n_ctx` (int): Context window size (default: 8192).
*   `n_gpu_layers` (int): Number of layers to offload to GPU (-1 for all layers).
*   `verbose` (bool): Enable verbose logging from llama.cpp.

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

This project leverages the contributions of [Transformers](https://huggingface.co/transformers) and [HuggingFace](https://huggingface.co/).

## Contact

For inquiries, contact our open-source team or via email: hunyuan\_opensource@tencent.com.

## GitHub Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Hunyuan-PromptEnhancer/PromptEnhancer&type=Date)](https://www.star-history.com/#Hunyuan-PromptEnhancer/PromptEnhancer&Date)