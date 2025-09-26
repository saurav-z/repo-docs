# PromptEnhancer: Revolutionizing Text-to-Image Generation with Advanced Prompt Rewriting

Enhance your text-to-image creations with PromptEnhancer, a powerful tool that refines your prompts for stunning results!  For more information, visit the original repository:  [Hunyuan-PromptEnhancer/PromptEnhancer](https://github.com/Hunyuan-PromptEnhancer/PromptEnhancer).

[![arXiv](https://img.shields.io/badge/Paper-arXiv:2509.04545-red?logo=arxiv)](https://www.arxiv.org/abs/2509.04545)
[![Zhihu](https://img.shields.io/badge/知乎-技术解读-0084ff?logo=zhihu)](https://zhuanlan.zhihu.com/p/1949013083109459515)
[![Hugging Face Model](https://img.shields.io/badge/Model-PromptEnhancer_7B-blue?logo=huggingface)](https://huggingface.co/tencent/HunyuanImage-2.1/tree/main/reprompt)
[![T2I-Keypoints-Eval Dataset](https://img.shields.io/badge/Benchmark-T2I_Keypoints_Eval-blue?logo=huggingface)](https://huggingface.co/datasets/PromptEnhancer/T2I-Keypoints-Eval)
[![Homepage](https://img.shields.io/badge/Homepage-PromptEnhancer-1abc9c?logo=homeassistant&logoColor=white)](https://hunyuan-promptenhancer.github.io/)
[![HunyuanImage2.1 Code](https://img.shields.io/badge/Code-HunyuanImage2.1-2ecc71?logo=github)](https://github.com/Tencent-Hunyuan/HunyuanImage-2.1)
[![HunyuanImage2.1 Model](https://img.shields.io/badge/Model-HunyuanImage2.1-3498db?logo=huggingface)](https://huggingface.co/tencent/HunyuanImage-2.1)
[![Hunyuan](https://img.shields.io/badge/Hunyuan-black.svg?logo=x height=22px)](https://x.com/TencentHunyuan)

---

<p align="center">
  <img src="assets/teaser-1.png" alt="PromptEnhancer Teaser"/>
</p>

## Key Features

*   **Enhanced Prompt Rewriting:** Transforms input prompts into clearer, more structured formats optimized for image generation.
*   **Preserves Intent:**  Ensures key elements (subject, action, style, etc.) are maintained throughout the rewriting process.
*   **Chain-of-Thought Approach:** Employs a "global–details–summary" structure for logically consistent and layered prompts.
*   **Robust Parsing:**  Prioritizes `<answer>` tags for clean output and offers graceful fallback mechanisms.
*   **Flexible Configuration:** Adjust inference parameters (temperature, top_p, max\_new\_tokens) for tailored results.
*   **GGUF Model Support:** Now supports GGUF models for memory-efficient and faster inference, even on consumer hardware.

## What's New

*   **[2025-09-22]** Added GGUF model support for efficient inference with quantized models!
*   **[2025-09-18]** Try the [PromptEnhancer-32B](https://huggingface.co/PromptEnhancer/PromptEnhancer-32B) for higher-quality prompt enhancement!
*   **[2025-09-16]** Released [T2I-Keypoints-Eval dataset](https://huggingface.co/datasets/PromptEnhancer/T2I-Keypoints-Eval).
*   **[2025-09-07]** Released [PromptEnhancer-7B model](https://huggingface.co/tencent/HunyuanImage-2.1/tree/main/reprompt).
*   **[2025-09-07]** Released [technical report](https://arxiv.org/abs/2509.04545).

## Getting Started

### Prerequisites

*   **Python:** 3.8 or higher
*   **CUDA:** 11.8+ (recommended for GPU acceleration)
*   **Storage:** At least 20GB free space for models
*   **Memory:** 8GB+ RAM (16GB+ recommended for 32B models)

### Installation

Choose the appropriate installation method:

**Option 1: Standard Installation (Recommended)**

```bash
pip install -r requirements.txt
```

**Option 2: GGUF Installation (For Quantized Models)**

```bash
chmod +x script/install_gguf.sh && ./script/install_gguf.sh
```

> **💡 Tip:** GGUF installation is recommended for optimal performance and efficiency, particularly with the 32B model.

## Model Download

### 🎯 Quick Start (Recommended)

For most users, start with the **PromptEnhancer-7B** model for the best balance of quality and efficiency:

```bash
# Download PromptEnhancer-7B (13GB)
huggingface-cli download tencent/HunyuanImage-2.1/reprompt --local-dir ./models/promptenhancer-7b
```

### Model Comparison & Selection Guide

| Model                     | Size  | Quality   | Memory       | Best For                                  |
| ------------------------- | ----- | --------- | ------------ | ----------------------------------------- |
| **PromptEnhancer-7B**     | 13GB  | High      | 8GB+         | Most users, balanced performance          |
| **PromptEnhancer-32B**    | 64GB  | Highest   | 32GB+        | Research, highest quality needs           |
| **32B-Q8\_0 (GGUF)**     | 35GB  | Highest   | 35GB+        | High-end GPUs (H100, A100)              |
| **32B-Q6\_K (GGUF)**     | 27GB  | Excellent | 27GB+        | RTX 4090, RTX 5090                       |
| **32B-Q4\_K\_M (GGUF)** | 20GB  | Good      | 20GB+        | RTX 3090, RTX 4080                       |

### Standard Models (Full Precision)

```bash
# PromptEnhancer-7B (recommended for most users)
huggingface-cli download tencent/HunyuanImage-2.1/reprompt --local-dir ./models/promptenhancer-7b

# PromptEnhancer-32B (for highest quality)
huggingface-cli download PromptEnhancer/PromptEnhancer-32B --local-dir ./models/promptenhancer-32b
```

### GGUF Models (Quantized - Memory Efficient)

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

> **🚀 Performance Tip:**  GGUF models offer significant memory reduction (50-75%) without compromising quality.  Consider Q6\_K for the best quality/memory trade-off.

## Quickstart

### Using HunyuanPromptEnhancer (Standard Models)

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

## GGUF Model Benefits

🚀 **Why Use GGUF Models?**

*   **Memory Efficiency:** Reduces VRAM usage by 50-75% compared to full precision models.
*   **Faster Inference:** Optimized for CPU and GPU acceleration with llama.cpp.
*   **Quality Preservation:**  Q8\_0 and Q6\_K models maintain excellent output quality.
*   **Easy Deployment:**  Single file format with no complex dependencies.
*   **GPU Acceleration:** Full CUDA support for high-performance inference.

| Model         | Size  | Quality   | VRAM Usage | Best For                        |
| ------------- | ----- | --------- | ---------- | ------------------------------- |
| Q8\_0          | 35GB  | Highest   | ~35GB      | High-end GPUs (H100, A100)      |
| Q6\_K          | 27GB  | Excellent | ~27GB     | RTX 4090, RTX 5090              |
| Q4\_K\_M      | 20GB  | Good      | ~20GB      | RTX 3090, RTX 4080              |

## Parameters

### Standard Models (Transformers)

*   `models_root_path`: Local path or repo id; supports `trust_remote_code` models.
*   `device_map`: Device mapping (default `auto`).
*   `predict(...)`:
    *   `prompt_cot` (str): Input prompt to rewrite.
    *   `sys_prompt` (str): Optional system prompt; a default is provided for image prompt rewriting.
    *   `temperature` (float): `>0` enables sampling; `0` for deterministic generation.
    *   `top_p` (float): Nucleus sampling threshold (effective when sampling).
    *   `max_new_tokens` (int): Maximum number of new tokens to generate.

### GGUF Models

*   `model_path` (str): Path to GGUF model file (auto-detected if in models/ folder).
*   `n_ctx` (int): Context window size (default: 8192, recommended: 1024 for short prompts).
*   `n_gpu_layers` (int): Number of layers to offload to GPU (-1 for all layers).
*   `verbose` (bool): Enable verbose logging from llama.cpp.

## Citation

If you utilize this project, please cite the following:

```bibtex
@article{promptenhancer,
  title={PromptEnhancer: A Simple Approach to Enhance Text-to-Image Models via Chain-of-Thought Prompt Rewriting},
  author={Wang, Linqing and Xing, Ximing and Cheng, Yiji and Zhao, Zhiyuan and Tao, Jiale and Wang, QiXun and Li, Ruihuang and Chen, Comi and Li, Xin and Wu, Mingrui and Deng, Xinchi and Wang, Chunyu and Lu, Qinglin},
  journal={arXiv preprint arXiv:2509.04545},
  year={2025}
}
```

## Acknowledgements

We extend our gratitude to the open-source projects and communities, including [Transformers](https://huggingface.co/transformers) and [HuggingFace](https://huggingface.co/), for their invaluable contributions to open research and exploration.

## Contact

For inquiries or feedback, contact our open-source team or reach us via email at hunyuan\_opensource@tencent.com.

## GitHub Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Hunyuan-PromptEnhancer/PromptEnhancer&type=Date)](https://www.star-history.com/#Hunyuan-PromptEnhancer/PromptEnhancer&Date)