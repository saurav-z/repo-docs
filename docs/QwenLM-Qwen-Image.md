<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/qwen_image_logo.png" width="400" alt="Qwen-Image Logo"/>
</p>

<p align="center">
    **Qwen-Image: Unleash Your Creativity with State-of-the-Art Image Generation and Editing.**
</p>

<p align="center">
    &nbsp&nbsp💜 <a href="https://chat.qwen.ai/">Qwen Chat</a>&nbsp&nbsp |
           &nbsp&nbsp🤗 <a href="https://huggingface.co/Qwen/Qwen-Image">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/models/Qwen/Qwen-Image">ModelScope</a>&nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://arxiv.org/abs/2508.02324">Tech Report</a> &nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://qwenlm.github.io/blog/qwen-image/">Blog</a> &nbsp&nbsp 
<br>
🖥️ <a href="https://huggingface.co/spaces/Qwen/Qwen-Image">Demo</a>&nbsp&nbsp | &nbsp&nbsp💬 <a href="https://github.com/QwenLM/Qwen-Image/blob/main/assets/wechat.png">WeChat (微信)</a>&nbsp&nbsp | &nbsp&nbsp🫨 <a href="https://discord.gg/CV4E9rpNSD">Discord</a>&nbsp&nbsp
</p>

<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/merge3.jpg" width="1024" alt="Qwen-Image Example"/>
</p>

## Overview

Qwen-Image is a powerful 20B parameter MMDiT image foundation model, setting new standards in **complex text rendering** and **precise image editing**. This model excels in both image generation and editing, with a particular strength in rendering text, especially Chinese. Access the original repository on [GitHub](https://github.com/QwenLM/Qwen-Image).

## Key Features

*   **Exceptional Text Rendering:** Accurately renders text in multiple languages, maintaining typographic details and contextual harmony.
*   **Versatile Image Generation:** Generates images in various styles, from photorealistic to artistic.
*   **Advanced Image Editing:** Enables style transfer, object manipulation, and pose modification.
*   **Image Understanding Capabilities:** Supports object detection, segmentation, and super-resolution.
*   **Multi-Language Support:**  Includes a prompt enhancement tool for multi-language optimization.

## Quick Start

### 1. Prerequisites

*   Ensure you have `transformers>=4.51.3`.
*   Install the latest version of `diffusers`:
    ```bash
    pip install git+https://github.com/huggingface/diffusers
    ```

### 2. Code Example

```python
from diffusers import DiffusionPipeline
import torch

model_name = "Qwen/Qwen-Image"

# Load the pipeline
if torch.cuda.is_available():
    torch_dtype = torch.bfloat16
    device = "cuda"
else:
    torch_dtype = torch.float32
    device = "cpu"

pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch_dtype)
pipe = pipe.to(device)

positive_magic = {
    "en": "Ultra HD, 4K, cinematic composition.", # for english prompt
    "zh": "超清，4K，电影级构图" # for chinese prompt
}

# Generate image
prompt = '''A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup," with a neon light beside it displaying "通义千问". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the poster is written "π≈3.1415926-53589793-23846264-33832795-02384197".'''

negative_prompt = " " # Recommended if you don't use a negative prompt.


# Generate with different aspect ratios
aspect_ratios = {
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1104),
    "3:4": (1104, 1472),
    "3:2": (1584, 1056),
    "2:3": (1056, 1584),
}

width, height = aspect_ratios["16:9"]

image = pipe(
    prompt=prompt + positive_magic["en"],
    negative_prompt=negative_prompt,
    width=width,
    height=height,
    num_inference_steps=50,
    true_cfg_scale=4.0,
    generator=torch.Generator(device="cuda").manual_seed(42)
).images[0]

image.save("example.png")
```

## Show Cases

[Include the existing Show Cases images here, each with a short, SEO-friendly description.  For example:]

**High-Fidelity Text Rendering:** Qwen-Image accurately renders text in diverse images.

<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/s1.jpg#center" alt="Text rendering example"/>
</p>

**Versatile Image Generation:** Explore a wide range of artistic styles.

<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/s2.jpg#center" alt="Image generation example"/>
</p>

**Advanced Image Editing:** Refine and manipulate images with ease.

<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/s3.jpg#center" alt="Image editing example"/>
</p>

**Image Understanding:** Enhanced image comprehension capabilities.

<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/s4.jpg#center" alt="Image understanding example"/>
</p>

## Advanced Usage

### Prompt Enhancement

Improve prompts using the Qwen-Plus powered Prompt Enhancement Tool:

```python
from tools.prompt_utils import rewrite
prompt = rewrite(prompt)
```

Or, run the script from the command line:

```bash
cd src
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx python examples/generate_w_prompt_enhance.py
```

## Deployment

### Multi-GPU API Server

Set up a Gradio-based web interface with:

*   Multi-GPU parallel processing
*   Queue management
*   Automatic prompt optimization
*   Support for multiple aspect ratios

**Configuration:**

```bash
export NUM_GPUS_TO_USE=4          # Number of GPUs to use
export TASK_QUEUE_SIZE=100        # Task queue size
export TASK_TIMEOUT=300           # Task timeout in seconds
```

**Run the demo:**

```bash
cd src
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxx python examples/demo.py
```

## AI Arena

[Optional: If it's important to promote the AI Arena, include a small section.  Otherwise, remove this section.]

Evaluate Qwen-Image's performance on the [AI Arena](https://aiarena.alibaba-inc.com), an open benchmarking platform with the Elo rating system.

[Include the AI Arena image, and a brief description.]

## Community Support

*   **Hugging Face:**  Supported via Diffusers, with LoRA and fine-tuning workflows in development.
*   **ModelScope:** Comprehensive support with offload, quantization and training.
*   **WaveSpeedAI:** Deployed on their platform.
*   **LiblibAI:** Native support.
*   **cache-dit:** Offers cache acceleration.

## License

Qwen-Image is licensed under the Apache 2.0 License.

## Citation

```bibtex
@misc{wu2025qwenimagetechnicalreport,
      title={Qwen-Image Technical Report}, 
      author={Chenfei Wu and Jiahao Li and Jingren Zhou and Junyang Lin and Kaiyuan Gao and Kun Yan and Sheng-ming Yin and Shuai Bai and Xiao Xu and Yilei Chen and Yuxiang Chen and Zecheng Tang and Zekai Zhang and Zhengyi Wang and An Yang and Bowen Yu and Chen Cheng and Dayiheng Liu and Deqing Li and Hang Zhang and Hao Meng and Hu Wei and Jingyuan Ni and Kai Chen and Kuan Cao and Liang Peng and Lin Qu and Minggang Wu and Peng Wang and Shuting Yu and Tingkun Wen and Wensen Feng and Xiaoxiao Xu and Yi Wang and Yichang Zhang and Yongqiang Zhu and Yujia Wu and Yuxuan Cai and Zenan Liu},
      year={2025},
      eprint={2508.02324},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2508.02324}, 
}
```

## Get in Touch

Join our [Discord](https://discord.gg/z3GAxXZ9Ce) or connect via WeChat.  We welcome your contributions, issues, and pull requests on GitHub.  For full-time or research internship opportunities, contact fulai.hr@alibaba-inc.com.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=QwenLM/Qwen-Image&type=Date)](https://www.star-history.com/#QwenLM/Qwen-Image&Date)