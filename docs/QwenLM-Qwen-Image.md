<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/qwen_image_logo.png" width="400"/>
<p> 

<p align="center">
    💜 <a href="https://chat.qwen.ai/">Qwen Chat</a> |
    🤗 <a href="https://huggingface.co/Qwen/Qwen-Image">HuggingFace(T2I)</a> |
    🤗 <a href="https://huggingface.co/Qwen/Qwen-Image-Edit">HuggingFace(Edit)</a> | 
    🤖 <a href="https://modelscope.cn/models/Qwen/Qwen-Image">ModelScope-T2I</a> | 
    🤖 <a href="https://modelscope.cn/models/Qwen/Qwen-Image-Edit">ModelScope-Edit</a> | 
    📑 <a href="https://arxiv.org/abs/2508.02324">Tech Report</a> | 
    📑 <a href="https://qwenlm.github.io/blog/qwen-image/">Blog(T2I)</a> | 
    📑 <a href="https://qwenlm.github.io/blog/qwen-image-edit/">Blog(Edit)</a>
<br>
    🖥️ <a href="https://huggingface.co/spaces/Qwen/Qwen-Image">T2I Demo</a> | 
    🖥️ <a href="https://huggingface.co/spaces/Qwen/Qwen-Image-Edit">Edit Demo</a> | 
    💬 <a href="https://github.com/QwenLM/Qwen-Image/blob/main/assets/wechat.png">WeChat (微信)</a> | 
    🫨 <a href="https://discord.gg/CV4E9rpNSD">Discord</a>
</p>

<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/merge3.jpg" width="1024"/>
<p>

# Qwen-Image: Unleash Your Creativity with Advanced Image Generation and Editing

**Qwen-Image is a cutting-edge 20B MMDiT image foundation model that excels in complex text rendering and precise image editing, revolutionizing how you create and manipulate images.** Explore the [Qwen-Image GitHub repository](https://github.com/QwenLM/Qwen-Image) for more details.

## Key Features

*   **Exceptional Text Rendering:** Generate images with unparalleled accuracy in rendering text, especially for Chinese characters.
*   **Precise Image Editing:** Perform advanced editing tasks, including style transfer, object manipulation, and detail enhancement.
*   **Versatile Image Generation:** Create a wide array of images, from photorealistic scenes to artistic styles.
*   **Image Understanding Capabilities:** Supports object detection, semantic segmentation, and other image understanding tasks.
*   **Multi-GPU API Server:** Efficient local deployment with a Gradio-based web interface.

## Quick Start Guide

### 1. Prerequisites

*   Ensure you have transformers >=4.51.3
*   Install the latest version of diffusers:

    ```bash
    pip install git+https://github.com/huggingface/diffusers
    ```

### 2. Text-to-Image Generation

```python
from diffusers import DiffusionPipeline
import torch

model_name = "Qwen/Qwen-Image"

# Load the pipeline (use CUDA if available)
if torch.cuda.is_available():
    torch_dtype = torch.bfloat16
    device = "cuda"
else:
    torch_dtype = torch.float32
    device = "cpu"

pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch_dtype)
pipe = pipe.to(device)

positive_magic = {
    "en": ", Ultra HD, 4K, cinematic composition.",
    "zh": ", 超清，4K，电影级构图."
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

### 3. Image Editing

```python
import os
from PIL import Image
import torch

from diffusers import QwenImageEditPipeline

pipeline = QwenImageEditPipeline.from_pretrained("Qwen/Qwen-Image-Edit")
print("pipeline loaded")
pipeline.to(torch.bfloat16)
pipeline.to("cuda")
pipeline.set_progress_bar_config(disable=None)

image = Image.open("./input.png").convert("RGB")
prompt = "Change the rabbit's color to purple, with a flash light background."

inputs = {
    "image": image,
    "prompt": prompt,
    "generator": torch.manual_seed(0),
    "true_cfg_scale": 4.0,
    "negative_prompt": " ",
    "num_inference_steps": 50,
}

with torch.inference_mode():
    output = pipeline(**inputs)
    output_image = output.images[0]
    output_image.save("output_image_edit.png")
    print("image saved at", os.path.abspath("output_image_edit.png"))
```

## Advanced Usage

### Prompt Enhancement

Use the prompt enhancement tool for improved results:

*   **Text-to-Image:**

    ```python
    from tools.prompt_utils import rewrite
    prompt = rewrite(prompt)
    ```
*   **Image Editing:**

    ```python
    from tools.prompt_utils import polish_edit_prompt
    prompt = polish_edit_prompt(prompt, pil_image)
    ```

### Multi-GPU API Server Deployment

1.  Configure environment variables:

    ```bash
    export NUM_GPUS_TO_USE=4          # Number of GPUs to use
    export TASK_QUEUE_SIZE=100        # Task queue size
    export TASK_TIMEOUT=300           # Task timeout in seconds
    ```

2.  Run the demo server:

    ```bash
    cd src
    DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxx python examples/demo.py
    ```

## Show Cases

[Include concise descriptions and images showcasing the capabilities of Qwen-Image: general cases, editing, and tutorials.]

## AI Arena

Qwen-Image is evaluated on AI Arena for fair comparison with other models. Visit the [AI Arena Leaderboard](https://aiarena.alibaba-inc.com/corpora/arena/leaderboard?arenaType=text2image) to view the latest rankings.

## Community Support

*   **Hugging Face:** Day 0 support, LoRA, and fine-tuning workflows.
*   **ModelScope:** DiffSynth-Studio, DiffSynth-Engine, and ModelScope AIGC Central.
*   **WaveSpeedAI:** Integrated on their platform.
*   **LiblibAI:** Native support.
*   **cache-dit:** Cache acceleration support with DBCache, TaylorSeer, and Cache CFG.

## License Agreement

Qwen-Image is licensed under Apache 2.0.

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

## Contact and Join Us

Connect with the research team and stay updated:

*   Join our [Discord](https://discord.gg/CV4E9rpNSD)
*   Connect via [WeChat groups](assets/wechat.png)

Contribute and collaborate:

*   Submit issues and pull requests on [GitHub](https://github.com/QwenLM/Qwen-Image)

Recruitment:

*   FTE and research intern opportunities: fulai.hr@alibaba-inc.com

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=QwenLM/Qwen-Image&type=Date)](https://www.star-history.com/#QwenLM/Qwen-Image&Date)
```
Key improvements and optimizations:

*   **SEO-Optimized Headline:**  Uses a strong keyword ("Qwen-Image") and action-oriented language ("Unleash Your Creativity") to attract users and improve search engine visibility.
*   **Clear Sectioning with Headings:** Improves readability and allows users to quickly find relevant information.
*   **Concise Bulleted Lists:** Highlights key features for easy scanning.
*   **One-Sentence Hook:**  Immediately grabs the reader's attention and introduces the core benefit.
*   **Actionable Quick Start:** Provides clear, executable code snippets to get started.
*   **Emphasis on Practical Application:**  Showcases how to use Qwen-Image, not just what it is.
*   **Stronger Calls to Action:** Encourages users to explore the repo, join the community, and contribute.
*   **Links to Related Resources:** Provides easy access to demos, blogs, and technical reports.
*   **Reorganized Content:** Streamlined the content for clarity and better flow.
*   **Concise Summaries:** Avoids unnecessary repetition and focuses on the most important information.
*   **Clear Structure for Community Support:**  Makes it easy to find the resources that matter.
*   **Updated Contact Information:**  Ensures users have the most up-to-date methods of reaching the team.
*   **Star History:**  Provides social proof of the project's popularity