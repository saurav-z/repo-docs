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

## Qwen-Image: Unleash Stunning AI-Powered Image Generation and Editing

**Qwen-Image is a powerful suite of AI models for both text-to-image generation and advanced image editing, offering exceptional performance in complex text rendering, especially for Chinese.** Explore the cutting edge of AI-driven visual creation with Qwen-Image! For more details, see the [original repository](https://github.com/QwenLM/Qwen-Image).

### Key Features:

*   **Text-to-Image Generation:** Create high-quality images from text prompts with impressive text rendering capabilities.
*   **Image Editing:**  Refine and modify existing images with precision, including support for:
    *   Multi-image editing (Qwen-Image-Edit-2509)
    *   Enhanced single-image consistency
    *   Native ControlNet support
*   **Chinese Language Excellence:** Optimized for superior text rendering and understanding of Chinese language prompts.
*   **Easy to Use:** Simple installation and code snippets to get you started quickly.
*   **Multi-GPU API Server:** Deploy the model locally with a Gradio-based web interface for efficient processing and queue management.
*   **Extensive Community Support:** Active support from Hugging Face, ModelScope, WaveSpeedAI, LiblibAI, and cache-dit.
*   **Open Source:** Licensed under Apache 2.0 for free use and modification.

### What's New

*   **Qwen-Image-Edit-2509 (September 2025 Update):**  Significant improvements including multi-image editing, enhanced single-image consistency (person, product, and text), and native support for ControlNet.  Try it now on [Qwen Chat](https://qwen.ai/)!
*   **LoRA Support:**  Supports a variety of LoRA models, such as MajicBeauty LoRA, to generate realistic beauty images.

### Quick Start

#### 1. Requirements

Make sure your `transformers>=4.51.3` (Supporting Qwen2.5-VL).

#### 2. Installation

Install the latest version of diffusers:

```bash
pip install git+https://github.com/huggingface/diffusers
```

#### 3. Code Examples

**Qwen-Image (Text-to-Image)**

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
    "en": ", Ultra HD, 4K, cinematic composition.", # for english prompt
    "zh": ", 超清，4K，电影级构图." # for chinese prompt
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

**Qwen-Image-Edit (Image Editing - Single Image Input)**

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

**Qwen-Image-Edit-2509 (Image Editing - Multiple Image Support & Improved Consistency)**

```python
import os
import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline
from io import BytesIO
import requests

pipeline = QwenImageEditPlusPipeline.from_pretrained("Qwen/Qwen-Image-Edit-2509", torch_dtype=torch.bfloat16)
print("pipeline loaded")

pipeline.to('cuda')
pipeline.set_progress_bar_config(disable=None)
image1 = Image.open(BytesIO(requests.get("https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2509/edit2509_1.jpg").content))
image2 = Image.open(BytesIO(requests.get("https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2509/edit2509_2.jpg").content))
prompt = "The magician bear is on the left, the alchemist bear is on the right, facing each other in the central park square."
inputs = {
    "image": [image1, image2],
    "prompt": prompt,
    "generator": torch.manual_seed(0),
    "true_cfg_scale": 4.0,
    "negative_prompt": " ",
    "num_inference_steps": 40,
    "guidance_scale": 1.0,
    "num_images_per_prompt": 1,
}
with torch.inference_mode():
    output = pipeline(**inputs)
    output_image = output.images[0]
    output_image.save("output_image_edit_plus.png")
    print("image saved at", os.path.abspath("output_image_edit_plus.png"))
```

### Advanced Usage

*   **Prompt Enhancement:**  Enhance prompt optimization and multi-language support using the official Prompt Enhancement Tool powered by Qwen-Plus.

    ```python
    from tools.prompt_utils import rewrite
    prompt = rewrite(prompt)
    ```

*   **Prompt Enhancement for Image Edit:** Improve the stability of editing tasks using the official Prompt Enhancement Tool powered by Qwen-VL-Max.

    ```python
    from tools.prompt_utils import polish_edit_prompt
    prompt = polish_edit_prompt(prompt, pil_image)
    ```

### Deploy Qwen-Image

Qwen-Image supports Multi-GPU API Server for local deployment:

### Multi-GPU API Server Pipeline & Usage

The Multi-GPU API Server will start a Gradio-based web interface with:
- Multi-GPU parallel processing
- Queue management for high concurrency
- Automatic prompt optimization
- Support for multiple aspect ratios

Configuration via environment variables:
```bash
export NUM_GPUS_TO_USE=4          # Number of GPUs to use
export TASK_QUEUE_SIZE=100        # Task queue size
export TASK_TIMEOUT=300           # Task timeout in seconds
```

```bash
# Start the gradio demo server, api key for prompt enhance
cd src
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxx python examples/demo.py 
```

### Showcase

#### Qwen-Image Edit-2509 Examples

*   **Multi-Image Input:** Demonstrations with "person + person", "person + scene", and "person + object" inputs.

*   **Enhanced Consistency:** Showcases improved person consistency, including various portrait styles and pose transformations. Also shows enhanced product and text consistency.

*   **ControlNet Support:** Example showing support for keypoint control and sketch inputs.

*Refer to the original README for example images.*

### AI Arena

Qwen-Image is evaluated on [AI Arena](https://aiarena.alibaba-inc.com), an open benchmarking platform built on the Elo rating system, for a transparent comparison against state-of-the-art closed-source APIs. View the latest leaderboard at [AI Arena Learboard](https://aiarena.alibaba-inc.com/corpora/arena/leaderboard?arenaType=text2image). Contact weiyue.wy@alibaba-inc.com for model deployment on AI Arena.

### Community Support

*   **Hugging Face:** Day-0 support in Diffusers, LoRA, and finetuning development.
*   **ModelScope:** Comprehensive support with DiffSynth-Studio, DiffSynth-Engine, and ModelScope AIGC Central.
*   **WaveSpeedAI:** Deployed on their platform from day 0.
*   **LiblibAI:** Native support.
*   **cache-dit:**  Offers cache acceleration support.

### License

Qwen-Image is licensed under the [Apache 2.0 License](LICENSE).

### Citation

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

### Contact and Join Us

*   Join our [Discord](https://discord.gg/z3GAxXZ9Ce) or scan the QR code to connect via our [WeChat groups](assets/wechat.png) .
*   Report issues and submit pull requests on [GitHub](https://github.com/QwenLM/Qwen-Image).
*   For FTE or research intern opportunities, contact fulai.hr@alibaba-inc.com.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=QwenLM/Qwen-Image&type=Date)](https://www.star-history.com/#QwenLM/Qwen-Image&Date)