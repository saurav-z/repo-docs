<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/qwen_image_logo.png" width="400"/>
</p>

<p align="center">
    💜 <a href="https://chat.qwen.ai/">Qwen Chat</a> |
    🤗 <a href="https://huggingface.co/Qwen/Qwen-Image">HuggingFace(T2I)</a> |
    🤗 <a href="https://huggingface.co/Qwen/Qwen-Image-Edit">HuggingFace(Edit)</a> |
    🤖 <a href="https://modelscope.cn/models/Qwen/Qwen-Image">ModelScope-T2I</a> |
    🤖 <a href="https://modelscope.cn/models/Qwen/Qwen-Image-Edit">ModelScope-Edit</a> |
    📑 <a href="https://arxiv.org/abs/2508.02324">Tech Report</a> |
    📑 <a href="https://qwenlm.github.io/blog/qwen-image/">Blog(T2I)</a> |
    📑 <a href="https://qwenlm.github.io/blog/qwen-image-edit/">Blog(Edit)</a>
</p>

<p align="center">
    🖥️ <a href="https://huggingface.co/spaces/Qwen/Qwen-Image">T2I Demo</a> |
    🖥️ <a href="https://huggingface.co/spaces/Qwen/Qwen-Image-Edit">Edit Demo</a> |
    💬 <a href="https://github.com/QwenLM/Qwen-Image/blob/main/assets/wechat.png">WeChat (微信)</a> |
    🫨 <a href="https://discord.gg/CV4E9rpNSD">Discord</a>
</p>

<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/merge3.jpg" width="1024"/>
</p>

## Qwen-Image: Revolutionizing Image Generation and Editing

Qwen-Image is a cutting-edge, open-source image foundation model designed to generate stunning visuals from text prompts and offers advanced image editing capabilities.  Check out the original repo: [https://github.com/QwenLM/Qwen-Image](https://github.com/QwenLM/Qwen-Image).

**Key Features:**

*   **Text-to-Image Generation:** Create high-quality images from textual descriptions.
*   **Image Editing:**  Refine existing images with precise control over content and style.
*   **Exceptional Text Rendering:**  Achieves state-of-the-art performance in complex text rendering, especially for Chinese.
*   **Multi-Image Editing (Qwen-Image-Edit-2509):**  Edit multiple images in combination.
*   **Enhanced Consistency (Qwen-Image-Edit-2509):**  Improved identity preservation, product editing and text editing.
*   **ControlNet Support:** Native support for ControlNet for greater image manipulation control.
*   **LoRA Model Support:** Supports various LoRA models, e.g. MajicBeauty LoRA, for specialized image generation.
*   **Easy Integration:**  Quick start code snippets and comprehensive community support make it easy to use.
*   **Open Benchmarking:** AI Arena allows for dynamic environment for model evaluation,

## What's New

*   **[2025.09.22] Qwen-Image-Edit-2509 Release:** Major updates including multi-image editing support, enhanced single-image consistency and native ControlNet support.
    *   **Multi-image Editing:** Supports editing with multiple input images (1-3 images recommended).
    *   **Enhanced Consistency:** Improved person, product, and text editing.

*   **[2025.08.19] Update:** Recommended to update diffusers for improved performance.
*   **[2025.08.18] Qwen-Image-Edit Open-Sourced!**
*   **[2025.08.09] LoRA Support:**  Added support for LoRA models like MajicBeauty.
*   **[2025.08.05] ComfyUI Integration & Qwen Chat Integration:** Qwen-Image is now natively supported in ComfyUI and available on Qwen Chat.
*   **[2025.08.05] Technical Report Released:** Check out the [Technical Report](https://arxiv.org/abs/2508.02324) on arXiv!
*   **[2025.08.04] Qwen-Image Weights Released:** Available on [Hugging Face](https://huggingface.co/Qwen/Qwen-Image) and [ModelScope](https://modelscope.cn/models/Qwen/Qwen-Image).
*   **[2025.08.04] Qwen-Image Released:** Read the full details on the [Qwen-Image Blog](https://qwenlm.github.io/blog/qwen-image).

## Quick Start

### 1. Prerequisites
* Make sure your transformers>=4.51.3 (Supporting Qwen2.5-VL)
* Install the latest version of diffusers:

```bash
pip install git+https://github.com/huggingface/diffusers
```

### 2. Code Examples:

#### Qwen-Image (Text-to-Image)
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

#### Qwen-Image-Edit (Image Editing)
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
#### Qwen-Image-Edit-2509 (Multiple image Support)
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

## Advanced Usage

### Prompt Enhancement

*   **Text-to-Image:** Use the official Prompt Enhancement Tool powered by Qwen-Plus for improved optimization and multi-language support.
    ```python
    from tools.prompt_utils import rewrite
    prompt = rewrite(prompt)
    ```

*   **Image Edit:** Leverage Qwen-VL-Max for enhanced stability.

    ```python
    from tools.prompt_utils import polish_edit_prompt
    prompt = polish_edit_prompt(prompt, pil_image)
    ```

## Deploy Qwen-Image

Qwen-Image supports Multi-GPU API Server for local deployment.
### Multi-GPU API Server Pipeline & Usage
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

## Showcase

Showcases of Qwen-Image and Qwen-Image-Edit are available at the following links:
*   [Qwen-Image](./Qwen-Image.md)
*   [Qwen-Image-Edit](./Qwen-Image.md)
### Showcase of Qwen-Image Edit-2509

*   **Multi-Image Input:**  Examples of person + person, person + scene, and person + object edits.  Also includes support for ControlNet keypoint maps.
*   **Enhanced Consistency:** Showcases improved person consistency, including various portrait styles and pose changes. Further examples of product and text consistency, including font type, color and material editing.

## AI Arena

Qwen-Image is evaluated on the [AI Arena](https://aiarena.alibaba-inc.com), a benchmarking platform with the Elo rating system.  The platform provides a fair, transparent, and dynamic environment for model evaluation. See the latest leaderboard at [AI Arena Learboard](https://aiarena.alibaba-inc.com/corpora/arena/leaderboard?arenaType=text2image).

## Community Support

*   **Hugging Face:**  Diffusers support, LoRA and finetuning workflows in development.
*   **ModelScope:**
    *   [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) (Inference within 4GB VRAM, FP8 quantization, LoRA / full training)
    *   [DiffSynth-Engine](https://github.com/modelscope/DiffSynth-Engine) (FBCache-based acceleration, CFG parallel)
    *   [ModelScope AIGC Central](https://www.modelscope.cn/aigc)
*   **WaveSpeedAI:** [model page](https://wavespeed.ai/models/wavespeed-ai/qwen-image/text-to-image)
*   **LiblibAI:** [community](https://www.liblib.art/modelinfo/c62a103bd98a4246a2334e2d952f7b21?from=sd&versionUuid=75e0be0c93b34dd8baeec9c968013e0c)
*   **cache-dit:** [example](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline/run_qwen_image.py)

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

Join our [Discord](https://discord.gg/z3GAxXZ9Ce) or scan the QR code to connect via our [WeChat groups](assets/wechat.png).

We welcome your issues and pull requests on GitHub. Reach out to us at fulai.hr@alibaba-inc.com for full-time positions or research internships.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=QwenLM/Qwen-Image&type=Date)](https://www.star-history.com/#QwenLM/Qwen-Image&Date)