<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/qwen_image_logo.png" width="400"/>
</p>

<p align="center">
    💜 <a href="https://chat.qwen.ai/">Qwen Chat</a>&nbsp;&nbsp; |
    🤗 <a href="https://huggingface.co/Qwen/Qwen-Image">HuggingFace(T2I)</a>&nbsp;&nbsp; |
    🤗 <a href="https://huggingface.co/Qwen/Qwen-Image-Edit">HuggingFace(Edit)</a>&nbsp;&nbsp; |
    🤖 <a href="https://modelscope.cn/models/Qwen/Qwen-Image">ModelScope-T2I</a>&nbsp;&nbsp; |
    🤖 <a href="https://modelscope.cn/models/Qwen/Qwen-Image-Edit">ModelScope-Edit</a>&nbsp;&nbsp; |
    📑 <a href="https://arxiv.org/abs/2508.02324">Tech Report</a>&nbsp;&nbsp; |
    📑 <a href="https://qwenlm.github.io/blog/qwen-image/">Blog(T2I)</a>&nbsp;&nbsp; |
    📑 <a href="https://qwenlm.github.io/blog/qwen-image-edit/">Blog(Edit)</a>&nbsp;&nbsp;
    <br>
    🖥️ <a href="https://huggingface.co/spaces/Qwen/Qwen-Image">T2I Demo</a>&nbsp;&nbsp; |
    🖥️ <a href="https://huggingface.co/spaces/Qwen/Qwen-Image-Edit">Edit Demo</a>&nbsp;&nbsp; |
    💬 <a href="https://github.com/QwenLM/Qwen-Image/blob/main/assets/wechat.png">WeChat (微信)</a>&nbsp;&nbsp; |
    🫨 <a href="https://discord.gg/CV4E9rpNSD">Discord</a>
</p>

# Qwen-Image: Revolutionizing Image Generation and Editing

**Qwen-Image, a 20B MMDiT image foundation model, empowers users with exceptional text rendering and precise image editing capabilities.**

[**Visit the original repository for more details**](https://github.com/QwenLM/Qwen-Image)

## Key Features:

*   **Exceptional Text Rendering:**  Generate images with high-fidelity text, preserving typographic details, layout, and contextual harmony, especially for Chinese characters.
*   **Advanced Image Editing:**  Perform style transfer, object manipulation (insertion/removal), and intricate edits with intuitive prompts.
*   **Diverse Artistic Styles:** Create images across a wide spectrum of styles, from photorealism to artistic interpretations.
*   **Comprehensive Image Understanding:** Supports object detection, semantic segmentation, and other image analysis tasks, enabling intelligent editing.
*   **Image Editing Tutorials:** Step-by-step guidance on semantic and appearance editing, demonstrating practical applications.

## News:

*   **2025.08.19:**  Update to the latest diffusers commit for improved Qwen-Image-Edit performance, especially in identity preservation and instruction following.
*   **2025.08.18:** Open-sourcing of Qwen-Image-Edit!  Try it out on [Qwen Chat](https://chat.qwen.ai/) or [Huggingface Demo](https://huggingface.co/spaces/Qwen/Qwen-Image-Edit)
*   **2025.08.09:**  Qwen-Image supports LoRA models, such as MajicBeauty LoRA. Check the available weights on [ModelScope](https://modelscope.cn/models/merjic/majicbeauty-qwen1/summary).
*   **2025.08.05:**  Native support for Qwen-Image in ComfyUI ([Qwen-Image in ComfyUI: New Era of Text Generation in Images!](https://blog.comfy.org/p/qwen-image-in-comfyui-new-era-of)).
*   **2025.08.05:**  Qwen-Image available on Qwen Chat ([Qwen Chat](https://chat.qwen.ai/), select "Image Generation").
*   **2025.08.05:** Technical Report released on Arxiv ([Technical Report](https://arxiv.org/abs/2508.02324)).
*   **2025.08.04:** Qwen-Image weights released ([Huggingface](https://huggingface.co/Qwen/Qwen-Image) and [ModelScope](https://modelscope.cn/models/Qwen/Qwen-Image)).
*   **2025.08.04:**  Qwen-Image released! Check our [Blog](https://qwenlm.github.io/blog/qwen-image) for details.

> [!NOTE]
> Due to heavy traffic, explore the online demos on DashScope, WaveSpeed, and LibLib. Links are provided in the Community Support section.

## Quick Start

### 1. Prerequisites

*   Ensure `transformers>=4.51.3` (Supports Qwen2.5-VL).
*   Install the latest version of diffusers:

```bash
pip install git+https://github.com/huggingface/diffusers
```

### 2. Text-to-Image Generation
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
> [!NOTE]
>  Prompt rewriting is highly recommended for stable editing results.  See the [demo script](src/examples/tools/prompt_utils.py) or Advanced Usage below for examples.

## Advanced Usage

### Prompt Enhancement for Text-to-Image

Use the Qwen-Plus powered Prompt Enhancement Tool:
```python
from tools.prompt_utils import rewrite
prompt = rewrite(prompt)
```

Or run the example script:

```bash
cd src
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx python examples/generate_w_prompt_enhance.py
```

### Prompt Enhancement for Image Edit

Use the Qwen-VL-Max powered Prompt Enhancement Tool:
```python
from tools.prompt_utils import polish_edit_prompt
prompt = polish_edit_prompt(prompt, pil_image)
```

## Deploy Qwen-Image

Qwen-Image supports Multi-GPU API Server for local deployment:

### Multi-GPU API Server Pipeline & Usage

Features:

*   Multi-GPU parallel processing
*   Queue management for high concurrency
*   Automatic prompt optimization
*   Support for multiple aspect ratios

Configuration:
```bash
export NUM_GPUS_TO_USE=4          # Number of GPUs to use
export TASK_QUEUE_SIZE=100        # Task queue size
export TASK_TIMEOUT=300           # Task timeout in seconds
```

Run the Gradio demo server:
```bash
cd src
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxx python examples/demo.py 
```

## Show Cases

### General Cases

[Showcases of the different capabilities of Qwen-Image]

### Tutorial for Image Editing

[Detailed tutorial with examples about image editing]

## AI Arena

Qwen-Image is evaluated on the [AI Arena](https://aiarena.alibaba-inc.com) platform. The latest leaderboard rankings can be viewed at [AI Arena Learboard](https://aiarena.alibaba-inc.com/corpora/arena/leaderboard?arenaType=text2image).

Contact weiyue.wy@alibaba-inc.com if you wish to deploy your model and participate in the evaluation.

## Community Support

*   **Hugging Face:**  Diffusers support; LoRA and finetuning development in progress.
*   **ModelScope:**  Comprehensive support including layer-by-layer offload, quantization, and training ([DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio), [DiffSynth-Engine](https://github.com/modelscope/DiffSynth-Engine), [ModelScope AIGC Central](https://www.modelscope.cn/aigc)).
*   **WaveSpeedAI:** Native support. See their [model page](https://wavespeed.ai/models/wavespeed-ai/qwen-image/text-to-image).
*   **LiblibAI:** Native support. See their [community](https://www.liblib.art/modelinfo/c62a103bd98a4246a2334e2d952f7b21?from=sd&versionUuid=75e0be0c93b34dd8baeec9c968013e0c).
*   **cache-dit:** Cache acceleration support.  See their [example](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline/run_qwen_image.py).

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

Connect with our research team via [Discord](https://discord.gg/z3GAxXZ9Ce) or scan the WeChat QR code.  Contribute via issues and pull requests.  We are hiring at fulai.hr@alibaba-inc.com.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=QwenLM/Qwen-Image&type=Date)](https://www.star-history.com/#QwenLM/Qwen-Image&Date)