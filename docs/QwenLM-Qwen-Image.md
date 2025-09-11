<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/qwen_image_logo.png" width="400" alt="Qwen-Image Logo"/>
</p>
<p align="center">
  Unleash your creativity with **Qwen-Image**, a powerful image generation and editing model, built for complex text rendering and precise edits.  
  <br/>
  <br/>
  💜 <a href="https://chat.qwen.ai/">Qwen Chat</a>&nbsp;&nbsp; | &nbsp;&nbsp;🤗 <a href="https://huggingface.co/Qwen/Qwen-Image">HuggingFace (T2I)</a>&nbsp;&nbsp; | &nbsp;&nbsp;🤗 <a href="https://huggingface.co/Qwen/Qwen-Image-Edit">HuggingFace (Edit)</a>&nbsp;&nbsp; | &nbsp;&nbsp;🤖 <a href="https://modelscope.cn/models/Qwen/Qwen-Image">ModelScope-T2I</a>&nbsp;&nbsp; | &nbsp;&nbsp;🤖 <a href="https://modelscope.cn/models/Qwen/Qwen-Image-Edit">ModelScope-Edit</a>&nbsp;&nbsp; | &nbsp;&nbsp;📑 <a href="https://arxiv.org/abs/2508.02324">Tech Report</a> &nbsp;&nbsp; | &nbsp;&nbsp;📑 <a href="https://qwenlm.github.io/blog/qwen-image/">Blog (T2I)</a> &nbsp;&nbsp; | &nbsp;&nbsp;📑 <a href="https://qwenlm.github.io/blog/qwen-image-edit/">Blog (Edit)</a>
  <br/>
  🖥️ <a href="https://huggingface.co/spaces/Qwen/Qwen-Image">T2I Demo</a>&nbsp;&nbsp; | &nbsp;&nbsp;🖥️ <a href="https://huggingface.co/spaces/Qwen/Qwen-Image-Edit">Edit Demo</a>&nbsp;&nbsp; | &nbsp;&nbsp;💬 <a href="https://github.com/QwenLM/Qwen-Image/blob/main/assets/wechat.png">WeChat (微信)</a>&nbsp;&nbsp; | &nbsp;&nbsp;🫨 <a href="https://discord.gg/CV4E9rpNSD">Discord</a>
</p>
<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/merge3.jpg" width="1024" alt="Generated Image"/>
</p>

## Key Features

*   **Exceptional Text Rendering:** Generate images with impressive accuracy in rendering complex text, including Chinese and English.
*   **Precise Image Editing:**  Easily edit images with style transfer, object manipulation, text modifications, and more.
*   **Diverse Artistic Styles:** Supports various styles, from photorealistic to artistic, catering to various creative needs.
*   **Multi-GPU API Server:** Deploy locally with the provided server for parallel processing, queue management, and aspect ratio support.
*   **Prompt Enhancement Tools:** Integrate tools like Qwen-Plus and Qwen-VL-Max for enhanced prompt optimization and improved editing stability.

## Introduction

**Qwen-Image** is a state-of-the-art 20B MMDiT image foundation model designed for advanced image generation and editing tasks.  It excels in generating high-quality images with complex text and enables users to perform precise edits.  Explore its capabilities for both image generation and editing, with exceptional performance in text rendering, especially for Chinese.

## What's New

*   **2025.08.19:**  Qwen-Image-Edit performance improvements; update to the latest diffusers commit for optimal results.
*   **2025.08.18:**  Qwen-Image-Edit open-sourced!  Try it out in the [Hugging Face Demo](https://huggingface.co/spaces/Qwen/Qwen-Image-Edit).  Star the repo if you enjoy our work!
*   **2025.08.09:** Support for LoRA models such as MajicBeauty LoRA for generating realistic beauty images ([ModelScope](https://modelscope.cn/models/merjic/majicbeauty-qwen1/summary)).
*   **2025.08.05:** Native support in ComfyUI ([Qwen-Image in ComfyUI](https://blog.comfy.org/p/qwen-image-in-comfyui-new-era-of)).  Available on [Qwen Chat](https://chat.qwen.ai/) with "Image Generation".
*   **2025.08.05:** Released the [Technical Report](https://arxiv.org/abs/2508.02324) on Arxiv.
*   **2025.08.04:** Qwen-Image weights released!  Check them out on [Hugging Face](https://huggingface.co/Qwen/Qwen-Image) and [ModelScope](https://modelscope.cn/models/Qwen/Qwen-Image)!
*   **2025.08.04:**  Qwen-Image released! Read more details in the [Blog](https://qwenlm.github.io/blog/qwen-image).

> [!NOTE]
>  Due to heavy traffic, online demo options are available on DashScope, WaveSpeed, and LibLib (links in Community Support).

## Quick Start

### Prerequisites

*   `transformers>=4.51.3` (Supporting Qwen2.5-VL)
*   Latest version of `diffusers` (Install using the command below)

```bash
pip install git+https://github.com/huggingface/diffusers
```

### Text-to-Image

Generate images from text prompts.

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

### Image Editing

Edit images using text prompts.

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
>  Enhance editing stability using prompt rewriting. See the [demo script](src/examples/tools/prompt_utils.py) or Advanced Usage below.  Qwen-Image-Edit is under active development.

### Advanced Usage

#### Prompt Enhancement for Text-to-Image

Improve prompt optimization and multi-language support using the official Prompt Enhancement Tool (powered by Qwen-Plus).

Integrate it directly into your code:

```python
from tools.prompt_utils import rewrite
prompt = rewrite(prompt)
```

Alternatively, run the example script from the command line:

```bash
cd src
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx python examples/generate_w_prompt_enhance.py
```

#### Prompt Enhancement for Image Edit

Enhance stability using the official Prompt Enhancement Tool (powered by Qwen-VL-Max).

Integrate it directly into your code:

```python
from tools.prompt_utils import polish_edit_prompt
prompt = polish_edit_prompt(prompt, pil_image)
```

## Deploy Qwen-Image

Supports Multi-GPU API Server for local deployment.

### Multi-GPU API Server Pipeline & Usage

*   **Features:** Multi-GPU parallel processing, Queue management for high concurrency, Automatic prompt optimization, Support for multiple aspect ratios.

**Configuration:**

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

## Show Cases

### General Cases

Qwen-Image excels in creating high-fidelity text rendering across various images. It precisely integrates text into images, preserving typographic details, layout, and context.

<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/s1.jpg#center" alt="Text Rendering Example"/>
</p>

It supports various artistic styles.

<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/s2.jpg#center" alt="Artistic Style Example"/>
</p>

The image editing capabilities of Qwen-Image enable advanced operations such as style transfer, object insertion/removal, and human pose manipulation.

<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/s3.jpg#center" alt="Image Editing Example"/>
</p>

It understands images and supports tasks like object detection, semantic segmentation, and super-resolution.

<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/s4.jpg#center" alt="Image Understanding Example"/>
</p>

### Tutorial for Image Editing

Learn more about image editing capabilities.

[See the detailed explanation and examples of Image Editing](https://github.com/QwenLM/Qwen-Image#tutorial-for-image-editing) (See original README for detailed images and explanation)

## AI Arena

Qwen-Image is evaluated using [AI Arena](https://aiarena.alibaba-inc.com), an open benchmarking platform.

<p align="center">
    <img src="assets/figure_aiarena_website.png" alt="AI Arena Leaderboard" />
</p>

Find the latest leaderboard rankings at [AI Arena Learboard](https://aiarena.alibaba-inc.com/corpora/arena/leaderboard?arenaType=text2image).

Contact weiyue.wy@alibaba-inc.com to deploy your model on AI Arena.

## Community Support

Find support from:

*   **Hugging Face:** Diffusers support, LoRA, and fine-tuning development.
*   **ModelScope:** DiffSynth-Studio, DiffSynth-Engine, and ModelScope AIGC Central for hands-on experiences.
*   **WaveSpeedAI:** Deployment on their platform. ([model page](https://wavespeed.ai/models/wavespeed-ai/qwen-image/text-to-image))
*   **LiblibAI:** Native support. ([community](https://www.liblib.art/modelinfo/c62a103bd98a4246a2334e2d952f7b21?from=sd&versionUuid=75e0be0c93b34dd8baeec9c968013e0c))
*   **cache-dit:** Cache acceleration support. ([example](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline/run_qwen_image.py))

## License Agreement

Qwen-Image is licensed under Apache 2.0.

## Citation

Please cite our work:

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

Join our community and contribute!

*   **Discord:** [https://discord.gg/CV4E9rpNSD](https://discord.gg/CV4E9rpNSD)
*   **WeChat:** Scan the QR code (located in the original README)
*   **GitHub:**  Issues and pull requests are welcome!  [Qwen-Image GitHub](https://github.com/QwenLM/Qwen-Image)
*   **Hiring:** FTEs and Research Interns - Contact fulai.hr@alibaba-inc.com
    
## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=QwenLM/Qwen-Image&type=Date)](https://www.star-history.com/#QwenLM/Qwen-Image&Date)
```
Key improvements and explanations:

*   **SEO Optimization:**  Keywords like "image generation", "image editing", "text rendering", "foundation model" are incorporated naturally. The use of descriptive section headings also aids SEO.
*   **One-Sentence Hook:**  A compelling opening sentence to grab attention.
*   **Clear Structure:** Uses headings, bullet points, and clear formatting for readability and easy skimming.
*   **Summarization:** Condenses the original text while retaining essential information.
*   **Emphasis on Key Features:** The most important aspects are highlighted.
*   **Call to Action:** Encourages users to try the demos, join the community, and contribute.
*   **Direct Links:** Includes links to relevant resources and demos.
*   **Images and Alt Text:** Includes `alt` text for all images, and incorporates their use into the narrative.
*   **Up-to-date information:** the information from the original README about updates is included.
*   **Conciseness:** Removes unnecessary fluff and focuses on essential information.
*   **Improved Style:**  Uses a more professional and engaging tone.
*   **Corrected Markdown:** Fixes any rendering issues in the original README.