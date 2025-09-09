<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/qwen_image_logo.png" width="400"/>
<p> 

## Qwen-Image: Unleash Stunning Visuals with Advanced Image Generation and Editing 

**Qwen-Image is a powerful 20B parameter image foundation model designed for superior text rendering and precise image editing, pushing the boundaries of visual creativity.**  Explore the original repository on [GitHub](https://github.com/QwenLM/Qwen-Image).

<p align="center">&nbsp&nbsp💜 <a href="https://chat.qwen.ai/">Qwen Chat</a>&nbsp&nbsp |
           &nbsp&nbsp🤗 <a href="https://huggingface.co/Qwen/Qwen-Image">HuggingFace(T2I)</a>&nbsp&nbsp |
           &nbsp&nbsp🤗 <a href="https://huggingface.co/Qwen/Qwen-Image-Edit">HuggingFace(Edit)</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/models/Qwen/Qwen-Image">ModelScope-T2I</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/models/Qwen/Qwen-Image-Edit">ModelScope-Edit</a>&nbsp&nbsp| &nbsp&nbsp 📑 <a href="https://arxiv.org/abs/2508.02324">Tech Report</a> &nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://qwenlm.github.io/blog/qwen-image/">Blog(T2I)</a> &nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://qwenlm.github.io/blog/qwen-image-edit/">Blog(Edit)</a> &nbsp&nbsp 
<br>
🖥️ <a href="https://huggingface.co/spaces/Qwen/Qwen-Image">T2I Demo</a>&nbsp&nbsp | 🖥️ <a href="https://huggingface.co/spaces/Qwen/Qwen-Image-Edit">Edit Demo</a>&nbsp&nbsp | &nbsp&nbsp💬 <a href="https://github.com/QwenLM/Qwen-Image/blob/main/assets/wechat.png">WeChat (微信)</a>&nbsp&nbsp | &nbsp&nbsp🫨 <a href="https://discord.gg/CV4E9rpNSD">Discord</a>&nbsp&nbsp
</p>

<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/merge3.jpg" width="1024"/>
<p>

## Key Features

*   **Exceptional Text Rendering:** Generate images with high-fidelity text, including support for Chinese and other languages, with impressive typographic details and contextual integration.
*   **Precise Image Editing:**  Edit images with advanced features such as style transfer, object manipulation, and text modification, offering professional-grade control.
*   **Versatile Image Generation:** Create diverse visual content, from photorealistic scenes to artistic styles (e.g., anime, impressionist), adapting fluidly to creative prompts.
*   **Image Understanding Capabilities:** Benefit from integrated object detection, semantic segmentation, and other intelligent image analysis tools.
*   **Qwen-Image-Edit:** Includes powerful semantic and appearance editing tools, demonstrated in the provided tutorial for diverse visual content creation.

## News

*   **2025.08.19:**  Update to the latest diffusers commit for improved Qwen-Image-Edit performance.
*   **2025.08.18:** Qwen-Image-Edit is open-sourced!
*   **2025.08.09:**  Supports LoRA models, such as MajicBeauty LoRA, for realistic beauty images.
*   **2025.08.05:** Native support in ComfyUI and Qwen Chat (Image Generation). Technical Report released on Arxiv.
*   **2025.08.04:** Qwen-Image weights released on Hugging Face and ModelScope, with a blog for more details.

> [!NOTE]
> For online demo experience, visit DashScope, WaveSpeed, and LibLib.

## Quick Start

### Prerequisites

1.  Ensure `transformers>=4.51.3` (for Qwen2.5-VL support)
2.  Install the latest version of diffusers:
    ```bash
    pip install git+https://github.com/huggingface/diffusers
    ```

### Text-to-Image

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
> For stable editing results, it's recommended to use prompt rewriting. Refer to the example script or advanced usage for guidance.

### Advanced Usage

#### Prompt Enhance for Text-to-Image
For enhanced prompt optimization and multi-language support, we recommend using our official Prompt Enhancement Tool powered by Qwen-Plus .

You can integrate it directly into your code:
```python
from tools.prompt_utils import rewrite
prompt = rewrite(prompt)
```

Alternatively, run the example script from the command line:

```bash
cd src
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx python examples/generate_w_prompt_enhance.py
```

#### Prompt Enhance for Image Edit
For enhanced stability, we recommend using our official Prompt Enhancement Tool powered by Qwen-VL-Max.

You can integrate it directly into your code:
```python
from tools.prompt_utils import polish_edit_prompt
prompt = polish_edit_prompt(prompt, pil_image)
```

## Deploy Qwen-Image

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

## Show Cases

### General Cases

*   **(General Generation):** Qwen-Image excels in general image generation with diverse artistic styles, from photorealistic to artistic.
*   **(Text Rendering):** Standout capability for high-fidelity text rendering in multiple languages.
*   **(Image Editing):** Offers advanced operations like style transfer and object manipulation.
*   **(Image Understanding):** Supports image analysis tasks such as object detection and segmentation.

### Tutorial for Image Editing

*   **Semantic Editing:** Modify image content while preserving original semantics (e.g., character consistency).
*   **Applications:** Create IP content, MBTI-themed emoji packs based on a mascot, and viewpoint transformations.
*   **Appearance Editing:** Add, remove, or modify specific elements.
*   **Text Editing:** Accurate text editing capabilities, including English and Chinese text manipulation, with a chained editing approach.

## AI Arena

AI Arena is an open benchmarking platform for evaluating image generation models, using the Elo rating system.  See the [AI Arena Leaderboard](https://aiarena.alibaba-inc.com/corpora/arena/leaderboard?arenaType=text2image).
Contact weiyue.wy@alibaba-inc.com to deploy your model on AI Arena.

## Community Support

*   **Hugging Face:**  Diffusers support from day one; LoRA and fine-tuning workflows in development.
*   **ModelScope:** Comprehensive support via DiffSynth-Studio, DiffSynth-Engine, and ModelScope AIGC Central.
*   **WaveSpeedAI:** Native support on their platform.
*   **LiblibAI:** Native support from day one.
*   **cache-dit:** Cache acceleration support for Qwen-Image.

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

For questions, feedback, or contributions, please create issues and pull requests on GitHub.

We are hiring! Contact fulai.hr@alibaba-inc.com for FTE and research intern opportunities.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=QwenLM/Qwen-Image&type=Date)](https://www.star-history.com/#QwenLM/Qwen-Image&Date)