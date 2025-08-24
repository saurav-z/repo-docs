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

<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/merge3.jpg" width="1024"/>
</p>

## Qwen-Image: Unleash Your Creativity with Advanced Image Generation and Editing

**Qwen-Image** is a cutting-edge 20B MMDiT image foundation model offering unparalleled capabilities in text-to-image generation and image editing, providing exceptional performance. [Explore the Qwen-Image GitHub Repository](https://github.com/QwenLM/Qwen-Image) for comprehensive details.

**Key Features:**

*   **Exceptional Text Rendering:** Generate images with stunning accuracy in various languages, including English and Chinese, preserving typographic details and layout.
*   **Versatile Image Generation:** Create diverse visuals across a wide range of artistic styles, from photorealistic to anime.
*   **Advanced Image Editing:** Perform style transfer, object manipulation (insertion/removal), detail enhancement, and text editing within images.
*   **Image Understanding Capabilities:** Supports object detection, semantic segmentation, and other image understanding tasks.
*   **Image Editing Tutorial and Examples:** Includes detailed guidance on how to leverage Qwen-Image-Edit for a diverse range of editing applications.
*   **Quick Start Guide:** Provides a straightforward guide for installing the necessary dependencies and generating images with the models.
*   **Multi-GPU API Server:** Supports multi-GPU deployment with a Gradio web interface for parallel processing, queue management, and prompt optimization.

### News

*   **[2025.08.19]:** Performance adjustments have been made to the Qwen-Image-Edit, users are encouraged to upgrade to the latest diffusers commit for better results.
*   **[2025.08.18]:** Qwen-Image-Edit has been open-sourced! Experience it on [Qwen Chat](https://chat.qwen.ai/) or [Huggingface Demo](https://huggingface.co/spaces/Qwen/Qwen-Image-Edit).
*   **[2025.08.09]:** Qwen-Image now supports LoRA models like MajicBeauty LoRA, generating more realistic images. See [ModelScope](https://modelscope.cn/models/merjic/majicbeauty-qwen1/summary).
*   **[2025.08.05]:** Native ComfyUI support and availability on Qwen Chat. [Technical Report](https://arxiv.org/abs/2508.02324) released.
*   **[2025.08.04]:** Qwen-Image weights released on [Huggingface](https://huggingface.co/Qwen/Qwen-Image) and [ModelScope](https://modelscope.cn/models/Qwen/Qwen-Image). [Blog](https://qwenlm.github.io/blog/qwen-image) released.

> [!NOTE]
> For online demo access, consider DashScope, WaveSpeed, and LibLib. Links are in the Community Support section.

## Quick Start

1.  Ensure `transformers>=4.51.3`.
2.  Install the latest `diffusers`:
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
> Utilize prompt rewriting for more stable editing. See the official [demo script](src/examples/edit_demo.py). Qwen-Image-Edit is constantly evolving.

## Show Cases

*(Content from Original README)*

### General Cases
*(Content from Original README)*

### Tutorial for Image Editing
*(Content from Original README)*

### Advanced Usage

#### Prompt Enhance

For better prompt optimization use our Prompt Enhancement Tool powered by Qwen-Plus .

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

## AI Arena

For comprehensive evaluation using the Elo rating system, use [AI Arena](https://aiarena.alibaba-inc.com).

![AI Arena](assets/figure_aiarena_website.png)

Latest rankings are at [AI Arena Learboard](https://aiarena.alibaba-inc.com/corpora/arena/leaderboard?arenaType=text2image).

Contact weiyue.wy@alibaba-inc.com to deploy your model.

## Community Support

### Huggingface

*   Diffusers support is integrated. LoRA and finetuning are coming soon.

### ModelScope

*   **[DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)**: Supports Qwen-Image, including low-GPU-memory, FP8 quantization, and LoRA training.
*   **[DiffSynth-Engine](https://github.com/modelscope/DiffSynth-Engine)**: Optimized inference and deployment.
*   **[ModelScope AIGC Central](https://www.modelscope.cn/aigc)**: Hands-on experiences.

### WaveSpeedAI

*   [Wavespeed AI model page](https://wavespeed.ai/models/wavespeed-ai/qwen-image/text-to-image).

### LiblibAI

*   [LiblibAI community](https://www.liblib.art/modelinfo/c62a103bd98a4246a2334e2d952f7b21?from=sd&versionUuid=75e0be0c93b34dd8baeec9c968013e0c).

### Inference Acceleration Method: cache-dit

*   [cache-dit example](https://github.com/vipshop/cache-dit/blob/main/examples/run_qwen_image.py).

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

Contribute via issues and pull requests on GitHub. For full-time and intern positions, contact fulai.hr@alibaba-inc.com.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=QwenLM/Qwen-Image&type=Date)](https://www.star-history.com/#QwenLM/Qwen-Image&Date)