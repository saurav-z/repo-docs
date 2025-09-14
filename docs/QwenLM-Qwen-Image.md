<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/qwen_image_logo.png" width="400" alt="Qwen-Image Logo"/>
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
<br>
    🖥️ <a href="https://huggingface.co/spaces/Qwen/Qwen-Image">T2I Demo</a> |
    🖥️ <a href="https://huggingface.co/spaces/Qwen/Qwen-Image-Edit">Edit Demo</a> |
    💬 <a href="https://github.com/QwenLM/Qwen-Image/blob/main/assets/wechat.png">WeChat (微信)</a> |
    🫨 <a href="https://discord.gg/CV4E9rpNSD">Discord</a>
</p>

<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/merge3.jpg" width="1024" alt="Qwen-Image Example"/>
</p>

# Qwen-Image: Unleash Your Creativity with Advanced Image Generation and Editing

**Qwen-Image is a powerful 20B parameter foundation model delivering state-of-the-art performance in text-to-image generation and precise image editing, including exceptional text rendering capabilities.**  Explore the original repository [here](https://github.com/QwenLM/Qwen-Image).

## Key Features

*   **Exceptional Text Rendering:** Achieve stunning accuracy in complex text rendering, particularly for Chinese characters.
*   **Versatile Image Generation:** Create diverse images in various styles, from photorealistic to artistic.
*   **Advanced Image Editing:** Perform complex edits like style transfer, object manipulation, and text modifications.
*   **Image Understanding:** Leverage integrated capabilities for object detection, segmentation, and other advanced image analysis tasks.
*   **Multi-GPU API Server:** Deploy your own instance with a Gradio-based web interface, queue management, and prompt optimization.
*   **Supports LoRA models:** Generate highly realistic beauty images with LoRA models, such as MajicBeauty LoRA.

## What's New

*   **2025.08.19:**  Updated Qwen-Image-Edit for improved performance and stability, especially in identity preservation and instruction following.
*   **2025.08.18:** Qwen-Image-Edit open-sourced!
*   **2025.08.09:** LoRA model support including MajicBeauty LoRA
*   **2025.08.05:** Native support in ComfyUI and on Qwen Chat.
*   **2025.08.05:** Technical Report released on Arxiv.
*   **2025.08.04:** Qwen-Image weights released on Hugging Face and ModelScope.
*   **2025.08.04:** Qwen-Image officially released, learn more on the blog.

>   [!NOTE]
>   Due to high traffic, explore the online demos on DashScope, WaveSpeed, and LibLib for a seamless experience.

## Quick Start

1.  **Prerequisites:** Ensure `transformers>=4.51.3` (supporting Qwen2.5-VL).
2.  **Install Diffusers:**

    ```bash
    pip install git+https://github.com/huggingface/diffusers
    ```

### Text-to-Image Example

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

### Image Editing Example

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

>   [!NOTE]
>   For enhanced editing stability, use prompt rewriting; reference the official [demo script](src/examples/tools/prompt_utils.py) or the Advanced Usage section below.

### Advanced Usage

#### Prompt Enhancement for Text-to-Image

Enhance your prompts using the official Prompt Enhancement Tool powered by Qwen-Plus.

```python
from tools.prompt_utils import rewrite
prompt = rewrite(prompt)
```

Or, run the example script:

```bash
cd src
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx python examples/generate_w_prompt_enhance.py
```

#### Prompt Enhancement for Image Edit

Improve editing stability with the Qwen-VL-Max powered Prompt Enhancement Tool.

```python
from tools.prompt_utils import polish_edit_prompt
prompt = polish_edit_prompt(prompt, pil_image)
```

## Deploy Qwen-Image

### Multi-GPU API Server Pipeline & Usage

Deploy your own API server for local image generation:

**Configuration:**

```bash
export NUM_GPUS_TO_USE=4          # Number of GPUs to use
export TASK_QUEUE_SIZE=100        # Task queue size
export TASK_TIMEOUT=300           # Task timeout in seconds
```

**Run the Gradio demo server:**

```bash
cd src
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxx python examples/demo.py
```

## Show Cases

**(Refer to the original README for visual examples)**

*   **General Cases:** Showcasing text rendering, diverse image styles, and image editing capabilities.
*   **Tutorial for Image Editing:**  Step-by-step guide demonstrating Qwen-Image-Edit's semantic and appearance editing features (MBTI emojis, viewpoint transformations, style transfer, object addition/removal, text editing, etc.).

## AI Arena

Evaluate Qwen-Image's performance on the [AI Arena](https://aiarena.alibaba-inc.com), an open benchmarking platform using the Elo rating system.  View the latest rankings at [AI Arena Learboard](https://aiarena.alibaba-inc.com/corpora/arena/leaderboard?arenaType=text2image).

## Community Support

*   **Hugging Face:** Day 0 support and LoRA/finetuning development.
*   **ModelScope:** Comprehensive support including offload, quantization, and training.
*   **WaveSpeedAI:** Platform deployment.
*   **LiblibAI:** Native support.
*   **cache-dit:**  Cache acceleration support.

## License

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

*   **Discord:** [Discord](https://discord.gg/z3GAxXZ9Ce)
*   **WeChat:** Scan the QR code.
*   **GitHub:** Issues and pull requests are welcome.
*   **Hiring:** FTEs and research interns, contact fulai.hr@alibaba-inc.com.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=QwenLM/Qwen-Image&type=Date)](https://www.star-history.com/#QwenLM/Qwen-Image&Date)