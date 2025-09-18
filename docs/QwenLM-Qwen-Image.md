<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/qwen_image_logo.png" width="400"/>
<p> 

<p align="center">
    &nbsp&nbsp💜 <a href="https://chat.qwen.ai/">Qwen Chat</a>&nbsp&nbsp |
    &nbsp&nbsp🤗 <a href="https://huggingface.co/Qwen/Qwen-Image">HuggingFace(T2I)</a>&nbsp&nbsp |
    &nbsp&nbsp🤗 <a href="https://huggingface.co/Qwen/Qwen-Image-Edit">HuggingFace(Edit)</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/models/Qwen/Qwen-Image">ModelScope-T2I</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/models/Qwen/Qwen-Image-Edit">ModelScope-Edit</a>&nbsp&nbsp| &nbsp&nbsp 📑 <a href="https://arxiv.org/abs/2508.02324">Tech Report</a> &nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://qwenlm.github.io/blog/qwen-image/">Blog(T2I)</a> &nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://qwenlm.github.io/blog/qwen-image-edit/">Blog(Edit)</a> &nbsp&nbsp 
    <br>
    🖥️ <a href="https://huggingface.co/spaces/Qwen/Qwen-Image">T2I Demo</a>&nbsp&nbsp | 🖥️ <a href="https://huggingface.co/spaces/Qwen/Qwen-Image-Edit">Edit Demo</a>&nbsp&nbsp | &nbsp&nbsp💬 <a href="https://github.com/QwenLM/Qwen-Image/blob/main/assets/wechat.png">WeChat (微信)</a>&nbsp&nbsp | &nbsp&nbsp🫨 <a href="https://discord.gg/CV4E9rpNSD">Discord</a>&nbsp&nbsp
</p>

<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/merge3.jpg" width="1024"/>
<p>

# Qwen-Image: Revolutionizing Image Generation and Editing

**Qwen-Image is a powerful 20B parameter foundation model, excelling in complex text rendering and precise image editing, offering unparalleled control over visual content, and more at [the original repository](https://github.com/QwenLM/Qwen-Image).**

## Key Features

*   **Superior Text Rendering:** Generate images with incredibly accurate and detailed text, especially for Chinese.
*   **Precise Image Editing:** Edit images with advanced features like style transfer, object manipulation, and text editing.
*   **Versatile Image Generation:** Create diverse images across various artistic styles, from photorealistic to stylized.
*   **Advanced Image Understanding:** Supports object detection, semantic segmentation, and more for intelligent image manipulation.
*   **Multi-Platform Support:** Available on Hugging Face, ModelScope, and integrated into ComfyUI and Qwen Chat.
*   **Open-Source & Accessible:** Released under Apache 2.0 license, with comprehensive documentation and community support.
*   **Image Editing Capabilities:**
    *   **Semantic Editing:** Modify image content while preserving original visual semantics, allowing for effortless original IP content creation.
    *   **Appearance Editing:** Add, remove, or modify specific elements while keeping certain regions unchanged.
    *   **Text Editing:** Accurate text editing capabilities.
    *   **Examples:** Modifying backgrounds and clothing.
    *   **Chained Editing:** Allows for the progressive correction of errors.

## Quickstart

### Prerequisites

*   `transformers>=4.51.3`
*   Latest version of `diffusers`

### Installation

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

## Advanced Usage

### Prompt Enhancement

*   **Text-to-Image:** Utilize the Prompt Enhancement Tool powered by Qwen-Plus.
    ```python
    from tools.prompt_utils import rewrite
    prompt = rewrite(prompt)
    ```
    Or run from the command line:
    ```bash
    cd src
    DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx python examples/generate_w_prompt_enhance.py
    ```

*   **Image Edit:** Leverage the Prompt Enhancement Tool powered by Qwen-VL-Max for enhanced stability.
    ```python
    from tools.prompt_utils import polish_edit_prompt
    prompt = polish_edit_prompt(prompt, pil_image)
    ```

## Deploy Qwen-Image

### Multi-GPU API Server

*   Supports Gradio-based web interface.
*   Features multi-GPU parallel processing, queue management, and automatic prompt optimization.

### Configuration

```bash
export NUM_GPUS_TO_USE=4          # Number of GPUs to use
export TASK_QUEUE_SIZE=100        # Task queue size
export TASK_TIMEOUT=300           # Task timeout in seconds
```

### Run the Server

```bash
# Start the gradio demo server, api key for prompt enhance
cd src
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxx python examples/demo.py
```

## Showcases

### General Cases

Qwen-Image showcases high-fidelity text rendering, support for diverse artistic styles and versatile image editing.

### Tutorial for Image Editing

*   Illustrates how to modify images while preserving original visual semantics.
*   Covers editing scenarios such as:
    *   Semantic editing
    *   Appearance editing
    *   Text editing

## AI Arena

*   Open benchmarking platform ([AI Arena](https://aiarena.alibaba-inc.com)) built on the Elo rating system.
*   Compare Qwen-Image with other models.
*   View the latest leaderboard rankings at [AI Arena Learboard](https://aiarena.alibaba-inc.com/corpora/arena/leaderboard?arenaType=text2image).

## Community Support

*   **Hugging Face:** Diffusers has supported Qwen-Image since day 0. LoRA and fine-tuning workflows are in development.
*   **ModelScope:** Extensive support with offload, quantization, and LoRA training.
*   **WaveSpeedAI:** Deployed Qwen-Image on their platform.
*   **LiblibAI:** Native support for Qwen-Image.
*   **cache-dit:** Acceleration support.

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

*   Join our [Discord](https://discord.gg/z3GAxXZ9Ce) or [WeChat groups](assets/wechat.png).
*   Report issues, submit pull requests on GitHub.
*   For full-time or internship opportunities, contact fulai.hr@alibaba-inc.com.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=QwenLM/Qwen-Image&type=Date)](https://www.star-history.com/#QwenLM/Qwen-Image&Date)