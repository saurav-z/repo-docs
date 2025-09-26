<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/qwen_image_logo.png" width="400" alt="Qwen-Image Logo"/>
<p> 

<p align="center">
    &nbsp&nbsp💜 <a href="https://chat.qwen.ai/">Qwen Chat</a>&nbsp&nbsp |
    &nbsp&nbsp🤗 <a href="https://huggingface.co/Qwen/Qwen-Image">HuggingFace(T2I)</a>&nbsp&nbsp |
    &nbsp&nbsp🤗 <a href="https://huggingface.co/Qwen/Qwen-Image-Edit">HuggingFace(Edit)</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/models/Qwen/Qwen-Image">ModelScope-T2I</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/models/Qwen/Qwen-Image-Edit">ModelScope-Edit</a>&nbsp&nbsp| &nbsp&nbsp 📑 <a href="https://arxiv.org/abs/2508.02324">Tech Report</a> &nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://qwenlm.github.io/blog/qwen-image/">Blog(T2I)</a> &nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://qwenlm.github.io/blog/qwen-image-edit/">Blog(Edit)</a> &nbsp&nbsp 
<br>
🖥️ <a href="https://huggingface.co/spaces/Qwen/Qwen-Image">T2I Demo</a>&nbsp&nbsp | 🖥️ <a href="https://huggingface.co/spaces/Qwen/Qwen-Image-Edit">Edit Demo</a>&nbsp&nbsp | &nbsp&nbsp💬 <a href="https://github.com/QwenLM/Qwen-Image/blob/main/assets/wechat.png">WeChat (微信)</a>&nbsp&nbsp | &nbsp&nbsp🫨 <a href="https://discord.gg/CV4E9rpNSD">Discord</a>&nbsp&nbsp
</p>
## Qwen-Image: Unleash Your Imagination with Advanced Image Generation and Editing

Qwen-Image is a powerful image generation and editing model that lets you create stunning visuals from text prompts and refine existing images with unparalleled precision; **explore its capabilities on the [original GitHub repository](https://github.com/QwenLM/Qwen-Image)!**

<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/merge3.jpg" width="1024" alt="Qwen-Image Example"/>
<p>

## Key Features

*   **Text-to-Image Generation:** Create high-quality images from detailed text descriptions, supporting English and Chinese prompts.
*   **Precise Image Editing:** Refine existing images with accuracy, including content, style, and text modifications.
*   **Advanced Text Rendering:** Achieve exceptional results in rendering complex text within images, particularly in Chinese.
*   **Multi-Image Editing Support:** Edit multiple images simultaneously, offering diverse composition possibilities.
*   **Enhanced Consistency:** Ensure superior quality and coherence in image edits, preserving identity and details.
*   **Native ControlNet Support:** Leverage ControlNet features for precise image manipulation through depth maps, edge maps, and more.
*   **LoRA Model Compatibility:** Compatible with various LoRA models, such as MajicBeauty LoRA, for diverse image styles.
*   **Comprehensive Integration:** Supported by popular platforms like ComfyUI, Hugging Face, ModelScope, and more.

## News and Updates

*   **Qwen-Image-Edit-2509 (September 2025):** This latest iteration introduces significant enhancements, including:
    *   Multi-Image Editing Support: Now supports editing multiple images, enabling complex compositions.
    *   Enhanced Single-Image Consistency: Improved identity preservation, support for various styles, and better text and product editing.
    *   Native ControlNet Support: Built-in compatibility for depth maps, edge maps, and keypoint maps.

*   **August 2025 Updates:**
    *   Open-sourcing of Qwen-Image-Edit.
    *   Support for LoRA models like MajicBeauty LoRA.
    *   Native support in ComfyUI.
    *   Release of the Technical Report on arXiv.
    *   Model weights released on Hugging Face and ModelScope.

## Quick Start

### 1. Prerequisites

*   Ensure you have `transformers>=4.51.3`.
*   Install the latest version of `diffusers`:

```bash
pip install git+https://github.com/huggingface/diffusers
```

### 2. Text-to-Image (Qwen-Image)

```python
from diffusers import DiffusionPipeline
import torch

model_name = "Qwen/Qwen-Image"

# Load the pipeline (select device and torch_dtype)
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

### 3. Image Editing (Qwen-Image-Edit)

*   **For Single Image Input**
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
*   **For Multi Image Input (Qwen-Image-Edit-2509 - Recommended)**

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

### 4. Advanced Usage

*   **Prompt Enhancement:** Use the Prompt Enhancement Tool (powered by Qwen-Plus) for enhanced prompt optimization and multi-language support. Integrate into your code or run the example script.
*   **Prompt Enhancement for Image Edit**:  Use the Prompt Enhancement Tool powered by Qwen-VL-Max for enhanced stability
*   **Reference:** [Demo script](src/examples/tools/prompt_utils.py) includes example system prompts.

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


## Showcase

### Showcase of Qwen-Image Edit-2509

**[Multi-Image Input Examples]**

*   Person + Person
    ![Person + Person](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2509/幻灯片19.JPG#center)
*   Person + Scene
    ![Person + Scene](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2509/幻灯片20.JPG#center)
*   Person + Object
    ![Person + Object](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2509/幻灯片21.JPG#center)
*   Keypoint Pose Change
    ![Keypoint Pose Change](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2509/幻灯片22.JPG#center)
*   Three Images 1
    ![Three Images 1](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2509/幻灯片23.JPG#center)
*   Three Images 2
    ![Three Images 2](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2509/幻灯片24.JPG#center)
*   Three Images 3
    ![Three Images 3](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2509/幻灯片25.JPG#center)

**[Enhanced Consistency Examples]**
*   Portrait Styles
    ![Portrait Styles](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2509/幻灯片1.JPG#center)
*   Pose Change with Identity
    ![Pose Change with Identity](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2509/幻灯片2.JPG#center)
*   Meme Generation
    ![Meme Generation](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2509/幻灯片3.JPG#center)
*   Long Text with Identity
    ![Long Text with Identity](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2509/幻灯片4.JPG#center)
*   Old Photo Restoration 1
    ![Old Photo Restoration 1](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2509/幻灯片17.JPG#center)
*   Old Photo Restoration 2
    ![Old Photo Restoration 2](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2509/幻灯片18.JPG#center)
*   Cartoon & Cultural Creation
    ![Cartoon & Cultural Creation](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2509/幻灯片15.JPG#center)
*   Product Poster
    ![Product Poster](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2509/幻灯片5.JPG#center)
*   Logo Generation
    ![Logo Generation](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2509/幻灯片16.JPG#center)
*   Font Type
    ![Font Type](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2509/幻灯片10.JPG#center)
*   Font Color
    ![Font Color](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2509/幻灯片11.JPG#center)
*   Font Material
    ![Font Material](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2509/幻灯片12.JPG#center)
*   Precise Text Editing 1
    ![Precise Text Editing 1](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2509/幻灯片13.JPG#center)
*   Precise Text Editing 2
    ![Precise Text Editing 2](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2509/幻灯片14.JPG#center)
*   Poster Editing
    ![Poster Editing](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2509/幻灯片6.JPG#center)
*   Keypoint Control
    ![Keypoint Control](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2509/幻灯片7.JPG#center)
*   Sketch Input 1
    ![Sketch Input 1](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2509/幻灯片8.JPG#center)
*   Sketch Input 2
    ![Sketch Input 2](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2509/幻灯片9.JPG#center)

## AI Arena

AI Arena is an open benchmarking platform built on the Elo rating system to evaluate the image generation capabilities of Qwen-Image and other state-of-the-art closed-source APIs.

![AI Arena](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/figure_aiarena_website.png)

*   **Leaderboard:**  View the latest rankings at [AI Arena Learboard](https://aiarena.alibaba-inc.com/corpora/arena/leaderboard?arenaType=text2image).
*   **Participation:** Contact weiyue.wy@alibaba-inc.com if you want to deploy your model and participate in the evaluation.

## Community Support

*   **Hugging Face:** Day 0 Diffusers support, LoRA and finetuning workflows in development.
*   **ModelScope:**
    *   DiffSynth-Studio: Comprehensive support including low-GPU-memory offload, FP8 quantization, and training support.
    *   DiffSynth-Engine: Advanced optimizations.
    *   ModelScope AIGC Central: Hands-on experience with image generation and LoRA training.
*   **WaveSpeedAI:**  Qwen-Image deployed on their platform.
*   **LiblibAI:** Native support for Qwen-Image.
*   **cache-dit:**  Cache acceleration support for Qwen-Image with DBCache, TaylorSeer and Cache CFG.

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

*   **Discord:** Join our [Discord](https://discord.gg/z3GAxXZ9Ce)
*   **WeChat:** Scan the QR code to connect via our [WeChat groups](assets/wechat.png)
*   **GitHub:** Issues and pull requests are welcome on GitHub.
*   **Careers:** Reach out to fulai.hr@alibaba-inc.com for full-time and internship opportunities.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=QwenLM/Qwen-Image&type=Date)](https://www.star-history.com/#QwenLM/Qwen-Image&Date)