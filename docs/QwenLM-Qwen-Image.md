<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/qwen_image_logo.png" width="400"/>
</p>

<p align="center">
    💜 <a href="https://chat.qwen.ai/">Qwen Chat</a> | 🤗 <a href="https://huggingface.co/Qwen/Qwen-Image">HuggingFace(T2I)</a> | 🤗 <a href="https://huggingface.co/Qwen/Qwen-Image-Edit">HuggingFace(Edit)</a> | 🤖 <a href="https://modelscope.cn/models/Qwen/Qwen-Image">ModelScope-T2I</a> | 🤖 <a href="https://modelscope.cn/models/Qwen/Qwen-Image-Edit">ModelScope-Edit</a> | 📑 <a href="https://arxiv.org/abs/2508.02324">Tech Report</a> | 📑 <a href="https://qwenlm.github.io/blog/qwen-image/">Blog(T2I)</a> | 📑 <a href="https://qwenlm.github.io/blog/qwen-image-edit/">Blog(Edit)</a>
<br>
🖥️ <a href="https://huggingface.co/spaces/Qwen/Qwen-Image">T2I Demo</a> | 🖥️ <a href="https://huggingface.co/spaces/Qwen/Qwen-Image-Edit">Edit Demo</a> | 💬 <a href="https://github.com/QwenLM/Qwen-Image/blob/main/assets/wechat.png">WeChat (微信)</a> | 🫨 <a href="https://discord.gg/CV4E9rpNSD">Discord</a>
</p>

## Qwen-Image: Unleashing Advanced Image Generation and Editing

**Qwen-Image is a cutting-edge 20B MMDiT image foundation model, excelling in complex text rendering and precise image editing.  [Explore the original repository](https://github.com/QwenLM/Qwen-Image) to learn more!**

<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/merge3.jpg" width="1024"/>
</p>

### Key Features

*   **Exceptional Text Rendering:** Achieve high-fidelity text integration in diverse languages, including English and Chinese, preserving typographic details and contextual harmony.
*   **Versatile Image Generation:** Generate a wide range of artistic styles, from photorealistic scenes to anime aesthetics, adapting to diverse creative prompts.
*   **Advanced Image Editing:** Perform style transfer, object manipulation (insertion/removal), detail enhancement, and text editing within images with intuitive control.
*   **Image Understanding Capabilities:** Supports object detection, semantic segmentation, and other intelligent image editing tasks powered by deep visual comprehension.

### Key Updates
*   **2025.08.19:** Observed performance misalignments of Qwen-Image-Edit; update to the latest diffusers commit for optimal results.
*   **2025.08.18:**  Qwen-Image-Edit has been open-sourced! 🎉
*   **2025.08.09:**  Supports various LoRA models like MajicBeauty LoRA for generating realistic beauty images (ModelScope available).
*   **2025.08.05:**  Natively supported in ComfyUI.  Also available on Qwen Chat.
*   **2025.08.05:**  Technical Report released on Arxiv.
*   **2025.08.04:** Qwen-Image weights released on Huggingface and ModelScope.
*   **2025.08.04:** Qwen-Image released. Check our [Blog](https://qwenlm.github.io/blog/qwen-image) for more details!

### Quick Start

1.  **Prerequisites:** Ensure `transformers>=4.51.3` and the latest `diffusers` version.
2.  **Install Diffusers:**
    ```bash
    pip install git+https://github.com/huggingface/diffusers
    ```

#### Text-to-Image Example

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

#### Image Editing Example

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

### Advanced Usage

#### Prompt Enhancement
*   **Text-to-Image:**  Use the official Prompt Enhancement Tool powered by Qwen-Plus.
    ```python
    from tools.prompt_utils import rewrite
    prompt = rewrite(prompt)
    ```
*   **Image Editing:** For enhanced stability, use the Prompt Enhancement Tool powered by Qwen-VL-Max.
    ```python
    from tools.prompt_utils import polish_edit_prompt
    prompt = polish_edit_prompt(prompt, pil_image)
    ```

### Deploy Qwen-Image
Qwen-Image supports a Multi-GPU API Server for local deployment.  See the original repo for setup instructions.

### Show Cases

*   **General Cases:** Showcase images demonstrating text rendering, diverse artistic styles, image editing capabilities (style transfer, object manipulation), and image understanding tasks.
*   **Tutorial for Image Editing:** Detailed examples with the Capybara mascot, showcasing semantic editing, MBTI-themed emoji packs, novel view synthesis, style transfer, background/clothing modifications, and text editing.

### AI Arena

*   [AI Arena](https://aiarena.alibaba-inc.com) is an open benchmarking platform for evaluating Qwen-Image using the Elo rating system.
*   View the latest rankings on the [AI Arena Leaderboard](https://aiarena.alibaba-inc.com/corpora/arena/leaderboard?arenaType=text2image).

### Community Support

*   **Hugging Face:**  Diffusers support since day 0. LoRA and finetuning workflows in development.
*   **ModelScope:** Offers extensive support including low-VRAM inference, FP8 quantization, and training.
*   **WaveSpeedAI:** Deployed Qwen-Image on their platform.
*   **LiblibAI:** Native support.
*   **cache-dit:** Cache acceleration support with DBCache, TaylorSeer, and Cache CFG.

### License

Qwen-Image is licensed under Apache 2.0.

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

*   Join our [Discord](https://discord.gg/z3GAxXZ9Ce).
*   Connect via our [WeChat groups](assets/wechat.png) (scan QR code).
*   Submit issues and pull requests on GitHub.
*   Contact fulai.hr@alibaba-inc.com for FTE/internship opportunities.

### Star History

[![Star History Chart](https://api.star-history.com/svg?repos=QwenLM/Qwen-Image&type=Date)](https://www.star-history.com/#QwenLM/Qwen-Image&Date)