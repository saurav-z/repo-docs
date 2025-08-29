<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/qwen_image_logo.png" width="400"/>
<p> 
<p align="center">&nbsp&nbsp💜 <a href="https://chat.qwen.ai/">Qwen Chat</a>&nbsp&nbsp |
           &nbsp&nbsp🤗 <a href="https://huggingface.co/Qwen/Qwen-Image">HuggingFace(T2I)</a>&nbsp&nbsp |
           &nbsp&nbsp🤗 <a href="https://huggingface.co/Qwen/Qwen-Image-Edit">HuggingFace(Edit)</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/models/Qwen/Qwen-Image">ModelScope-T2I</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/models/Qwen/Qwen-Image-Edit">ModelScope-Edit</a>&nbsp&nbsp| &nbsp&nbsp 📑 <a href="https://arxiv.org/abs/2508.02324">Tech Report</a> &nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://qwenlm.github.io/blog/qwen-image/">Blog(T2I)</a> &nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://qwenlm.github.io/blog/qwen-image-edit/">Blog(Edit)</a> &nbsp&nbsp 
<br>
🖥️ <a href="https://huggingface.co/spaces/Qwen/Qwen-Image">T2I Demo</a>&nbsp&nbsp | 🖥️ <a href="https://huggingface.co/spaces/Qwen/Qwen-Image-Edit">Edit Demo</a>&nbsp&nbsp | &nbsp&nbsp💬 <a href="https://github.com/QwenLM/Qwen-Image/blob/main/assets/wechat.png">WeChat (微信)</a>&nbsp&nbsp | &nbsp&nbsp🫨 <a href="https://discord.gg/CV4E9rpNSD">Discord</a>&nbsp&nbsp
</p>

<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/merge3.jpg" width="1024"/>
<p>

# Qwen-Image: Unleash Your Creativity with Advanced Image Generation and Editing

**Qwen-Image is a cutting-edge 20B MMDiT model that empowers users with unparalleled control over image creation and manipulation.**  Explore the original repository on [GitHub](https://github.com/QwenLM/Qwen-Image) for the latest updates and contributions.

## Key Features

*   **Exceptional Text Rendering:** Generate images with highly accurate and detailed text, especially in Chinese.
*   **Versatile Image Generation:** Create diverse images in various styles, from photorealistic to artistic, anime, and minimalist designs.
*   **Advanced Image Editing:** Perform complex edits, including style transfer, object manipulation, and text editing within images.
*   **Image Understanding Capabilities:** Utilize built-in features for object detection, semantic segmentation, and more.
*   **Image Editing Tutorials:** In-depth tutorials demonstrate Qwen-Image-Edit's powerful editing capabilities for semantic and appearance editing.

## News

*   **2025.08.19:** Performance improvements with the latest diffusers commit for Qwen-Image-Edit, especially in identity preservation and instruction following.
*   **2025.08.18:** Qwen-Image-Edit open-sourced! 🎉 Experience the online demo at [Qwen Chat](https://chat.qwen.ai/) or the [Huggingface Demo](https://huggingface.co/spaces/Qwen/Qwen-Image-Edit).
*   **2025.08.09:** Support for LoRA models, such as MajicBeauty LoRA, available on [ModelScope](https://modelscope.cn/models/merjic/majicbeauty-qwen1/summary).
*   **2025.08.05:** Native support in ComfyUI, see [Qwen-Image in ComfyUI: New Era of Text Generation in Images!](https://blog.comfy.org/p/qwen-image-in-comfyui-new-era-of)
*   **2025.08.05:** Now on Qwen Chat. Click [Qwen Chat](https://chat.qwen.ai/) and choose "Image Generation".
*   **2025.08.05:** Released [Technical Report](https://arxiv.org/abs/2508.02324) on Arxiv.
*   **2025.08.04:** Qwen-Image weights released! Check [Huggingface](https://huggingface.co/Qwen/Qwen-Image) and [ModelScope](https://modelscope.cn/models/Qwen/Qwen-Image)!
*   **2025.08.04:** Qwen-Image released! Check our [Blog](https://qwenlm.github.io/blog/qwen-image) for more details!

## Quick Start

### Prerequisites
*   transformers>=4.51.3 (Supporting Qwen2.5-VL)
*   Latest version of diffusers
```bash
pip install git+https://github.com/huggingface/diffusers
```

### Text to Image

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

## Advanced Usage

### Prompt Enhancement
For enhanced prompt optimization and multi-language support, use the Prompt Enhancement Tool powered by Qwen-Plus.

*   Integrate it in your code:
    ```python
    from tools.prompt_utils import rewrite
    prompt = rewrite(prompt)
    ```
*   Or run the example script from the command line:

```bash
cd src
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx python examples/generate_w_prompt_enhance.py
```

### Prompt Enhancement for Image Edit
For enhanced stability in image editing, use the Prompt Enhancement Tool powered by Qwen-VL-Max.

*   Integrate it in your code:
    ```python
    from tools.prompt_utils import polish_edit_prompt
    prompt = polish_edit_prompt(prompt, pil_image)
    ```

## Deploy Qwen-Image

### Multi-GPU API Server

*   Set up Multi-GPU API Server for local deployment
*   Configuration via environment variables:

```bash
export NUM_GPUS_TO_USE=4          # Number of GPUs to use
export TASK_QUEUE_SIZE=100        # Task queue size
export TASK_TIMEOUT=300           # Task timeout in seconds
```

*   Start the Gradio demo server:

```bash
cd src
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxx python examples/demo.py 
```

## Show Cases

### General Cases

*   High-fidelity text rendering across different languages.
*   Supports a wide range of artistic styles.
*   Enables advanced editing operations.
*   Supports image understanding tasks, including object detection, semantic segmentation, and more.

### Tutorial for Image Editing
*   Demonstrates capabilities for semantic and appearance editing.

## AI Arena

*   [AI Arena](https://aiarena.alibaba-inc.com) for objective comparison with state-of-the-art closed-source APIs using the Elo rating system.
*   View latest leaderboard at [AI Arena Learboard](https://aiarena.alibaba-inc.com/corpora/arena/leaderboard?arenaType=text2image).
*   Contact weiyue.wy@alibaba-inc.com if you wish to deploy your model on AI Arena.

## Community Support

### Huggingface
*   Diffusers supports Qwen-Image. Support for LoRA and finetuning workflows is in development.

### ModelScope
*   Comprehensive support for Qwen-Image, including low-GPU-memory layer-by-layer offload, FP8 quantization, and LoRA / full training.
*   [ModelScope AIGC Central](https://www.modelscope.cn/aigc) provides hands-on experiences.

### WaveSpeedAI

*   WaveSpeed has deployed Qwen-Image on their platform, visit their [model page](https://wavespeed.ai/models/wavespeed-ai/qwen-image/text-to-image) for more details.

### LiblibAI

*   LiblibAI offers native support for Qwen-Image, visit their [community](https://www.liblib.art/modelinfo/c62a103bd98a4246a2334e2d952f7b21?from=sd&versionUuid=75e0be0c93b34dd8baeec9c968013e0c) page for more details and discussions.

### Inference Acceleration Method: cache-dit

*   cache-dit offers cache acceleration support for Qwen-Image with DBCache, TaylorSeer and Cache CFG. Visit their [example](https://github.com/vipshop/cache-dit/blob/main/examples/run_qwen_image.py) for more details.

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

*   Join our [Discord](https://discord.gg/z3GAxXZ9Ce) or scan the QR code to connect via our [WeChat groups](assets/wechat.png).
*   Welcome issues and pull requests on GitHub.
*   Reach out to fulai.hr@alibaba-inc.com for full-time or research intern opportunities.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=QwenLM/Qwen-Image&type=Date)](https://www.star-history.com/#QwenLM/Qwen-Image&Date)