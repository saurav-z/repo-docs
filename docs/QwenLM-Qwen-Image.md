<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/qwen_image_logo.png" width="400"/>
</p>

<p align="center">
  <a href="https://chat.qwen.ai/">Qwen Chat</a> |
  <a href="https://huggingface.co/Qwen/Qwen-Image">Hugging Face (T2I)</a> |
  <a href="https://huggingface.co/Qwen/Qwen-Image-Edit">Hugging Face (Edit)</a> |
  <a href="https://modelscope.cn/models/Qwen/Qwen-Image">ModelScope-T2I</a> |
  <a href="https://modelscope.cn/models/Qwen/Qwen-Image-Edit">ModelScope-Edit</a> |
  <a href="https://arxiv.org/abs/2508.02324">Tech Report</a> |
  <a href="https://qwenlm.github.io/blog/qwen-image/">Blog (T2I)</a> |
  <a href="https://qwenlm.github.io/blog/qwen-image-edit/">Blog (Edit)</a>
<br>
  <a href="https://huggingface.co/spaces/Qwen/Qwen-Image">T2I Demo</a> |
  <a href="https://huggingface.co/spaces/Qwen/Qwen-Image-Edit">Edit Demo</a> |
  <a href="https://github.com/QwenLM/Qwen-Image/blob/main/assets/wechat.png">WeChat</a> |
  <a href="https://discord.gg/CV4E9rpNSD">Discord</a>
</p>

<p align="center">
  <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/merge3.jpg" width="1024"/>
</p>

# Qwen-Image: Advanced Image Generation and Editing

**Qwen-Image is a powerful 20B MMDiT foundation model, revolutionizing image creation with exceptional text rendering, precise image editing, and impressive performance across diverse visual tasks.**  [Explore the original repository](https://github.com/QwenLM/Qwen-Image).

## Key Features

*   **Superior Text Rendering:** Generates images with remarkable accuracy in rendering complex text, especially in Chinese.
*   **Precise Image Editing:** Enables advanced editing capabilities, including style transfer, object manipulation, and detailed refinements.
*   **Versatile Image Generation:** Supports a wide range of artistic styles, from photorealistic to artistic, making it a versatile tool for various creative applications.
*   **Comprehensive Image Understanding:** Supports image understanding tasks like object detection, semantic segmentation, and depth estimation, enhancing editing capabilities.
*   **User-Friendly Deployment:** Offers a Multi-GPU API Server for easy local deployment and Gradio-based web interface.

## Quick Start

### 1. Installation
Make sure your transformers is the latest version (transformers>=4.51.3 (Supporting Qwen2.5-VL)).
```bash
pip install git+https://github.com/huggingface/diffusers
```

### 2. Text-to-Image

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

### Advanced Usage

*   **Prompt Enhancement:** Use the official Prompt Enhancement Tool powered by Qwen-Plus or Qwen-VL-Max to enhance text-to-image and image editing stability and results. Integrate the tools into your code or use the provided example scripts.

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

[Example images showcasing the model's capabilities can be found in the original README. You'll see high-fidelity text rendering, diverse artistic styles, advanced image editing features and image understanding capabilities]

## AI Arena

[The original README contains information on the AI Arena platform. You can view the leaderboards and participate in model evaluations.]

## Community Support

*   **Hugging Face:**  Supports Qwen-Image with LoRA and fine-tuning workflows in development.
*   **ModelScope:** Provides comprehensive support, including low-GPU-memory offload, FP8 quantization, and LoRA training.
*   **WaveSpeedAI:** Deploys Qwen-Image on their platform.
*   **LiblibAI:** Offers native support for Qwen-Image.
*   **cache-dit:** Offers cache acceleration support.

## License Agreement

Qwen-Image is licensed under Apache 2.0.

## Citation

If you find Qwen-Image useful, please cite our work:

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

Join our [Discord](https://discord.gg/z3GAxXZ9Ce) or scan the QR code to connect via our [WeChat groups](assets/wechat.png).  We welcome your feedback, issues, and pull requests on GitHub.  We're also hiring for full-time employees and research interns.  Contact fulai.hr@alibaba-inc.com

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=QwenLM/Qwen-Image&type=Date)](https://www.star-history.com/#QwenLM/Qwen-Image&Date)