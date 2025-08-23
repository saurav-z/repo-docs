<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/qwen_image_logo.png" width="400" alt="Qwen-Image Logo"/>
</p>

<p align="center">
    &nbsp&nbsp💜 <a href="https://chat.qwen.ai/">Qwen Chat</a>&nbsp&nbsp |
    &nbsp&nbsp🤗 <a href="https://huggingface.co/Qwen/Qwen-Image">HuggingFace(T2I)</a>&nbsp&nbsp |
    &nbsp&nbsp🤗 <a href="https://huggingface.co/Qwen/Qwen-Image-Edit">HuggingFace(Edit)</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/models/Qwen/Qwen-Image">ModelScope-T2I</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/models/Qwen/Qwen-Image-Edit">ModelScope-Edit</a>&nbsp&nbsp| &nbsp&nbsp 📑 <a href="https://arxiv.org/abs/2508.02324">Tech Report</a> &nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://qwenlm.github.io/blog/qwen-image/">Blog(T2I)</a> &nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://qwenlm.github.io/blog/qwen-image-edit/">Blog(Edit)</a> &nbsp&nbsp
<br>
    🖥️ <a href="https://huggingface.co/spaces/Qwen/Qwen-Image">T2I Demo</a>&nbsp&nbsp | 🖥️ <a href="https://huggingface.co/spaces/Qwen/Qwen-Image-Edit">Edit Demo</a>&nbsp&nbsp | &nbsp&nbsp💬 <a href="https://github.com/QwenLM/Qwen-Image/blob/main/assets/wechat.png">WeChat (微信)</a>&nbsp&nbsp | &nbsp&nbsp🫨 <a href="https://discord.gg/CV4E9rpNSD">Discord</a>&nbsp&nbsp
</p>

# Qwen-Image: Unleash the Power of Advanced Text-to-Image Generation and Editing

**Qwen-Image is a cutting-edge 20B MMDiT model that excels at generating stunning images from text prompts and provides powerful image editing capabilities.** ([Original Repository](https://github.com/QwenLM/Qwen-Image))

## Key Features

*   **Exceptional Text Rendering:** Achieve unparalleled accuracy in rendering complex text within images, including support for both English and Chinese.
*   **Versatile Image Generation:** Generate diverse and high-quality images across various artistic styles, including photorealistic, anime, and more.
*   **Advanced Image Editing:** Edit images with precision, including style transfer, object manipulation (insertion, removal), detail enhancement, and text editing.
*   **Image Understanding Capabilities:** Leverage built-in features for object detection, semantic segmentation, and other intelligent image manipulation tasks.
*   **Seamless Integration:** Integrate easily with existing tools and platforms, including ComfyUI and Qwen Chat.
*   **Open Source & Accessible:** Benefit from open-source availability and easy access through Hugging Face and ModelScope.

## Quick Start

### Installation

1.  Ensure you have `transformers>=4.51.3` installed (supports Qwen2.5-VL).
2.  Install the latest version of `diffusers`:
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

## Show Cases

**(Images and detailed descriptions from the original README are retained here. Due to length constraints, these descriptions are summarized. For complete details, see the original README.)**

### General Cases

*   **High-Fidelity Text Rendering:**  Demonstrates Qwen-Image's ability to accurately render text in various languages and styles.
    **(Image: s1.jpg)**
*   **Diverse Image Generation:** Showcases the model's versatility across various artistic styles.
    **(Image: s2.jpg)**
*   **Advanced Image Editing:**  Highlights capabilities such as style transfer, object manipulation, and text editing.
    **(Image: s3.jpg)**
*   **Image Understanding Tasks:**  Illustrates support for tasks such as object detection and semantic segmentation.
    **(Image: s4.jpg)**

### Tutorial for Image Editing

**(This section details the tutorial and examples for image editing, with the descriptions from the original README retained. Due to length, the tutorial can't be replicated completely.)**

## Advanced Usage

### Prompt Enhancement

For enhanced prompt optimization and multi-language support, use our Prompt Enhancement Tool powered by Qwen-Plus:
```python
from tools.prompt_utils import rewrite
prompt = rewrite(prompt)
```

Alternatively, from the command line:
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

Evaluate Qwen-Image's general image generation capabilities and compare it with state-of-the-art APIs on the open benchmarking platform, [AI Arena](https://aiarena.alibaba-inc.com). The latest leaderboard rankings can be viewed at [AI Arena Learboard](https://aiarena.alibaba-inc.com/corpora/arena/leaderboard?arenaType=text2image).

Contact weiyue.wy@alibaba-inc.com to deploy your model.

## Community Support

*   **Hugging Face:**  Full support in diffusers.
*   **ModelScope:** Comprehensive support, including low-GPU-memory offload, FP8 quantization, and LoRA/full training (DiffSynth-Studio, DiffSynth-Engine, ModelScope AIGC Central).
*   **WaveSpeedAI:**  Deployment on their platform.
*   **LiblibAI:** Native support.
*   **cache-dit:** Cache acceleration support.

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

Connect with our research team and the community:

*   [Discord](https://discord.gg/z3GAxXZ9Ce)
*   [WeChat groups](assets/wechat.png)

Contribute with issues and pull requests.

Full-time positions and research internships are available; contact fulai.hr@alibaba-inc.com.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=QwenLM/Qwen-Image&type=Date)](https://www.star-history.com/#QwenLM/Qwen-Image&Date)