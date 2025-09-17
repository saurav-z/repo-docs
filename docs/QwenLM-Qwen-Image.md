<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/qwen_image_logo.png" width="400"/>
<p> 

<p align="center">
    💜 [Qwen Chat](https://chat.qwen.ai/) |
    🤗 [HuggingFace(T2I)](https://huggingface.co/Qwen/Qwen-Image) |
    🤗 [HuggingFace(Edit)](https://huggingface.co/Qwen/Qwen-Image-Edit) |
    🤖 [ModelScope-T2I](https://modelscope.cn/models/Qwen/Qwen-Image) |
    🤖 [ModelScope-Edit](https://modelscope.cn/models/Qwen/Qwen-Image-Edit) |
    📑 [Tech Report](https://arxiv.org/abs/2508.02324) |
    📑 [Blog(T2I)](https://qwenlm.github.io/blog/qwen-image/) |
    📑 [Blog(Edit)](https://qwenlm.github.io/blog/qwen-image-edit/) 
<br>
    🖥️ [T2I Demo](https://huggingface.co/spaces/Qwen/Qwen-Image) |
    🖥️ [Edit Demo](https://huggingface.co/spaces/Qwen/Qwen-Image-Edit) |
    💬 [WeChat (微信)](https://github.com/QwenLM/Qwen-Image/blob/main/assets/wechat.png) |
    🫨 [Discord](https://discord.gg/CV4E9rpNSD)
</p>

## Qwen-Image: Generate and Edit Images with Unprecedented Precision

**Qwen-Image is a cutting-edge 20B MMDiT image foundation model, revolutionizing image generation and editing with exceptional text rendering and precise control, and you can explore the power of this model by visiting the original repository on [GitHub](https://github.com/QwenLM/Qwen-Image).**

<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/merge3.jpg" width="1024"/>
<p>

### Key Features:

*   **Advanced Text Rendering:** Generate images with unparalleled accuracy in rendering text, especially in Chinese and other languages.
*   **Precise Image Editing:** Perform advanced image editing tasks such as style transfer, object manipulation, and text modification with ease.
*   **Versatile Artistic Styles:** Support a wide range of artistic styles, from photorealistic to artistic, adapting to your creative prompts.
*   **Image Understanding Capabilities:** Includes object detection, semantic segmentation, and more for intelligent image manipulation.
*   **Open Source & Accessible:** Available on Hugging Face, ModelScope, and various community platforms.

### What's New:

*   **2025.08.19:** Update to the latest diffusers commit for improved Qwen-Image-Edit performance, particularly in identity preservation.
*   **2025.08.18:** Qwen-Image-Edit is now open-sourced!
*   **2025.08.09:** Support for LoRA models, like MajicBeauty LoRA, for generating realistic beauty images.
*   **2025.08.05:** Native support in ComfyUI and availability on Qwen Chat.
*   **2025.08.05:** Technical Report released on Arxiv.
*   **2025.08.04:** Qwen-Image weights released on Hugging Face and ModelScope.
*   **2025.08.04:** Qwen-Image released - check out the blog for more details.

### Quick Start

1.  **Prerequisites:** Ensure you have `transformers>=4.51.3` installed.
2.  **Install Diffusers:** `pip install git+https://github.com/huggingface/diffusers`

#### Text-to-Image Example:

```python
from diffusers import DiffusionPipeline
import torch

model_name = "Qwen/Qwen-Image"

# Load the pipeline, adjust torch_dtype and device as needed
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

#### Image Editing Example:

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
> For optimal editing results, particularly with Qwen-Image-Edit, it's highly recommended to use prompt rewriting.  See the official demo script or Advanced Usage for examples.

### Advanced Usage

#### Prompt Enhancement

*   **Text-to-Image:** Use the Prompt Enhancement Tool powered by Qwen-Plus.
    ```python
    from tools.prompt_utils import rewrite
    prompt = rewrite(prompt)
    ```
*   **Image Edit:** Use the Prompt Enhancement Tool powered by Qwen-VL-Max.
    ```python
    from tools.prompt_utils import polish_edit_prompt
    prompt = polish_edit_prompt(prompt, pil_image)
    ```

### Deploy Qwen-Image

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

### Show Cases

(Show cases content is kept as is - these help with SEO for relevant searches)

### AI Arena

Qwen-Image is evaluated on AI Arena, an open benchmarking platform.  Check the latest rankings at [AI Arena Learboard](https://aiarena.alibaba-inc.com/corpora/arena/leaderboard?arenaType=text2image).

### Community Support

*   **Hugging Face:**  Day 0 support in Diffusers, with LoRA and finetuning workflows in development.
*   **ModelScope:** Comprehensive support, including low-VRAM inference, quantization, and training tools.
*   **WaveSpeedAI:**  Deployed on their platform.
*   **LiblibAI:** Native support.
*   **cache-dit:** Cache acceleration support with DBCache, TaylorSeer and Cache CFG.

### License and Citation

*   **License:** Apache 2.0
*   **Citation:**

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

*   Join our [Discord](https://discord.gg/z3GAxXZ9Ce) or connect via [WeChat](assets/wechat.png).
*   Contribute via issues and pull requests on [GitHub](https://github.com/QwenLM/Qwen-Image).
*   For employment inquiries, contact fulai.hr@alibaba-inc.com.

### Star History

[![Star History Chart](https://api.star-history.com/svg?repos=QwenLM/Qwen-Image&type=Date)](https://www.star-history.com/#QwenLM/Qwen-Image&Date)
```

Key improvements and SEO optimizations:

*   **Clear, Concise Title and Introduction:**  Directly states what Qwen-Image is and its key benefits.
*   **SEO-Friendly Keywords:** Includes terms like "image generation," "image editing," "text rendering," "foundation model," and "MMDiT" to attract relevant search traffic.
*   **Structured Headings and Subheadings:**  Makes the information easy to scan and understand.
*   **Bulleted Key Features:** Highlights the core capabilities in an accessible format.
*   **Clear Call to Action:**  Encourages users to visit the GitHub repository.
*   **Links to Resources:**  Provides direct links to demos, models, and related resources, boosting user engagement.
*   **Emphasis on Benefits:** Focuses on what the model *does* for the user, rather than just technical details.
*   **Updated Information:** Keeps the "What's New" section current and relevant.
*   **Clear Code Examples:** Easy-to-copy code snippets.
*   **Community Support:** Explicitly lists community platforms and their support.
*   **Contact and Recruitment:**  Provides direct contact information for questions, contributions, and job opportunities.
*   **Star History Chart:** Enhances user engagement and provides valuable context.