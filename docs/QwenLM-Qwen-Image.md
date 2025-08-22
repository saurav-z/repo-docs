<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/qwen_image_logo.png" width="400"/>
</p>

<p align="center">
    💜 <a href="https://chat.qwen.ai/">Qwen Chat</a> |
    🤗 <a href="https://huggingface.co/Qwen/Qwen-Image">HuggingFace (T2I)</a> |
    🤗 <a href="https://huggingface.co/Qwen/Qwen-Image-Edit">HuggingFace (Edit)</a> |
    🤖 <a href="https://modelscope.cn/models/Qwen/Qwen-Image">ModelScope-T2I</a> |
    🤖 <a href="https://modelscope.cn/models/Qwen/Qwen-Image-Edit">ModelScope-Edit</a> |
    📑 <a href="https://arxiv.org/abs/2508.02324">Tech Report</a> |
    📑 <a href="https://qwenlm.github.io/blog/qwen-image/">Blog (T2I)</a> |
    📑 <a href="https://qwenlm.github.io/blog/qwen-image-edit/">Blog (Edit)</a>
    <br>
    🖥️ <a href="https://huggingface.co/spaces/Qwen/Qwen-Image">T2I Demo</a> |
    🖥️ <a href="https://huggingface.co/spaces/Qwen/Qwen-Image-Edit">Edit Demo</a> |
    💬 <a href="https://github.com/QwenLM/Qwen-Image/blob/main/assets/wechat.png">WeChat (微信)</a> |
    🫨 <a href="https://discord.gg/CV4E9rpNSD">Discord</a>
</p>

## Qwen-Image: Unleash Your Imagination with Advanced Image Generation and Editing

**Qwen-Image is a powerful 20B MMDiT image foundation model enabling cutting-edge image generation, complex text rendering, and precise image editing capabilities. Explore the original repository on [GitHub](https://github.com/QwenLM/Qwen-Image) for the latest updates and resources.**

### Key Features

*   **Advanced Text Rendering:** Achieve high-fidelity text rendering across diverse languages, including English and Chinese, maintaining typographic details and contextual harmony.
*   **Versatile Image Generation:** Generate images in a wide range of artistic styles, from photorealistic to anime, adapting fluidly to creative prompts.
*   **Precise Image Editing:** Perform advanced image editing tasks, including style transfer, object manipulation, detail enhancement, and text editing within images.
*   **Image Understanding Capabilities:** Utilize built-in image understanding tasks such as object detection, semantic segmentation, and super-resolution, expanding your creative control.
*   **Qwen-Image-Edit:** Offers powerful semantic and appearance editing capabilities.

### What's New

*   **2025.08.19:** Performance improvements expected, especially in identity preservation and instruction following.
*   **2025.08.18:** Qwen-Image-Edit open-sourced!
*   **2025.08.09:** Support for LoRA models like MajicBeauty LoRA for generating realistic beauty images.
*   **2025.08.05:** Native support in ComfyUI and availability on Qwen Chat.
*   **2025.08.05:** Technical Report released on Arxiv!
*   **2025.08.04:** Qwen-Image weights released on Hugging Face and ModelScope.
*   **2025.08.04:** Qwen-Image released.

### Quick Start

1.  **Requirements:** `transformers>=4.51.3` and the latest version of `diffusers`.
2.  **Install diffusers:**
    ```bash
    pip install git+https://github.com/huggingface/diffusers
    ```

#### Text to Image
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
#### Image Editing
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
### Show Cases

The model's capabilities include:
*   High-fidelity text rendering across diverse images.
*   Support for a wide range of artistic styles.
*   Advanced image editing with semantic and appearance editing.
*   Image understanding tasks (object detection, semantic segmentation, etc.).

### Tutorial for Image Editing

Qwen-Image-Edit provides powerful capabilities for semantic and appearance editing.
![Capibara](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片3.JPG#center)
![MBTI meme series](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片4.JPG#center)
![Viewpoint transformation 90 degrees](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片12.JPG#center)
![Viewpoint transformation 180 degrees](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片13.JPG#center)
![Style transfer](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片1.JPG#center)
![Adding a signboard](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片6.JPG#center)
![Removing fine strands of hair](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片7.JPG#center)
![Modifying text color](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片8.JPG#center)
![Modifying backgrounds](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片11.JPG#center)
![Modifying clothing](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片5.JPG#center)
![Editing English text 1](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片15.JPG#center)
![Editing English text 2](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片16.JPG#center)
![Editing Chinese posters](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片17.JPG#center)
![Calligraphy artwork](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片18.JPG#center)
![Correcting characters](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片19.JPG#center)
![Fine-tuning character](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片20.JPG#center)
![Final version 1](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片21.JPG#center)
![Final version 2](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片22.JPG#center)
![Final version 3](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片23.JPG#center)
![Final version 4](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片24.JPG#center)
![Final version 5](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片25.JPG#center)

### Advanced Usage

#### Prompt Enhance
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


### AI Arena

Evaluate your model's performance and compare it with others on [AI Arena](https://aiarena.alibaba-inc.com).
![AI Arena](assets/figure_aiarena_website.png)
The latest leaderboard rankings can be viewed at [AI Arena Learboard](https://aiarena.alibaba-inc.com/corpora/arena/leaderboard?arenaType=text2image).

### Community Support

*   **Hugging Face:** Diffusers supports Qwen-Image.
*   **ModelScope:** Comprehensive support, including low-GPU-memory offload, FP8 quantization, and LoRA training.
*   **WaveSpeedAI:** Deployed Qwen-Image on their platform.
*   **LiblibAI:** Native support.
*   **cache-dit:** Offers cache acceleration support.

### License and Citation

Qwen-Image is licensed under Apache 2.0.  Please cite our work if you find it useful:
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

Connect with the research team on [Discord](https://discord.gg/z3GAxXZ9Ce) or via [WeChat](assets/wechat.png). Contribute via issues and pull requests on GitHub. Hiring for FTEs and research interns: fulai.hr@alibaba-inc.com.

### Star History
[![Star History Chart](https://api.star-history.com/svg?repos=QwenLM/Qwen-Image&type=Date)](https://www.star-history.com/#QwenLM/Qwen-Image&Date)