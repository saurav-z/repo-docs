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

## Qwen-Image: Unleash Your Imagination with Advanced Image Generation and Editing

Qwen-Image is a powerful 20B MMDiT foundation model, revolutionizing image creation and editing with its cutting-edge capabilities.  Check out the original repository [here](https://github.com/QwenLM/Qwen-Image).

<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/merge3.jpg" width="1024"/>
<p>

### Key Features:

*   **Exceptional Text Rendering:** Generate images with accurate and detailed text, especially for Chinese characters.
*   **Versatile Image Generation:** Create images in diverse artistic styles, from photorealistic to artistic.
*   **Advanced Image Editing:** Perform complex edits, including style transfer, object manipulation, and more.
*   **Image Understanding Capabilities:** Utilize object detection, semantic segmentation, and other AI-powered features.
*   **Simplified Deployment:** Ready-to-use with Hugging Face, ModelScope, and other platforms.

### What's New
*   **2025.08.19:** Update to the latest diffusers commit for optimal Qwen-Image-Edit results.
*   **2025.08.18:** Qwen-Image-Edit open-sourced!
*   **2025.08.09:** LoRA models support (e.g., MajicBeauty LoRA)
*   **2025.08.05:** Native support in ComfyUI, Qwen-Image integration on Qwen Chat.
*   **2025.08.05:** Technical Report released.
*   **2025.08.04:** Qwen-Image weights released.
*   **2025.08.04:** Qwen-Image released.

### Quick Start

#### 1. Prerequisites:
   * Ensure you have transformers >= 4.51.3 (Supporting Qwen2.5-VL)
   * Install the latest version of diffusers:
    ```bash
    pip install git+https://github.com/huggingface/diffusers
    ```

#### 2. Text to Image

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

#### 3. Image Editing

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

#### 4. Advanced Usage

*   **Prompt Enhancement for Text-to-Image:** Use the prompt rewriting tool for optimal results:
    ```python
    from tools.prompt_utils import rewrite
    prompt = rewrite(prompt)
    ```
    Or run the example script:
    ```bash
    cd src
    DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx python examples/generate_w_prompt_enhance.py
    ```
*   **Prompt Enhance for Image Edit:**  Use Qwen-VL-Max to improve stability:
    ```python
    from tools.prompt_utils import polish_edit_prompt
    prompt = polish_edit_prompt(prompt, pil_image)
    ```

### Deploy Qwen-Image

Qwen-Image supports a Multi-GPU API Server for local deployment:

#### 1. Multi-GPU API Server Pipeline & Usage
   The Multi-GPU API Server starts a Gradio-based web interface with:
    *   Multi-GPU parallel processing
    *   Queue management for high concurrency
    *   Automatic prompt optimization
    *   Support for multiple aspect ratios
#### 2. Configuration via environment variables:

```bash
export NUM_GPUS_TO_USE=4          # Number of GPUs to use
export TASK_QUEUE_SIZE=100        # Task queue size
export TASK_TIMEOUT=300           # Task timeout in seconds
```
#### 3. Start the gradio demo server:
```bash
cd src
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxx python examples/demo.py 
```

### Show Cases

[The original readme contains this section - it is very good content for a showcase]

### AI Arena

Qwen-Image is evaluated on [AI Arena](https://aiarena.alibaba-inc.com), an open benchmarking platform.

![AI Arena](assets/figure_aiarena_website.png)

View the latest rankings at [AI Arena Learboard](https://aiarena.alibaba-inc.com/corpora/arena/leaderboard?arenaType=text2image).

Contact weiyue.wy@alibaba-inc.com to deploy your model.

### Community Support

*   **Hugging Face:** Supports Qwen-Image with LoRA and finetuning support in development.
*   **ModelScope:** Comprehensive support with offload, quantization, and training options.
*   **WaveSpeedAI:**  Platform deployment of Qwen-Image.
*   **LiblibAI:** Native support for Qwen-Image.
*   **cache-dit:** Cache acceleration support for Qwen-Image.

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

*   Join the [Discord](https://discord.gg/z3GAxXZ9Ce)
*   Connect via [WeChat](assets/wechat.png)
*   Submit issues and pull requests on GitHub.
*   Contact fulai.hr@alibaba-inc.com for job opportunities.

### Star History
[![Star History Chart](https://api.star-history.com/svg?repos=QwenLM/Qwen-Image&type=Date)](https://www.star-history.com/#QwenLM/Qwen-Image&Date)