<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/qwen_image_logo.png" width="400" alt="Qwen-Image Logo"/>
<p>

<p align="center">
   <a href="https://chat.qwen.ai/">Qwen Chat</a>&nbsp;&nbsp; |
   <a href="https://huggingface.co/Qwen/Qwen-Image">HuggingFace (T2I)</a>&nbsp;&nbsp; |
   <a href="https://huggingface.co/Qwen/Qwen-Image-Edit">HuggingFace (Edit)</a>&nbsp;&nbsp; |
   <a href="https://modelscope.cn/models/Qwen/Qwen-Image">ModelScope-T2I</a>&nbsp;&nbsp; |
   <a href="https://modelscope.cn/models/Qwen/Qwen-Image-Edit">ModelScope-Edit</a>&nbsp;&nbsp; |
   <a href="https://arxiv.org/abs/2508.02324">Tech Report</a>&nbsp;&nbsp; |
   <a href="https://qwenlm.github.io/blog/qwen-image/">Blog (T2I)</a>&nbsp;&nbsp; |
   <a href="https://qwenlm.github.io/blog/qwen-image-edit/">Blog (Edit)</a>
<br>
   <a href="https://huggingface.co/spaces/Qwen/Qwen-Image">T2I Demo</a>&nbsp;&nbsp; |
   <a href="https://huggingface.co/spaces/Qwen/Qwen-Image-Edit">Edit Demo</a>&nbsp;&nbsp; |
   <a href="https://github.com/QwenLM/Qwen-Image/blob/main/assets/wechat.png">WeChat (微信)</a>&nbsp;&nbsp; |
   <a href="https://discord.gg/CV4E9rpNSD">Discord</a>
</p>

## Qwen-Image: Unleash Your Creative Vision with Advanced Image Generation and Editing

[Explore the Qwen-Image repository on GitHub](https://github.com/QwenLM/Qwen-Image) to unlock the power of this cutting-edge image foundation model.

<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/merge3.jpg" width="1024" alt="Qwen-Image Example"/>
<p>

### Key Features

*   **Text-to-Image Generation:** Transform text prompts into stunning visuals, with exceptional performance in complex text rendering, especially for Chinese.
*   **Image Editing:** Perform advanced editing operations such as style transfer, object manipulation, and text modification within images.
*   **Advanced Prompt Optimization:** Utilize tools for prompt enhancement and multi-language support.
*   **Multi-GPU API Server:** Deploy and manage Qwen-Image with a Gradio-based web interface, supporting multi-GPU parallel processing, queue management, and aspect ratio control.
*   **Integration with Popular Platforms:** Supported on Hugging Face, ModelScope, WaveSpeedAI, and LiblibAI.
*   **Image Understanding:** Supports various image understanding tasks including object detection and semantic segmentation.

### News

*   **2025.08.19:**  Qwen-Image-Edit performance improvements. Update to the latest diffusers commit for optimal results.
*   **2025.08.18:** Qwen-Image-Edit is open-sourced! Explore the online demos on [Qwen Chat](https://chat.qwen.ai/) and [Hugging Face](https://huggingface.co/spaces/Qwen/Qwen-Image-Edit).
*   **2025.08.09:**  Support for LoRA models, such as MajicBeauty LoRA, for generating realistic beauty images. Check out available weights on [ModelScope](https://modelscope.cn/models/merjic/majicbeauty-qwen1/summary).
*   **2025.08.05:**  Natively supported in ComfyUI and on Qwen Chat.
*   **2025.08.05:**  Technical Report released on Arxiv: [https://arxiv.org/abs/2508.02324](https://arxiv.org/abs/2508.02324).
*   **2025.08.04:**  Qwen-Image weights released! Check out [Hugging Face](https://huggingface.co/Qwen/Qwen-Image) and [ModelScope](https://modelscope.cn/models/Qwen/Qwen-Image).
*   **2025.08.04:**  Qwen-Image released! Read more details in the [Blog](https://qwenlm.github.io/blog/qwen-image).

> [!NOTE]
> Due to heavy traffic, online demos are also available on DashScope, WaveSpeed, and LibLib. Links in the Community Support section below.

### Quick Start

**1. Prerequisites:**

*   transformers>=4.51.3 (Supporting Qwen2.5-VL)
*   Latest version of diffusers

**2. Installation:**

```bash
pip install git+https://github.com/huggingface/diffusers
```

**3. Text-to-Image Example:**

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

**4. Image Editing Example:**

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
> To enhance editing stability, prompt rewriting is strongly recommended. See [demo script](src/examples/tools/prompt_utils.py) or the Advanced Usage section for examples. Qwen-Image-Edit is actively evolving.

### Advanced Usage

#### Prompt Enhancement

*   **Text-to-Image:** Use the Qwen-Plus powered Prompt Enhancement Tool:
    ```python
    from tools.prompt_utils import rewrite
    prompt = rewrite(prompt)
    ```
    Or, run the example script:  `cd src; DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx python examples/generate_w_prompt_enhance.py`
*   **Image Edit:** Use the Qwen-VL-Max powered Prompt Enhancement Tool:
    ```python
    from tools.prompt_utils import polish_edit_prompt
    prompt = polish_edit_prompt(prompt, pil_image)
    ```

### Deploy Qwen-Image

Qwen-Image supports Multi-GPU API Server for local deployment:

#### Multi-GPU API Server Pipeline & Usage

The Multi-GPU API Server will start a Gradio-based web interface with:
- Multi-GPU parallel processing
- Queue management for high concurrency
- Automatic prompt optimization
- Support for multiple aspect ratios

**Configuration:**

```bash
export NUM_GPUS_TO_USE=4          # Number of GPUs to use
export TASK_QUEUE_SIZE=100        # Task queue size
export TASK_TIMEOUT=300           # Task timeout in seconds
```

**Run the demo server:**

```bash
# Start the gradio demo server, api key for prompt enhance
cd src
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxx python examples/demo.py
```

### Show Cases

*   **General Image Generation:**  High-fidelity text rendering, diverse artistic styles, and support for a wide range of creative prompts.
*   **Image Editing:**  Advanced operations including style transfer, object manipulation, and text editing.
*   **Image Understanding:** Supports a suite of image understanding tasks, including object detection, semantic segmentation, depth and edge estimation, novel view synthesis, and super-resolution.

[Insert images from Show Cases section here.]

### AI Arena

Evaluate Qwen-Image's image generation capabilities on [AI Arena](https://aiarena.alibaba-inc.com), an open benchmarking platform.

[Insert image from AI Arena here.]

View the latest leaderboard rankings at [AI Arena Learboard](https://aiarena.alibaba-inc.com/corpora/arena/leaderboard?arenaType=text2image).

To deploy your model on AI Arena, contact weiyue.wy@alibaba-inc.com.

### Community Support

*   **Hugging Face:** Supported since day 0.
*   **ModelScope:** Comprehensive support including low-GPU-memory offload, quantization, and LoRA training.
*   **WaveSpeedAI:** Deployed on their platform from day 0.
*   **LiblibAI:** Native support from day 0.
*   **Inference Acceleration Method: cache-dit**: Offers cache acceleration support for Qwen-Image with DBCache, TaylorSeer and Cache CFG.

### License Agreement

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
*   Connect via our [WeChat groups](assets/wechat.png).
*   Contribute with issues and pull requests on GitHub.
*   For employment and research intern opportunities, contact fulai.hr@alibaba-inc.com.

### Star History

[![Star History Chart](https://api.star-history.com/svg?repos=QwenLM/Qwen-Image&type=Date)](https://www.star-history.com/#QwenLM/Qwen-Image&Date)