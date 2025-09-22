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

# Qwen-Image: Unleash Creative Visuals with Advanced Image Generation and Editing

**Qwen-Image is a state-of-the-art image foundation model that empowers users with exceptional image generation, intricate text rendering, and precise image editing capabilities.**  Explore the future of visual content creation with [Qwen-Image on GitHub](https://github.com/QwenLM/Qwen-Image).

## Key Features

*   **Advanced Text Rendering:** Generate images with stunning accuracy, particularly for complex text and Chinese characters.
*   **Precise Image Editing:**  Edit images with advanced features such as style transfer, object manipulation, and pose adjustments.
*   **Versatile Generation:** Create a wide range of visual styles, from photorealistic scenes to artistic styles like anime.
*   **Image Understanding:** Supports image understanding tasks including object detection, semantic segmentation, and super-resolution.
*   **Comprehensive Support:** Compatible with Hugging Face Diffusers, ModelScope, and various community platforms.
*   **Multi-GPU API Server:**  Deploy Qwen-Image locally with an easy-to-use Gradio-based web interface.

## News

*   **[2025.08.19]**:  Update to the latest diffusers commit for Qwen-Image-Edit to ensure optimal results.
*   **[2025.08.18]**:  Qwen-Image-Edit is now open-sourced!
*   **[2025.08.09]**:  Supports LoRA models such as MajicBeauty LoRA for realistic beauty images.
*   **[2025.08.05]**:  Native support in ComfyUI and integration into Qwen Chat.
*   **[2025.08.05]**:  Released Technical Report on Arxiv.
*   **[2025.08.04]**:  Released Qwen-Image weights on Hugging Face and ModelScope.
*   **[2025.08.04]**:  Announced Qwen-Image release on the blog.

> [!NOTE]
> Due to heavy traffic, consider using DashScope, WaveSpeed, or LibLib for online demo access.  See Community Support below.

## Quick Start

### Prerequisites

*   `transformers>=4.51.3` (Supporting Qwen2.5-VL)
*   Latest version of `diffusers`

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

> [!NOTE]
> For stable editing results, we strongly recommend using prompt rewriting. Refer to the [demo script](src/examples/tools/prompt_utils.py) or Advanced Usage.

### Advanced Usage

#### Prompt Enhancement

*   **Text-to-Image:**  Use the prompt enhancement tool powered by Qwen-Plus for improved prompt optimization and multi-language support:
    ```python
    from tools.prompt_utils import rewrite
    prompt = rewrite(prompt)
    ```

    Or, run the example script:
    ```bash
    cd src
    DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx python examples/generate_w_prompt_enhance.py
    ```

*   **Image Edit:** Utilize the prompt enhancement tool powered by Qwen-VL-Max for enhanced stability:

    ```python
    from tools.prompt_utils import polish_edit_prompt
    prompt = polish_edit_prompt(prompt, pil_image)
    ```

## Deploy Qwen-Image

### Multi-GPU API Server

Qwen-Image supports a Multi-GPU API Server with a Gradio-based web interface.

*   **Features:** Multi-GPU parallel processing, queue management, automatic prompt optimization, and multiple aspect ratio support.

*   **Configuration:**

    ```bash
    export NUM_GPUS_TO_USE=4          # Number of GPUs to use
    export TASK_QUEUE_SIZE=100        # Task queue size
    export TASK_TIMEOUT=300           # Task timeout in seconds
    ```

*   **Run the demo server:**

    ```bash
    cd src
    DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxx python examples/demo.py
    ```

## Show Cases

### General Cases

Qwen-Image showcases high-fidelity text rendering and versatile image generation capabilities across a wide range of styles.

### Tutorial for Image Editing

Qwen-Image-Edit enables semantic and appearance editing, allowing modifications while preserving the original visual semantics.

## AI Arena

Evaluate model performance on the open benchmarking platform, [AI Arena](https://aiarena.alibaba-inc.com), using the Elo rating system.

View latest rankings at [AI Arena Learboard](https://aiarena.alibaba-inc.com/corpora/arena/leaderboard?arenaType=text2image).

## Community Support

*   **Hugging Face:** Day 0 support for Qwen-Image in Diffusers; LoRA and finetuning workflows coming soon.
*   **ModelScope:** Comprehensive support for Qwen-Image, including low-GPU-memory offload, FP8 quantization, and LoRA/full training.
*   **WaveSpeedAI:** Deployed Qwen-Image on their platform.
*   **LiblibAI:** Native support for Qwen-Image.
*   **cache-dit:** Cache acceleration support for Qwen-Image.

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

Connect with the research team on [Discord](https://discord.gg/z3GAxXZ9Ce) or via [WeChat groups](assets/wechat.png).

For questions, feedback, contributions, or job inquiries, reach out via GitHub issues/PRs or fulai.hr@alibaba-inc.com.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=QwenLM/Qwen-Image&type=Date)](https://www.star-history.com/#QwenLM/Qwen-Image&Date)
```
Key improvements and optimization notes:

*   **SEO Optimization:** The title, headings, and descriptions use keywords like "image generation," "image editing," "text rendering," and "foundation model."
*   **Concise Hook:**  A strong opening sentence summarizes the project's core value.
*   **Clear Structure:** Uses headings, bullet points, and concise language for readability.
*   **Actionable Content:**  Provides clear instructions for Quick Start and Advanced Usage.
*   **Comprehensive Overview:** Includes all essential links, resources, and information from the original README.
*   **Focus on Benefits:**  Highlights the advantages of using Qwen-Image.
*   **Up-to-date Information:**  Incorporates the latest news and updates.
*   **Call to Action:** Encourages contributions, collaboration, and job applications.
*   **Easy to understand** The instructions are clear to help users to deploy and explore the model
*   **Uses Bold** Important information such as titles are in Bold for clarity.