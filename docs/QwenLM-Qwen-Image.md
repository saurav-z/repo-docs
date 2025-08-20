<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/qwen_image_logo.png" width="400"/>
</p>

<p align="center">
    &nbsp&nbsp💜 <a href="https://chat.qwen.ai/">Qwen Chat</a>&nbsp&nbsp |
    &nbsp&nbsp🤗 <a href="https://huggingface.co/Qwen/Qwen-Image">HuggingFace(T2I)</a>&nbsp&nbsp |
    &nbsp&nbsp🤗 <a href="https://huggingface.co/Qwen/Qwen-Image-Edit">HuggingFace(Edit)</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/models/Qwen/Qwen-Image">ModelScope-T2I</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/models/Qwen/Qwen-Image-Edit">ModelScope-Edit</a>&nbsp&nbsp| &nbsp&nbsp 📑 <a href="https://arxiv.org/abs/2508.02324">Tech Report</a> &nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://qwenlm.github.io/blog/qwen-image/">Blog(T2I)</a> &nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://qwenlm.github.io/blog/qwen-image-edit/">Blog(Edit)</a> &nbsp&nbsp 
    <br>
    🖥️ <a href="https://huggingface.co/spaces/Qwen/Qwen-Image">T2I Demo</a>&nbsp&nbsp | 🖥️ <a href="https://huggingface.co/spaces/Qwen/Qwen-Image-Edit">Edit Demo</a>&nbsp&nbsp | &nbsp&nbsp💬 <a href="https://github.com/QwenLM/Qwen-Image/blob/main/assets/wechat.png">WeChat (微信)</a>&nbsp&nbsp | &nbsp&nbsp🫨 <a href="https://discord.gg/CV4E9rpNSD">Discord</a>&nbsp&nbsp
</p>

<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/merge3.jpg" width="1024"/>
</p>

## Qwen-Image: Revolutionizing Image Generation and Editing

**Qwen-Image is a state-of-the-art, 20B parameter, image foundation model that excels in complex text rendering and precise image editing, offering exceptional capabilities for both image creation and modification.**  Explore the cutting edge of visual AI with Qwen-Image on [GitHub](https://github.com/QwenLM/Qwen-Image)!

**Key Features:**

*   **Exceptional Text Rendering:** Accurately renders text in various languages, including Chinese, maintaining typographic details and context.
*   **Versatile Image Generation:** Supports a wide range of artistic styles, from photorealistic to artistic, accommodating diverse creative prompts.
*   **Advanced Image Editing:** Enables style transfer, object manipulation (insertion/removal), and precise text editing within images.
*   **Image Understanding Capabilities:** Offers support for object detection, semantic segmentation, and other image analysis tasks.
*   **Image Editing Tutorial**: Included a detailed tutorial for chained editing to improve calligraphy artwork.

### News

*   **2025.08.19:** Update to the latest diffusers commit to ensure optimal Qwen-Image-Edit results.
*   **2025.08.18:** Qwen-Image-Edit is open-sourced! Explore the [Hugging Face demo](https://huggingface.co/spaces/Qwen/Qwen-Image-Edit) and contribute to the project by starring the repository.
*   **2025.08.09:**  Supports various LoRA models such as MajicBeauty LoRA on [ModelScope](https://modelscope.cn/models/merjic/majicbeauty-qwen1/summary).
*   **2025.08.05:** Native support in ComfyUI. [Read more](https://blog.comfy.org/p/qwen-image-in-comfyui-new-era-of) and also available on Qwen Chat.
*   **2025.08.05:** Technical Report [on Arxiv](https://arxiv.org/abs/2508.02324) is released.
*   **2025.08.04:** Qwen-Image weights released on [Hugging Face](https://huggingface.co/Qwen/Qwen-Image) and [ModelScope](https://modelscope.cn/models/Qwen/Qwen-Image)!
*   **2025.08.04:** Qwen-Image released! Check out the [Blog](https://qwenlm.github.io/blog/qwen-image) for more details.

> [!NOTE]
> Explore the online demo on DashScope, WaveSpeed, and LibLib for demo access.

## Quick Start

1.  Ensure `transformers>=4.51.3` (supporting Qwen2.5-VL)
2.  Install the latest `diffusers`:

```bash
pip install git+https://github.com/huggingface/diffusers
```

### Text to Image Example

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

> [!NOTE]
> Use prompt rewriting to stabilize editing. See [demo script](src/examples/edit_demo.py) for examples.

## Show Cases

**(Summarized from original, focusing on SEO and key benefits)**

*   **Text Rendering Mastery:** Superior text rendering across various scripts, seamlessly integrating text into images.
*   **Diverse Artistic Styles:** Generate images in a wide array of styles, perfect for artists and designers.
*   **Advanced Editing Capabilities:** Transform images with style transfer, object manipulation, and precise editing.
*   **Image Understanding:** Supports tasks like object detection and semantic segmentation for intelligent image manipulation.

**(Image and text description from original)**

*   **Semantic Editing:**  This model preserves original visual semantics.
    ![Capibara](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片3.JPG#center)
    (Edited image of a capybara mascot)
*   **MBTI Themed Emojis:**  Emojis made based on our capybara mascot
    ![MBTI meme series](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片4.JPG#center)
    (MBTI emoji series)
*   **Viewpoint Transformation:** It can rotate objects.
    ![Viewpoint transformation 90 degrees](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片12.JPG#center)
    ![Viewpoint transformation 180 degrees](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片13.JPG#center)
    (Rotate object examples)
*   **Style Transfer** Can transform portrait images into different artistic styles.
    ![Style transfer](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片1.JPG#center)
    (Portrait style transfer)
*   **Appearance Editing:**  This model supports the addition and removal of objects and elements, while preserving visual semantics.
    ![Adding a signboard](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片6.JPG#center)
    (Adding a signboard example)
*   **Remove unwanted fine hair strands.**
    ![Removing fine strands of hair](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片7.JPG#center)
    (Remove small object example)
*   **Modify text color.**
    ![Modifying text color](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片8.JPG#center)
    (Text Color Changing example)
*   **Modifying background and clothing examples.**
    ![Modifying backgrounds](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片11.JPG#center)
    ![Modifying clothing](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片5.JPG#center)
    (Modifying clothing and background examples)
*   **Precise Text Editing:**  Edit both English and Chinese text directly within images.
    ![Editing English text 1](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片15.JPG#center)
    ![Editing English text 2](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片16.JPG#center)
    (English Text Editing examples)
*   **Chinese Poster Editing:**  It can edit large headlines as well as small text elements.
    ![Editing Chinese posters](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片17.JPG#center)
    (Chinese poster editing)
*   **Chained editing of Calligraphy artwork:** A concrete image editing example.
    ![Calligraphy artwork](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片18.JPG#center)
    ![Correcting characters](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片19.JPG#center)
    ![Fine-tuning character](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片20.JPG#center)
    ![Final version 1](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片21.JPG#center)
    ![Final version 2](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片22.JPG#center)
    ![Final version 3](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片23.JPG#center)
    ![Final version 4](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片24.JPG#center)
    ![Final version 5](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片25.JPG#center)
    (Chained editing examples)

### Advanced Usage

#### Prompt Enhancement

Use the Prompt Enhancement Tool powered by Qwen-Plus for better results.

```python
from tools.prompt_utils import rewrite
prompt = rewrite(prompt)
```

Alternatively, run the example script:

```bash
cd src
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx python examples/generate_w_prompt_enhance.py
```

## Deploy Qwen-Image

Qwen-Image supports Multi-GPU API Server for local deployment:

### Multi-GPU API Server Pipeline & Usage

This server provides:

*   Multi-GPU parallel processing
*   Queue management for high concurrency
*   Automatic prompt optimization
*   Support for multiple aspect ratios

Configuration:

```bash
export NUM_GPUS_TO_USE=4          # Number of GPUs to use
export TASK_QUEUE_SIZE=100        # Task queue size
export TASK_TIMEOUT=300           # Task timeout in seconds
```

To run:

```bash
cd src
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxx python examples/demo.py
```

## AI Arena

[AI Arena](https://aiarena.alibaba-inc.com) offers open benchmarking using the Elo rating system.

![AI Arena](assets/figure_aiarena_website.png)

View the leaderboard at [AI Arena Learboard](https://aiarena.alibaba-inc.com/corpora/arena/leaderboard?arenaType=text2image).

Contact weiyue.wy@alibaba-inc.com to deploy your model.

## Community Support

*   **Hugging Face:** Supported since day 0. LoRA and finetuning workflows coming soon.
*   **ModelScope:** Comprehensive support for Qwen-Image including low-GPU-memory offload, FP8 quantization, LoRA/full training, and advanced inference optimizations.
*   **WaveSpeedAI:** Deployed on their platform. [Model page](https://wavespeed.ai/models/wavespeed-ai/qwen-image/text-to-image).
*   **LiblibAI:** Native support. [Community](https://www.liblib.art/modelinfo/c62a103bd98a4246a2334e2d952f7b21?from=sd&versionUuid=75e0be0c93b34dd8baeec9c968013e0c).
*   **cache-dit:** Offers cache acceleration with DBCache, TaylorSeer and Cache CFG.  [Example](https://github.com/vipshop/cache-dit/blob/main/examples/run_qwen_image.py).

## License Agreement

Licensed under Apache 2.0.

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

Join our [Discord](https://discord.gg/z3GAxXZ9Ce) or [WeChat groups](assets/wechat.png) for discussion and collaboration.

Contribute through issues and pull requests on GitHub.

For FTEs and research interns, contact fulai.hr@alibaba-inc.com

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=QwenLM/Qwen-Image&type=Date)](https://www.star-history.com/#QwenLM/Qwen-Image&Date)