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

## Qwen-Image: Unleash Your Creativity with Advanced Image Generation and Editing

Qwen-Image is a powerful 20B MMDiT image foundation model that excels in both generating and editing images with remarkable precision, especially when rendering complex text and intricate details.  Explore the [original repository](https://github.com/QwenLM/Qwen-Image) for more information and to contribute.

<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/merge3.jpg" width="1024"/>
<p>

### Key Features

*   **Text-to-Image Generation:**  Create stunning images from text prompts, with exceptional text rendering capabilities.
*   **Precise Image Editing:**  Perform advanced edits, including style transfer, object manipulation, and text editing.
*   **Chinese Language Mastery:**  Achieve outstanding performance in rendering Chinese text and generating scenes with Chinese elements.
*   **Diverse Artistic Styles:**  Generate images in various styles, from photorealistic to artistic, adapting to your creative vision.
*   **Image Understanding Capabilities:** Supports object detection, semantic segmentation, and other advanced image analysis tasks.

### What's New

*   **2025.08.19:** Update to the latest diffusers commit for improved Qwen-Image-Edit performance, focusing on identity preservation and instruction following.
*   **2025.08.18:** Qwen-Image-Edit is now open-sourced!  Try it out via [Qwen Chat](https://chat.qwen.ai/) or the [Hugging Face Demo](https://huggingface.co/spaces/Qwen/Qwen-Image-Edit).
*   **2025.08.09:** Support for LoRA models like MajicBeauty LoRA, enabling realistic beauty image generation (see [ModelScope](https://modelscope.cn/models/merjic/majicbeauty-qwen1/summary)).
*   **2025.08.05:** Native support in ComfyUI (see [ComfyUI blog](https://blog.comfy.org/p/qwen-image-in-comfyui-new-era-of)).  Available on [Qwen Chat](https://chat.qwen.ai/) for image generation.
*   **2025.08.05:** Technical Report released on [Arxiv](https://arxiv.org/abs/2508.02324).
*   **2025.08.04:** Qwen-Image weights released on [Hugging Face](https://huggingface.co/Qwen/Qwen-Image) and [ModelScope](https://modelscope.cn/models/Qwen/Qwen-Image), along with a detailed [Blog](https://qwenlm.github.io/blog/qwen-image).

> **Note:** Due to high traffic, consider using alternative demos on DashScope, WaveSpeed, and LibLib.  See the "Community Support" section below for links.

### Quick Start

1.  **Prerequisites:** Ensure you have `transformers>=4.51.3` (for Qwen2.5-VL support).

2.  **Install diffusers:**

    ```bash
    pip install git+https://github.com/huggingface/diffusers
    ```

#### Text-to-Image Example

```python
from diffusers import DiffusionPipeline
import torch

model_name = "Qwen/Qwen-Image"

# Load the pipeline (adjust device and dtype as needed)
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

> **Note:** Prompt rewriting is highly recommended for better editing results. See the [official demo script](src/examples/edit_demo.py) for examples.  Qwen-Image-Edit is actively being developed; stay tuned for updates!

### Show Cases

*   **(General Cases)**  High-fidelity text rendering, supports diverse artistic styles and general image generation, and advanced image editing capabilities.

    *   Image examples are provided showcasing these features.

*   **(Tutorial for Image Editing)**  Examples showcasing semantic and appearance editing capabilities, including:

    *   Modifying image content while preserving visual semantics.
    *   Using MBTI-themed emoji packs based on Qwen's mascot.
    *   Rotating objects, style transfer, object insertion/removal, and text editing.
    *   A step-by-step example of correcting errors in calligraphy artwork using chained editing.

*   **(Advanced Usage)**
    *   Prompt Enhancement
        ```python
        from tools.prompt_utils import rewrite
        prompt = rewrite(prompt)
        ```
        Alternatively, run the example script from the command line:
        ```bash
        cd src
        DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx python examples/generate_w_prompt_enhance.py
        ```
    *   Deploy Qwen-Image
    *   Multi-GPU API Server Pipeline & Usage
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

Evaluate Qwen-Image's performance on the open benchmarking platform [AI Arena](https://aiarena.alibaba-inc.com), built on the Elo rating system.

*   View the latest leaderboard rankings:  [AI Arena Leaderboard](https://aiarena.alibaba-inc.com/corpora/arena/leaderboard?arenaType=text2image).
*   Contact weiyue.wy@alibaba-inc.com to deploy your model.

### Community Support

*   **Hugging Face:** Day 0 support in Diffusers. LoRA and finetuning workflows are coming soon.
*   **ModelScope:** Extensive support including layer-by-layer offload, FP8 quantization, and LoRA/full training. See [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio),  [DiffSynth-Engine](https://github.com/modelscope/DiffSynth-Engine), and [ModelScope AIGC Central](https://www.modelscope.cn/aigc).
*   **WaveSpeedAI:**  Deployed Qwen-Image, visit their [model page](https://wavespeed.ai/models/wavespeed-ai/qwen-image/text-to-image).
*   **LiblibAI:**  Native support, visit their [community](https://www.liblib.art/modelinfo/c62a103bd98a4246a2334e2d952f7b21?from=sd&versionUuid=75e0be0c93b34dd8baeec9c968013e0c).
*   **cache-dit:** cache acceleration support, visit their [example](https://github.com/vipshop/cache-dit/blob/main/examples/run_qwen_image.py).

### License

Qwen-Image is licensed under the Apache 2.0 License.

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

*   Join our [Discord](https://discord.gg/z3GAxXZ9Ce) or use the QR code for our [WeChat groups](assets/wechat.png).
*   Report issues or submit pull requests on GitHub.
*   For full-time or research intern positions, contact fulai.hr@alibaba-inc.com.