<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/qwen_image_logo.png" width="400" alt="Qwen-Image Logo"/>
</p>

<p align="center">
    Unleash your creativity with **Qwen-Image**, a state-of-the-art image generation and editing model from Qwen.
    <br>
    &nbsp&nbsp💜 <a href="https://chat.qwen.ai/">Qwen Chat</a>&nbsp&nbsp |
           &nbsp&nbsp🤗 <a href="https://huggingface.co/Qwen/Qwen-Image">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/models/Qwen/Qwen-Image">ModelScope</a>&nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://arxiv.org/abs/2508.02324">Tech Report</a> &nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://qwenlm.github.io/blog/qwen-image/">Blog</a> &nbsp&nbsp 
    <br>
    🖥️ <a href="https://huggingface.co/spaces/Qwen/Qwen-Image">Demo</a>&nbsp&nbsp | &nbsp&nbsp💬 <a href="https://github.com/QwenLM/Qwen-Image/blob/main/assets/wechat.png">WeChat (微信)</a>&nbsp&nbsp | &nbsp&nbsp🫨 <a href="https://discord.gg/CV4E9rpNSD">Discord</a>&nbsp&nbsp
</p>

<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/merge3.jpg" width="1024" alt="Qwen-Image Example"/>
</p>

## Key Features

*   **Exceptional Text Rendering:** Generate images with accurate and legible text, including complex Chinese characters and diverse fonts.
*   **Versatile Image Generation:** Create stunning images in various styles, from photorealistic to artistic, based on your prompts.
*   **Advanced Image Editing:** Perform sophisticated edits like style transfer, object manipulation, and pose adjustments.
*   **Comprehensive Image Understanding:** Leverage built-in capabilities for object detection, segmentation, and super-resolution.
*   **LoRA Support:** Utilize LoRA models like MajicBeauty for highly realistic image generation.

## Introduction

**Qwen-Image** is a powerful 20B parameter MMDiT (Multimodal Diffusion Transformer) image foundation model, designed for advanced image generation and editing tasks.  It excels in creating images with precise text rendering, especially for Chinese, and offers robust capabilities for image manipulation and understanding.

## What's New

*   **2025.08.09:** Added support for LoRA models, such as MajicBeauty LoRA, available on [ModelScope](https://modelscope.cn/models/merjic/majicbeauty-qwen1/summary).
*   **2025.08.05:** Native support in ComfyUI ([Qwen-Image in ComfyUI: New Era of Text Generation in Images!](https://blog.comfy.org/p/qwen-image-in-comfyui-new-era-of)).
*   **2025.08.05:** Integrated into [Qwen Chat](https://chat.qwen.ai/) (choose "Image Generation").
*   **2025.08.05:** [Technical Report](https://arxiv.org/abs/2508.02324) released on Arxiv!
*   **2025.08.04:** Weights released on [Hugging Face](https://huggingface.co/Qwen/Qwen-Image) and [Modelscope](https://modelscope.cn/models/Qwen/Qwen-Image).
*   **2025.08.04:** Qwen-Image officially released! Check our [Blog](https://qwenlm.github.io/blog/qwen-image) for more details.

> [!NOTE]
> The editing version of Qwen-Image will be released soon. Stay tuned!
> 
> Due to heavy traffic, if you'd like to experience our demo online, we also recommend visiting DashScope, WaveSpeed, and LibLib. Please find the links below in the community support.

## Quick Start

Get started with Qwen-Image in a few simple steps.

1.  Make sure you have `transformers>=4.51.3` (Supporting Qwen2.5-VL).
2.  Install the latest version of `diffusers`:
    ```bash
    pip install git+https://github.com/huggingface/diffusers
    ```

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
    "en": "Ultra HD, 4K, cinematic composition.", # for english prompt
    "zh": "超清，4K，电影级构图" # for chinese prompt
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

## Show Cases

Qwen-Image demonstrates impressive capabilities across various image generation and editing tasks:

*   **High-Fidelity Text Rendering:** Generate images with crisp, readable text in various languages.
*   **Diverse Artistic Styles:** Produce images in a wide range of styles, from photorealistic to artistic.
*   **Advanced Editing Features:**  Perform sophisticated image manipulation tasks.
*   **Image Understanding Capabilities:** Supports object detection, segmentation, and more.

### Explore Qwen-Image's Advanced Features

#### Prompt Enhancement
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

## AI Arena: Benchmarking and Evaluation

Evaluate Qwen-Image's performance on the [AI Arena](https://aiarena.alibaba-inc.com), a platform that uses the Elo rating system for transparent and data-driven model comparisons.

![AI Arena](assets/figure_aiarena_website.png)

View the latest leaderboard rankings at [AI Arena Learboard](https://aiarena.alibaba-inc.com/corpora/arena/leaderboard?arenaType=text2image).

To deploy your model or participate in the evaluation, contact weiyue.wy@alibaba-inc.com.

## Community Support & Resources

*   **Hugging Face:** Supported in Diffusers.  LoRA and finetuning workflows coming soon.
*   **ModelScope:** Offers comprehensive support, including [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) and [DiffSynth-Engine](https://github.com/modelscope/DiffSynth-Engine), as well as hands-on experiences via [ModelScope AIGC Central](https://www.modelscope.cn/aigc).
*   **WaveSpeedAI:** Deploys Qwen-Image on their platform ([model page](https://wavespeed.ai/models/wavespeed-ai/qwen-image/text-to-image)).
*   **LiblibAI:** Native support ([community](https://www.liblib.art/modelinfo/c62a103bd98a4246a2334e2d952f7b21?from=sd&versionUuid=75e0be0c93b34dd8baeec9c968013e0c)).

## License

Qwen-Image is released under the Apache 2.0 License.

## Citation

If you use Qwen-Image in your work, please cite our technical report:

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

Connect with the research team and community:

*   Join our [Discord](https://discord.gg/CV4E9rpNSD).
*   Join our [WeChat groups](assets/wechat.png).

We welcome your contributions through issues and pull requests on [GitHub](https://github.com/QwenLM/Qwen-Image). We're also hiring (FTEs and research interns); contact fulai.hr@alibaba-inc.com.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=QwenLM/Qwen-Image&type=Date)](https://www.star-history.com/#QwenLM/Qwen-Image&Date)