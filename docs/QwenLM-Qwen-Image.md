<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/qwen_image_logo.png" width="400"/>
</p>

<p align="center">
    &nbsp&nbsp💜 <a href="https://chat.qwen.ai/">Qwen Chat</a>&nbsp&nbsp |
    &nbsp&nbsp🤗 <a href="https://huggingface.co/Qwen/Qwen-Image">Hugging Face</a>&nbsp&nbsp |
    &nbsp&nbsp🤖 <a href="https://modelscope.cn/models/Qwen/Qwen-Image">ModelScope</a>&nbsp&nbsp |
    &nbsp&nbsp 📑 <a href="https://arxiv.org/abs/2508.02324">Tech Report</a> &nbsp&nbsp |
    &nbsp&nbsp 📑 <a href="https://qwenlm.github.io/blog/qwen-image/">Blog</a> &nbsp&nbsp
    <br>
    🖥️ <a href="https://huggingface.co/spaces/Qwen/Qwen-Image">Demo</a>&nbsp&nbsp |
    &nbsp&nbsp💬 <a href="https://github.com/QwenLM/Qwen-Image/blob/main/assets/wechat.png">WeChat (微信)</a>&nbsp&nbsp |
    &nbsp&nbsp🫨 <a href="https://discord.gg/CV4E9rpNSD">Discord</a>&nbsp&nbsp
</p>

<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/merge3.jpg" width="1024"/>
</p>

## Qwen-Image: Unleash the Power of AI for Stunning Image Generation and Editing

**Qwen-Image** is a state-of-the-art 20B MMDiT image foundation model from Qwen, delivering exceptional performance in text rendering, image generation, and advanced editing.  Explore the original repository [here](https://github.com/QwenLM/Qwen-Image).

**Key Features:**

*   **Exceptional Text Rendering:**  Achieves unparalleled accuracy in rendering text within images, particularly for Chinese.
*   **Versatile Image Generation:**  Supports diverse artistic styles, from photorealistic to anime, adapting to creative prompts.
*   **Advanced Image Editing:**  Enables style transfer, object manipulation, text editing, and human pose control.
*   **Image Understanding Capabilities:**  Offers object detection, semantic segmentation, and more for intelligent image manipulation.

### Quick Start

1.  Ensure you have `transformers>=4.51.3` (supports Qwen2.5-VL)

2.  Install the latest `diffusers` package:
    ```bash
    pip install git+https://github.com/huggingface/diffusers
    ```

**Example Usage:**

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

### News

*   **[2025.08.09]**: Qwen-Image supports LoRA models like MajicBeauty LoRA ([ModelScope](https://modelscope.cn/models/merjic/majicbeauty-qwen1/summary)).
*   **[2025.08.05]**: Native support in ComfyUI ([ComfyUI Blog](https://blog.comfy.org/p/qwen-image-in-comfyui-new-era-of)).
*   **[2025.08.05]**: Available on Qwen Chat ([Qwen Chat](https://chat.qwen.ai/)).
*   **[2025.08.05]**: Technical Report published ([Arxiv](https://arxiv.org/abs/2508.02324)).
*   **[2025.08.04]**: Model weights released ([Hugging Face](https://huggingface.co/Qwen/Qwen-Image) and [ModelScope](https://modelscope.cn/models/Qwen/Qwen-Image)).
*   **[2025.08.04]**: Qwen-Image released ([Blog](https://qwenlm.github.io/blog/qwen-image)).

### Show Cases

[Show cases with relevant images and captions demonstrating the key features – text rendering, diverse image generation, image editing, and image understanding tasks.]

### Advanced Usage

*   **Prompt Enhancement:** Utilize the official Prompt Enhancement Tool powered by Qwen-Plus for improved results.  Integrate it directly into your code:

    ```python
    from tools.prompt_utils import rewrite
    prompt = rewrite(prompt)
    ```

    Or, run the example script from the command line:

    ```bash
    cd src
    DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx python examples/generate_w_prompt_enhance.py
    ```

### Deploy Qwen-Image

Qwen-Image supports Multi-GPU API Server for local deployment:

#### Multi-GPU API Server Pipeline & Usage

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

Evaluate Qwen-Image's image generation capabilities on the [AI Arena](https://aiarena.alibaba-inc.com), an open benchmarking platform using the Elo rating system.

![AI Arena](assets/figure_aiarena_website.png)

View latest rankings at [AI Arena Leaderboard](https://aiarena.alibaba-inc.com/corpora/arena/leaderboard?arenaType=text2image).  Contact weiyue.wy@alibaba-inc.com to deploy your model on AI Arena.

### Community Support

*   **Hugging Face:**  Supports Qwen-Image. LoRA and finetuning workflows are coming soon.
*   **ModelScope:** Provides comprehensive support with various tools:
    *   [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) (low-GPU memory, FP8 quantization, LoRA training)
    *   [DiffSynth-Engine](https://github.com/modelscope/DiffSynth-Engine) (inference optimizations)
    *   [ModelScope AIGC Central](https://www.modelscope.cn/aigc) (hands-on experiences)
*   **WaveSpeedAI:** Deploys Qwen-Image on their platform ([WaveSpeedAI](https://wavespeed.ai/models/wavespeed-ai/qwen-image/text-to-image)).
*   **LiblibAI:** Offers native support ([LiblibAI](https://www.liblib.art/modelinfo/c62a103bd98a4246a2334e2d952f7b21?from=sd&versionUuid=75e0be0c93b34dd8baeec9c968013e0c)).
*   **cache-dit:** Offers cache acceleration support for Qwen-Image with DBCache, TaylorSeer and Cache CFG. ([example](https://github.com/vipshop/cache-dit/blob/main/examples/run_qwen_image.py))

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

[Discord](https://discord.gg/z3GAxXZ9Ce) | [WeChat (scan QR code)](assets/wechat.png)

[GitHub Issues/PRs](link to GitHub repository) | fulai.hr@alibaba-inc.com (for full-time/internship opportunities)

### Star History

[![Star History Chart](https://api.star-history.com/svg?repos=QwenLM/Qwen-Image&type=Date)](https://www.star-history.com/#QwenLM/Qwen-Image&Date)