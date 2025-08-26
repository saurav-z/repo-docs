<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/qwen_image_logo.png" width="400"/>
</p>

<p align="center">
    &nbsp;💜 <a href="https://chat.qwen.ai/">Qwen Chat</a>&nbsp; |
    &nbsp;🤗 <a href="https://huggingface.co/Qwen/Qwen-Image">HuggingFace(T2I)</a>&nbsp; |
    &nbsp;🤗 <a href="https://huggingface.co/Qwen/Qwen-Image-Edit">HuggingFace(Edit)</a>&nbsp; | &nbsp;🤖 <a href="https://modelscope.cn/models/Qwen/Qwen-Image">ModelScope-T2I</a>&nbsp; | &nbsp;🤖 <a href="https://modelscope.cn/models/Qwen/Qwen-Image-Edit">ModelScope-Edit</a>&nbsp; | &nbsp;📑 <a href="https://arxiv.org/abs/2508.02324">Tech Report</a> &nbsp; | &nbsp;📑 <a href="https://qwenlm.github.io/blog/qwen-image/">Blog(T2I)</a> &nbsp; | &nbsp;📑 <a href="https://qwenlm.github.io/blog/qwen-image-edit/">Blog(Edit)</a> &nbsp;
    <br>
    🖥️ <a href="https://huggingface.co/spaces/Qwen/Qwen-Image">T2I Demo</a>&nbsp; | 🖥️ <a href="https://huggingface.co/spaces/Qwen/Qwen-Image-Edit">Edit Demo</a>&nbsp; | &nbsp;💬 <a href="https://github.com/QwenLM/Qwen-Image/blob/main/assets/wechat.png">WeChat (微信)</a>&nbsp; | &nbsp;🫨 <a href="https://discord.gg/CV4E9rpNSD">Discord</a>&nbsp;
</p>

<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/merge3.jpg" width="1024"/>
</p>

## Qwen-Image: Generate and Edit Stunning Images with Advanced AI 

**Qwen-Image is a cutting-edge 20B MMDiT image foundation model, revolutionizing image generation and editing with unparalleled text rendering and image manipulation capabilities. Explore the original repo on [GitHub](https://github.com/QwenLM/Qwen-Image) for more details.**

### Key Features

*   **Exceptional Text Rendering:** Accurately renders complex text in various languages (English, Chinese, etc.), maintaining typographic details and contextual harmony.
*   **Versatile Image Generation:** Supports diverse artistic styles, including photorealistic, anime, and artistic.
*   **Advanced Image Editing:** Enables style transfer, object manipulation (insertion/removal), detail enhancement, and text editing within images.
*   **Image Understanding Capabilities:** Includes object detection, semantic segmentation, and more for intelligent image manipulation.
*   **Image Editing Tutorial:** The tutorial gives concrete examples of semantic and appearance editing, including advanced chained editing.

### News

*   **2025.08.19:** Performance improvements expected with the latest diffusers commit.
*   **2025.08.18:** Qwen-Image-Edit is now open-sourced!
*   **2025.08.09:** Supports LoRA models, such as MajicBeauty LoRA.
*   **2025.08.05:** Native support in ComfyUI.
*   **2025.08.05:** Available on Qwen Chat.
*   **2025.08.05:** Technical Report released on Arxiv.
*   **2025.08.04:** Qwen-Image weights released on Hugging Face and ModelScope.
*   **2025.08.04:** Qwen-Image released.

> [!NOTE]
> Consider visiting DashScope, WaveSpeed, and LibLib for online demo experiences due to potential heavy traffic.

### Quick Start

1.  **Prerequisites:** Ensure you have `transformers>=4.51.3` and the latest `diffusers` installed:

    ```bash
    pip install git+https://github.com/huggingface/diffusers
    ```

2.  **Text-to-Image Example:**

    ```python
    from diffusers import DiffusionPipeline
    import torch

    model_name = "Qwen/Qwen-Image"

    # Load the pipeline (CUDA or CPU)
    if torch.cuda.is_available():
        torch_dtype = torch.bfloat16
        device = "cuda"
    else:
        torch_dtype = torch.float32
        device = "cpu"

    pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch_dtype)
    pipe = pipe.to(device)

    positive_magic = {
        "en": ", Ultra HD, 4K, cinematic composition.",  # English prompt
        "zh": ", 超清，4K，电影级构图.",  # Chinese prompt
    }

    # Generate Image
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

3.  **Image Editing Example:**

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
> Use prompt rewriting for more stable image editing. See `src/examples/edit_demo.py` for examples.

### Show Cases

*   **General Cases:** Showcases high-fidelity text rendering and support for various art styles.
*   **Image Editing Tutorial:** Includes clear examples of semantic and appearance editing, as well as how to chain edits.

### Advanced Usage

#### Prompt Enhancement
Use Prompt Enhancement Tool powered by Qwen-Plus.

```python
from tools.prompt_utils import rewrite
prompt = rewrite(prompt)
```

Or run from the command line:
```bash
cd src
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx python examples/generate_w_prompt_enhance.py
```

### Deploy Qwen-Image

Qwen-Image supports Multi-GPU API Server for local deployment.
**Multi-GPU API Server Pipeline & Usage**
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

The platform built on the Elo rating system, for open benchmarking.

Latest leaderboard rankings can be viewed at [AI Arena Learboard](https://aiarena.alibaba-inc.com/corpora/arena/leaderboard?arenaType=text2image).

Contact weiyue.wy@alibaba-inc.com to deploy your model.

### Community Support

*   **Hugging Face:** Supports LoRA and finetuning workflows.
*   **ModelScope:** Provides comprehensive support and hands-on experiences.
*   **WaveSpeedAI:** Deployed Qwen-Image on their platform.
*   **LiblibAI:** Native support for Qwen-Image.
*   **cache-dit:** Offers cache acceleration support.

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

Join our [Discord](https://discord.gg/z3GAxXZ9Ce) or scan the QR code to connect via our [WeChat groups](assets/wechat.png).

We welcome your issues and pull requests on GitHub.

Reach out to us at fulai.hr@alibaba-inc.com for full-time positions or research internships.

### Star History

[![Star History Chart](https://api.star-history.com/svg?repos=QwenLM/Qwen-Image&type=Date)](https://www.star-history.com/#QwenLM/Qwen-Image&Date)