<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/qwen_image_logo.png" width="400"/>
</p>

<p align="center">
    &nbsp; 💜 <a href="https://chat.qwen.ai/">Qwen Chat</a> &nbsp; |
    &nbsp; 🤗 <a href="https://huggingface.co/Qwen/Qwen-Image">HuggingFace(T2I)</a> &nbsp; |
    &nbsp; 🤗 <a href="https://huggingface.co/Qwen/Qwen-Image-Edit">HuggingFace(Edit)</a> &nbsp; | &nbsp; 🤖 <a href="https://modelscope.cn/models/Qwen/Qwen-Image">ModelScope-T2I</a> &nbsp; | &nbsp; 🤖 <a href="https://modelscope.cn/models/Qwen/Qwen-Image-Edit">ModelScope-Edit</a> &nbsp; | &nbsp; 📑 <a href="https://arxiv.org/abs/2508.02324">Tech Report</a> &nbsp; | &nbsp; 📑 <a href="https://qwenlm.github.io/blog/qwen-image/">Blog(T2I)</a> &nbsp; | &nbsp; 📑 <a href="https://qwenlm.github.io/blog/qwen-image-edit/">Blog(Edit)</a>
    <br>
    🖥️ <a href="https://huggingface.co/spaces/Qwen/Qwen-Image">T2I Demo</a> &nbsp; | 🖥️ <a href="https://huggingface.co/spaces/Qwen/Qwen-Image-Edit">Edit Demo</a> &nbsp; | &nbsp; 💬 <a href="https://github.com/QwenLM/Qwen-Image/blob/main/assets/wechat.png">WeChat (微信)</a> &nbsp; | &nbsp; 🫨 <a href="https://discord.gg/CV4E9rpNSD">Discord</a>
</p>

<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/merge3.jpg" width="1024"/>
</p>

## Qwen-Image: Unleash Your Imagination with Advanced AI Image Generation and Editing

**Qwen-Image is a cutting-edge 20B MMDiT image foundation model, revolutionizing the way you create and manipulate images.**  Explore a world of creative possibilities, from generating stunning visuals from text to seamlessly editing existing images with unprecedented precision.  Find the original repo [here](https://github.com/QwenLM/Qwen-Image).

**Key Features:**

*   **Exceptional Text Rendering:**  Generate images with complex text that's rendered accurately, preserving typographic details and layout.
*   **Precise Image Editing:**  Edit images with advanced features like style transfer, object manipulation, and text editing within images.
*   **Diverse Artistic Styles:**  Create images in various styles, including photorealistic, artistic, anime, and more, simply by describing what you want.
*   **Image Understanding Capabilities:**  Leverage built-in features like object detection, semantic segmentation, and super-resolution.
*   **Image Editing Tutorial:** Step-by-step guide that provides details for semantic, appearance, and advanced text editing.

## Latest News

*   **2025.08.19:** [Qwen-Image-Edit Update] To ensure optimal results, please update to the latest diffusers commit.
*   **2025.08.18:** [Qwen-Image-Edit Open Source] Qwen-Image-Edit is now open-sourced!
*   **2025.08.09:** [LoRA Support] Qwen-Image now supports various LoRA models, such as MajicBeauty LoRA.
*   **2025.08.05:** [ComfyUI Support] Qwen-Image is now natively supported in ComfyUI.
*   **2025.08.05:** [Qwen Chat Integration] Qwen-Image is now available on Qwen Chat.
*   **2025.08.05:** [Technical Report] Technical Report is available on Arxiv!
*   **2025.08.04:** [Weight Release] Qwen-Image weights are now available on Hugging Face and ModelScope.
*   **2025.08.04:** [Qwen-Image Release] Qwen-Image is now available. Check our blog for more details!

> [!NOTE]
>  If you experience high traffic on the online demo, try these alternative platforms: DashScope, WaveSpeed, and LibLib. Links are provided in the Community Support section.

## Quick Start

1.  **Prerequisites:** Ensure you have `transformers>=4.51.3` (supporting Qwen2.5-VL).
2.  **Install Diffusers:**
    ```bash
    pip install git+https://github.com/huggingface/diffusers
    ```

### Text-to-Image

Here's a code example to generate images from text prompts:

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
> To improve editing results, we strongly recommend using prompt rewriting. See the Advanced Usage section.

### Advanced Usage

#### Prompt Enhancement

*   **Text-to-Image:** Utilize the Prompt Enhancement Tool powered by Qwen-Plus for enhanced prompt optimization and multi-language support:

    ```python
    from tools.prompt_utils import rewrite
    prompt = rewrite(prompt)
    ```

    Or run the example script:
    ```bash
    cd src
    DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx python examples/generate_w_prompt_enhance.py
    ```

*   **Image Edit:**  For improved stability in image editing, use the Prompt Enhancement Tool powered by Qwen-VL-Max:
    ```python
    from tools.prompt_utils import polish_edit_prompt
    prompt = polish_edit_prompt(prompt, pil_image)
    ```

## Deploy Qwen-Image

Qwen-Image supports a Multi-GPU API Server for local deployment.

### Multi-GPU API Server Pipeline & Usage

The Multi-GPU API Server provides a Gradio-based web interface with:

*   Multi-GPU parallel processing
*   Queue management for high concurrency
*   Automatic prompt optimization
*   Support for multiple aspect ratios

**Configuration:**

```bash
export NUM_GPUS_TO_USE=4          # Number of GPUs to use
export TASK_QUEUE_SIZE=100        # Task queue size
export TASK_TIMEOUT=300           # Task timeout in seconds
```

**Run the demo:**

```bash
cd src
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxx python examples/demo.py
```

## Show Cases

*   **General Cases:** Qwen-Image excels at high-fidelity text rendering in multiple languages and generates diverse images across artistic styles.

    *   [Image 1 - Text rendering example](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/s1.jpg#center)
    *   [Image 2 - Diverse artistic styles](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/s2.jpg#center)
    *   [Image 3 - Image Editing](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/s3.jpg#center)
    *   [Image 4 - Image understanding tasks](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/s4.jpg#center)

*   **Image Editing Tutorial:** Comprehensive walkthrough of the image editing capabilites.

    *   [MBTI meme series](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片4.JPG#center)
    *   [Viewpoint transformation 90 degrees](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片12.JPG#center)
    *   [Viewpoint transformation 180 degrees](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片13.JPG#center)
    *   [Style transfer](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片1.JPG#center)
    *   [Adding a signboard](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片6.JPG#center)
    *   [Removing fine strands of hair](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片7.JPG#center)
    *   [Modifying text color](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片8.JPG#center)
    *   [Modifying backgrounds](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片11.JPG#center)
    *   [Modifying clothing](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片5.JPG#center)
    *   [Editing English text 1](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片15.JPG#center)
    *   [Editing English text 2](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片16.JPG#center)
    *   [Editing Chinese posters](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片17.JPG#center)

## AI Arena

[AI Arena](https://aiarena.alibaba-inc.com) is an open benchmarking platform for evaluating image generation models using the Elo rating system. Participate in model evaluations to compare Qwen-Image's performance against other models.

*   [AI Arena Learboard](https://aiarena.alibaba-inc.com/corpora/arena/leaderboard?arenaType=text2image)

To deploy your model on AI Arena, contact weiyue.wy@alibaba-inc.com.

## Community Support

*   **Hugging Face:**  Day-0 support and developing support for LoRA and finetuning workflows.
*   **ModelScope:** Comprehensive support, including low-GPU-memory offload, FP8 quantization, LoRA, and full training, via [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) and advanced optimizations for inference and deployment via [DiffSynth-Engine](https://github.com/modelscope/DiffSynth-Engine), and AIGC tools on [ModelScope AIGC Central](https://www.modelscope.cn/aigc).
*   **WaveSpeedAI:** Native support, visit their [model page](https://wavespeed.ai/models/wavespeed-ai/qwen-image/text-to-image).
*   **LiblibAI:** Native support, visit their [community](https://www.liblib.art/modelinfo/c62a103bd98a4246a2334e2d952f7b21?from=sd&versionUuid=75e0be0c93b34dd8baeec9c968013e0c).
*   **Inference Acceleration Method: cache-dit:** Offers cache acceleration support with DBCache, TaylorSeer and Cache CFG. Visit their [example](https://github.com/vipshop/cache-dit/blob/main/examples/run_qwen_image.py).

## License Agreement

Qwen-Image is licensed under Apache 2.0.

## Citation

If you use Qwen-Image, please cite our work:

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

Connect with our research team and the community:

*   [Discord](https://discord.gg/z3GAxXZ9Ce)
*   [WeChat QR code](assets/wechat.png)

For questions, feedback, contributions, or job opportunities, please:

*   Open issues and pull requests on [GitHub](https://github.com/QwenLM/Qwen-Image).
*   Contact us regarding full-time positions and research internships at fulai.hr@alibaba-inc.com.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=QwenLM/Qwen-Image&type=Date)](https://www.star-history.com/#QwenLM/Qwen-Image&Date)