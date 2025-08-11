<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/qwen_image_logo.png" width="400" alt="Qwen-Image Logo"/>
</p>

<p align="center">
  Unleash your creativity with **Qwen-Image**, a powerful image generation and editing model.  
  <br/>
  <a href="https://chat.qwen.ai/">Qwen Chat</a>&nbsp;&nbsp; |
  <a href="https://huggingface.co/Qwen/Qwen-Image">Hugging Face</a>&nbsp;&nbsp; |
  <a href="https://modelscope.cn/models/Qwen/Qwen-Image">ModelScope</a>&nbsp;&nbsp; |
  <a href="https://arxiv.org/abs/2508.02324">Tech Report</a>&nbsp;&nbsp; |
  <a href="https://qwenlm.github.io/blog/qwen-image/">Blog</a>&nbsp;&nbsp;
  <br/>
  <a href="https://huggingface.co/spaces/Qwen/Qwen-Image">Demo</a>&nbsp;&nbsp; |
  <a href="https://github.com/QwenLM/Qwen-Image/blob/main/assets/wechat.png">WeChat (微信)</a>&nbsp;&nbsp; |
  <a href="https://discord.gg/CV4E9rpNSD">Discord</a>
  <br/>
  <a href="https://github.com/QwenLM/Qwen-Image">View on GitHub</a>
</p>


## Qwen-Image: Advanced Image Generation and Editing

Qwen-Image is a 20B parameter MMDiT image foundation model, setting new benchmarks in both image generation and editing. Experience exceptional performance in text rendering and unlock your creative potential with advanced features.

<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/merge3.jpg" width="1024" alt="Qwen-Image Showcase"/>
</p>

**Key Features:**

*   **Superior Text Rendering:** Achieve unparalleled accuracy in rendering text within images, supporting various languages, including Chinese.
*   **Versatile Image Generation:** Generate images across diverse artistic styles, from photorealistic to artistic, with ease.
*   **Advanced Image Editing:** Perform complex edits such as style transfer, object manipulation, and text modification for professional-grade results.
*   **Image Understanding Capabilities:** Leverage object detection, segmentation, and other features for intelligent image manipulation.
*   **LoRA Model Support:** Supports LoRA models, like MajicBeauty, for tailored image generation.

## What's New
*   **2025.08.09**: Qwen-Image now supports LoRA models, such as MajicBeauty LoRA, enabling the generation of highly realistic beauty images. Check out the available weights on [ModelScope](https://modelscope.cn/models/merjic/majicbeauty-qwen1/summary)
*   **2025.08.05**: Native ComfyUI Support ([Qwen-Image in ComfyUI: New Era of Text Generation in Images!](https://blog.comfy.org/p/qwen-image-in-comfyui-new-era-of))
*   **2025.08.05**: Qwen-Image is now available on Qwen Chat.
*   **2025.08.05**: [Technical Report](https://arxiv.org/abs/2508.02324) released on Arxiv!
*   **2025.08.04**: Weights released on [Hugging Face](https://huggingface.co/Qwen/Qwen-Image) and [Modelscope](https://modelscope.cn/models/Qwen/Qwen-Image)!
*   **2025.08.04**: [Blog](https://qwenlm.github.io/blog/qwen-image) released!

> [!NOTE]
> The editing version of Qwen-Image will be released soon. Stay tuned!
> 
> Due to heavy traffic, if you'd like to experience our demo online, we also recommend visiting DashScope, WaveSpeed, and LibLib. Please find the links below in the community support.

## Quick Start

Follow these steps to get started:

1.  Ensure you have `transformers>=4.51.3`.
2.  Install the latest version of `diffusers`:
    ```bash
    pip install git+https://github.com/huggingface/diffusers
    ```

**Code Example:**

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

*   **High-Fidelity Text Rendering:** Witness precise and seamless integration of text, including Chinese and other languages, into your images.

<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/s1.jpg#center" alt="Text Rendering Example"/>
</p>

*   **Diverse Artistic Styles:** Generate a wide range of images, from realistic to artistic, anime to minimalist.

<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/s2.jpg#center" alt="Artistic Styles Example"/>
</p>

*   **Advanced Editing Capabilities:** Perform advanced editing tasks such as style transfer, object manipulation, and text editing within images.

<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/s3.jpg#center" alt="Image Editing Example"/>
</p>

*   **Image Understanding Tasks:** Leverage image understanding tasks for depth estimation, object detection and more, enhancing your editing workflow.

<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/s4.jpg#center" alt="Image Understanding Example"/>
</p>

### Advanced Usage

#### Prompt Enhancement

Enhance your prompts with our official Prompt Enhancement Tool, powered by Qwen-Plus.

**Integration:**

```python
from tools.prompt_utils import rewrite
prompt = rewrite(prompt)
```

**Command Line:**

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

## AI Arena

Evaluate Qwen-Image's capabilities on the [AI Arena](https://aiarena.alibaba-inc.com), an open benchmarking platform.

<p align="center">
    <img src="assets/figure_aiarena_website.png" alt="AI Arena"/>
</p>

For deployment and evaluation inquiries, contact weiyue.wy@alibaba-inc.com.

## Community Support

### Hugging Face

*   Fully supported by Diffusers.
*   Support for LoRA and finetuning workflows is currently in development.

### ModelScope

*   **[DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)**: Low-GPU-memory, FP8 quantization, and LoRA support.
*   **[DiffSynth-Engine](https://github.com/modelscope/DiffSynth-Engine)**: Advanced optimizations for inference and deployment.
*   **[ModelScope AIGC Central](https://www.modelscope.cn/aigc)**: Hands-on experiences, including Image Generation and LoRA Training.

### WaveSpeedAI

*   Deployed on their platform: [model page](https://wavespeed.ai/models/wavespeed-ai/qwen-image/text-to-image).

### LiblibAI

*   Native support available: [community](https://www.liblib.art/modelinfo/c62a103bd98a4246a2334e2d952f7b21?from=sd&versionUuid=75e0be0c93b34dd8baeec9c968013e0c).

## License Agreement

Qwen-Image is licensed under Apache 2.0.

## Citation

If you find this project helpful, please cite our work:

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

Connect with our research team and contribute to Qwen-Image.

*   Join our [Discord](https://discord.gg/z3GAxXZ9Ce).
*   Join our [WeChat groups](assets/wechat.png).
*   Submit issues and pull requests on [GitHub](https://github.com/QwenLM/Qwen-Image).
*   For full-time and intern positions, contact fulai.hr@alibaba-inc.com.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=QwenLM/Qwen-Image&type=Date)](https://www.star-history.com/#QwenLM/Qwen-Image&Date)