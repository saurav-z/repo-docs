<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/qwen_image_logo.png" width="400"/>
<p>
<p align="center">
    &nbsp&nbsp💜 <a href="https://chat.qwen.ai/">Qwen Chat</a>&nbsp&nbsp |
    &nbsp&nbsp🤗 <a href="https://huggingface.co/Qwen/Qwen-Image">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/models/Qwen/Qwen-Image">ModelScope</a>&nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://arxiv.org/abs/2508.02324">Tech Report</a> &nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://qwenlm.github.io/blog/qwen-image/">Blog</a> &nbsp&nbsp
<br>
    🖥️ <a href="https://huggingface.co/spaces/Qwen/Qwen-Image">Demo</a>&nbsp&nbsp | &nbsp&nbsp💬 <a href="https://github.com/QwenLM/Qwen-Image/blob/main/assets/wechat.png">WeChat (微信)</a>&nbsp&nbsp | &nbsp&nbsp🫨 <a href="https://discord.gg/CV4E9rpNSD">Discord</a>&nbsp&nbsp
</p>

<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/merge3.jpg" width="1024"/>
<p>

# Qwen-Image: Generate and Edit Images with Exceptional Text Rendering

Qwen-Image is a powerful 20B MMDiT image foundation model revolutionizing image generation and editing; explore its capabilities and get started with our [GitHub repository](https://github.com/QwenLM/Qwen-Image)!

## Key Features

*   **Exceptional Text Rendering:** Generate images with high-fidelity text, perfectly preserving typographic details and integrating text seamlessly, especially for Chinese.
*   **Versatile Image Generation:** Create diverse images in various artistic styles, from photorealistic scenes to anime, adapting fluidly to creative prompts.
*   **Advanced Image Editing:** Perform complex edits, including style transfer, object manipulation, detail enhancement, and even human pose adjustments.
*   **Image Understanding Capabilities:** Utilize advanced features such as object detection, semantic segmentation, and super-resolution for intelligent image manipulation.

## Quick Start

1.  Ensure you have `transformers>=4.51.3`.
2.  Install the latest version of `diffusers`:

    ```bash
    pip install git+https://github.com/huggingface/diffusers
    ```

3.  Run the following code snippet to generate images:

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
        "en": "Ultra HD, 4K, cinematic composition.",  # for english prompt
        "zh": "超清，4K，电影级构图"  # for chinese prompt
    }

    # Generate image
    prompt = '''A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup," with a neon light beside it displaying "通义千问". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the poster is written "π≈3.1415926-53589793-23846264-33832795-02384197".'''

    negative_prompt = " "  # Recommended if you don't use a negative prompt.

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

Qwen-Image excels in high-fidelity text rendering across various languages, including English and Chinese, preserving typographic details and maintaining contextual harmony.

(Include images from the original README here)

Beyond text, it supports a wide range of artistic styles, and enables advanced image editing operations, bringing professional-grade editing within reach.

## Advanced Usage

### Prompt Enhancement

For enhanced prompt optimization and multi-language support, use our official Prompt Enhancement Tool powered by Qwen-Plus:

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


## AI Arena

To comprehensively evaluate the general image generation capabilities of Qwen-Image and objectively compare it with state-of-the-art closed-source APIs, we introduce [AI Arena](https://aiarena.alibaba-inc.com), an open benchmarking platform built on the Elo rating system. AI Arena provides a fair, transparent, and dynamic environment for model evaluation.

(Include image from original README)

## Community Support

### Hugging Face

*   Diffusers has supported Qwen-Image since day 0. Support for LoRA and finetuning workflows is currently in development and will be available soon.

### ModelScope

*   [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio): Supports Qwen-Image with low-GPU-memory layer-by-layer offload (inference within 4GB VRAM), FP8 quantization, and LoRA/full training.
*   [DiffSynth-Engine](https://github.com/modelscope/DiffSynth-Engine): Delivers optimizations for Qwen-Image inference and deployment, including FBCache-based acceleration and classifier-free guidance (CFG) parallel.
*   [ModelScope AIGC Central](https://www.modelscope.cn/aigc): Provides hands-on experiences for Qwen Image, including Image Generation and LoRA Training.

### WaveSpeedAI

*   WaveSpeed has deployed Qwen-Image on their platform. Visit their [model page](https://wavespeed.ai/models/wavespeed-ai/qwen-image/text-to-image).

### LiblibAI

*   LiblibAI offers native support for Qwen-Image. Visit their [community](https://www.liblib.art/modelinfo/c62a103bd98a4246a2334e2d952f7b21?from=sd&versionUuid=75e0be0c93b34dd8baeec9c968013e0c) page for more details.

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

Connect with the research team and contribute to Qwen-Image.

*   Join our [Discord](https://discord.gg/z3GAxXZ9Ce).
*   Connect via our [WeChat groups](assets/wechat.png) (scan QR code).
*   Contribute through issues and pull requests on [GitHub](https://github.com/QwenLM/Qwen-Image).
*   For FTE and research intern opportunities, contact fulai.hr@alibaba-inc.com

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=QwenLM/Qwen-Image&type=Date)](https://www.star-history.com/#QwenLM/Qwen-Image&Date)