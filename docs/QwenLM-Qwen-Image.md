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

# Qwen-Image: Unleash Stunning Visual Creations with Advanced AI

**Qwen-Image is a powerful 20B MMDiT image foundation model that excels in generating and editing images with unparalleled text rendering capabilities, especially for Chinese.** ([See the original repository](https://github.com/QwenLM/Qwen-Image))

## Key Features

*   **Exceptional Text Rendering:** Generate images with highly accurate and legible text, including Chinese and English.
*   **Versatile Image Generation:** Create diverse images in various artistic styles, from photorealistic to artistic.
*   **Advanced Image Editing:**  Perform style transfers, object manipulation, detail enhancement, and text editing with ease.
*   **Image Understanding Capabilities:** Leverage built-in support for object detection, semantic segmentation, and more for intelligent editing.
*   **Multi-Language Support:** Optimized prompt enhancement for both English and Chinese prompts.

## News

*   **[2025.08.09]**  Supports a variety of LoRA models, such as MajicBeauty LoRA, enabling the generation of highly realistic beauty images. Check out the available weights on [ModelScope](https://modelscope.cn/models/merjic/majicbeauty-qwen1/summary)
*   **[2025.08.05]**  Natively supported in ComfyUI, see [Qwen-Image in ComfyUI: New Era of Text Generation in Images!](https://blog.comfy.org/p/qwen-image-in-comfyui-new-era-of)
*   **[2025.08.05]**  Now available on Qwen Chat. Click [Qwen Chat](https://chat.qwen.ai/) and choose "Image Generation".
*   **[2025.08.05]**  Technical Report released on Arxiv! ([Technical Report](https://arxiv.org/abs/2508.02324))
*   **[2025.08.04]**  Weights released on [Huggingface](https://huggingface.co/Qwen/Qwen-Image) and [Modelscope](https://modelscope.cn/models/Qwen/Qwen-Image)!
*   **[2025.08.04]**  Qwen-Image is released! Check our [Blog](https://qwenlm.github.io/blog/qwen-image) for more details!

>   [!NOTE]
>   The editing version of Qwen-Image will be released soon. Stay tuned!
>
>   Due to heavy traffic, if you'd like to experience our demo online, we also recommend visiting DashScope, WaveSpeed, and LibLib. Please find the links below in the community support.

## Quick Start

1.  Make sure your transformers>=4.51.3 (Supporting Qwen2.5-VL)

2.  Install the latest version of diffusers

    ```bash
    pip install git+https://github.com/huggingface/diffusers
    ```

    The following contains a code snippet illustrating how to use the model to generate images based on text prompts:

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

Qwen-Image showcases impressive capabilities across various image generation and manipulation tasks. Explore these examples:

### High-Fidelity Text Rendering

Qwen-Image excels at rendering text with remarkable accuracy, maintaining typographic details, layout coherence, and contextual harmony, even with Chinese characters.

![](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/s1.jpg#center)

### Diverse Artistic Styles

Generate images in a wide range of artistic styles, including photorealistic scenes, impressionist paintings, anime aesthetics, and minimalist designs.

![](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/s2.jpg#center)

### Advanced Image Editing

Perform complex editing tasks like style transfer, object manipulation (insertion/removal), detail enhancement, and even human pose adjustments.

![](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/s3.jpg#center)

### Image Understanding

Leverage built-in image understanding capabilities, including object detection, semantic segmentation, and depth estimation, for advanced visual manipulation.

![](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/s4.jpg#center)

## Advanced Usage

### Prompt Enhancement

For enhanced prompt optimization and multi-language support, use our official Prompt Enhancement Tool powered by Qwen-Plus.

Integrate into your code:

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

Qwen-Image supports a Multi-GPU API Server for local deployment.

### Multi-GPU API Server Pipeline & Usage

The Multi-GPU API Server offers:

*   Multi-GPU parallel processing
*   Queue management for high concurrency
*   Automatic prompt optimization
*   Support for multiple aspect ratios

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

[AI Arena](https://aiarena.alibaba-inc.com) provides an open benchmarking platform for evaluating Qwen-Image's general image generation capabilities against state-of-the-art closed-source APIs using the Elo rating system.

![AI Arena](assets/figure_aiarena_website.png)

View the latest leaderboard rankings at [AI Arena Learboard](https://aiarena.alibaba-inc.com/corpora/arena/leaderboard?arenaType=text2image). Contact weiyue.wy@alibaba-inc.com to deploy your model and participate in evaluations.

## Community Support

Explore community support from:

*   **Hugging Face:**  Diffusers supports Qwen-Image from day 0, with LoRA and finetuning workflows in development.
*   **Modelscope:** Provides comprehensive support with DiffSynth-Studio and DiffSynth-Engine, plus hands-on experiences via ModelScope AIGC Central.
*   **WaveSpeedAI:** Offers Qwen-Image on their platform.  See their [model page](https://wavespeed.ai/models/wavespeed-ai/qwen-image/text-to-image).
*   **LiblibAI:**  Offers native support. Visit their [community](https://www.liblib.art/modelinfo/c62a103bd98a4246a2334e2d952f7b21?from=sd&versionUuid=75e0be0c93b34dd8baeec9c968013e0c) page for details.

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

Connect with the research team on [Discord](https://discord.gg/z3GAxXZ9Ce) or via our [WeChat groups](assets/wechat.png). We welcome your issues, pull requests, and collaboration.  For full-time positions and research internships, contact fulai.hr@alibaba-inc.com.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=QwenLM/Qwen-Image&type=Date)](https://www.star-history.com/#QwenLM/Qwen-Image&Date)