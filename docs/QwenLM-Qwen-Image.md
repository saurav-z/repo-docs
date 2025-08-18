<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/qwen_image_logo.png" width="400" alt="Qwen-Image Logo"/>
<p>
<p align="center">
    &nbsp&nbsp💜 <a href="https://chat.qwen.ai/">Qwen Chat</a>&nbsp&nbsp |
    &nbsp&nbsp🤗 <a href="https://huggingface.co/Qwen/Qwen-Image">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/models/Qwen/Qwen-Image">ModelScope</a>&nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://arxiv.org/abs/2508.02324">Tech Report</a> &nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://qwenlm.github.io/blog/qwen-image/">Blog</a> &nbsp&nbsp
<br>
🖥️ <a href="https://huggingface.co/spaces/Qwen/Qwen-Image">Demo</a>&nbsp&nbsp | &nbsp&nbsp💬 <a href="https://github.com/QwenLM/Qwen-Image/blob/main/assets/wechat.png">WeChat (微信)</a>&nbsp&nbsp | &nbsp&nbsp🫨 <a href="https://discord.gg/CV4E9rpNSD">Discord</a>&nbsp&nbsp
</p>

## Qwen-Image: Unleash Stunning Visuals with Advanced Text Rendering and Image Editing

**Qwen-Image is a powerful 20B parameter MMDiT image foundation model, revolutionizing image generation and editing with exceptional text rendering and versatile capabilities.** ([See the original repo](https://github.com/QwenLM/Qwen-Image))

<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/merge3.jpg" width="1024" alt="Qwen-Image Example"/>
<p>

### Key Features

*   **Exceptional Text Rendering:** Generate images with unparalleled accuracy in complex text rendering, supporting English and Chinese, maintaining typographic details, layout coherence, and contextual harmony.
*   **Versatile Image Generation:** Create diverse artistic styles, from photorealistic scenes to anime, adapting fluidly to creative prompts.
*   **Advanced Image Editing:** Perform style transfer, object manipulation (insertion/removal), detail enhancement, and even human pose manipulation with ease.
*   **Image Understanding Capabilities:** Includes object detection, semantic segmentation, depth and edge estimation, and more.
*   **Multi-Language Support:** Optimized for both English and Chinese prompts.
*   **Integration & Support:**  Seamlessly integrates with Hugging Face, ModelScope, WaveSpeedAI, and LiblibAI, and offers a Multi-GPU API server for local deployment.
*   **AI Arena Evaluation**: Publicly available benchmarking on the AI Arena platform to compare Qwen-Image with other models.

### News

*   **2025.08.09**: Qwen-Image now supports LoRA models such as MajicBeauty, enabling the generation of realistic beauty images.
*   **2025.08.05**: Native support in ComfyUI and Qwen Chat.
*   **2025.08.05**: Technical Report released on Arxiv.
*   **2025.08.04**: Model weights released on Hugging Face and ModelScope.
*   **2025.08.04**: Qwen-Image released! Check our [Blog](https://qwenlm.github.io/blog/qwen-image) for more details!

> [!NOTE]
> Qwen-Image-Edit is coming soon!

### Quick Start

1.  Make sure your transformers is >=4.51.3.
2.  Install the latest version of diffusers:

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

### Show Cases

[Include images (s1.jpg, s2.jpg, s3.jpg, s4.jpg) here with alt text.  Keep it concise and impactful.  Example:]

<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/s1.jpg#center" alt="Qwen-Image Text Rendering Example"/>
<p>

### Advanced Usage

#### Prompt Enhance
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

To comprehensively evaluate the general image generation capabilities of Qwen-Image and objectively compare it with state-of-the-art closed-source APIs, we introduce [AI Arena](https://aiarena.alibaba-inc.com), an open benchmarking platform built on the Elo rating system. AI Arena provides a fair, transparent, and dynamic environment for model evaluation.

In each round, two images—generated by randomly selected models from the same prompt—are anonymously presented to users for pairwise comparison. Users vote for the better image, and the results are used to update both personal and global leaderboards via the Elo algorithm, enabling developers, researchers, and the public to assess model performance in a robust and data-driven way. AI Arena is now publicly available, welcoming everyone to participate in model evaluations.

![AI Arena](assets/figure_aiarena_website.png)

The latest leaderboard rankings can be viewed at [AI Arena Learboard](https://aiarena.alibaba-inc.com/corpora/arena/leaderboard?arenaType=text2image).

If you wish to deploy your model on AI Arena and participate in the evaluation, please contact weiyue.wy@alibaba-inc.com.

### Community Support

*   **Hugging Face:** Diffusers supports Qwen-Image, with LoRA and finetuning workflows in development.
*   **ModelScope:** (DiffSynth-Studio, DiffSynth-Engine)
*   **WaveSpeedAI:** Deployed on their platform.
*   **LiblibAI:** Native support.
*   **Inference Acceleration Method: cache-dit**: cache acceleration support for Qwen-Image with DBCache, TaylorSeer and Cache CFG.

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

Connect with the research team and contribute to the project.

*   Join our [Discord](https://discord.gg/z3GAxXZ9Ce).
*   Join our [WeChat groups](assets/wechat.png) by scanning the QR code.
*   Submit issues and pull requests on GitHub.
*   Reach out to fulai.hr@alibaba-inc.com for FTE and research intern positions.

### Star History

[![Star History Chart](https://api.star-history.com/svg?repos=QwenLM/Qwen-Image&type=Date)](https://www.star-history.com/#QwenLM/Qwen-Image&Date)