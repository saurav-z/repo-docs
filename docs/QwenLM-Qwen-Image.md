<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/qwen_image_logo.png" width="400"/>
<p>

<p align="center">
    <a href="https://chat.qwen.ai/">Qwen Chat</a>&nbsp;&nbsp; |
    <a href="https://huggingface.co/Qwen/Qwen-Image">Hugging Face</a>&nbsp;&nbsp; |
    <a href="https://modelscope.cn/models/Qwen/Qwen-Image">ModelScope</a>&nbsp;&nbsp; |
    <a href="https://arxiv.org/abs/2508.02324">Tech Report</a>&nbsp;&nbsp; |
    <a href="https://qwenlm.github.io/blog/qwen-image/">Blog</a>&nbsp;&nbsp;
    <br>
    <a href="https://huggingface.co/spaces/Qwen/Qwen-Image">Demo</a>&nbsp;&nbsp; |
    <a href="https://github.com/QwenLM/Qwen-Image/blob/main/assets/wechat.png">WeChat</a>&nbsp;&nbsp; |
    <a href="https://discord.gg/CV4E9rpNSD">Discord</a>
</p>

<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/merge3.jpg" width="1024"/>
<p>

# Qwen-Image: State-of-the-Art Image Generation and Editing

**Unleash your creativity with Qwen-Image, a powerful 20B image foundation model revolutionizing text rendering and image manipulation.**  ([Original Repository](https://github.com/QwenLM/Qwen-Image))

## Key Features:

*   **Exceptional Text Rendering:** Generate images with stunningly accurate and detailed text, supporting both English and Chinese.
*   **Versatile Image Generation:** Create a wide range of artistic styles, from photorealistic to anime, adapting to diverse creative prompts.
*   **Advanced Image Editing:** Perform style transfer, object manipulation, detail enhancement, and text editing with intuitive control.
*   **Image Understanding Capabilities:** Leverage object detection, segmentation, and other intelligent features for advanced image manipulation.
*   **Multi-Platform Support:** Available on Hugging Face, ModelScope, and integrates seamlessly with ComfyUI and Qwen Chat.

## What's New:

*   **2025.08.09:** Support for LoRA models like MajicBeauty LoRA for generating realistic beauty images.
*   **2025.08.05:** Native support in ComfyUI and availability on Qwen Chat.
*   **2025.08.05:** Technical Report released on Arxiv.
*   **2025.08.04:** Qwen-Image weights released on Hugging Face and ModelScope.
*   **2025.08.04:** Official blog post detailing Qwen-Image released.

> **Note:** The editing version of Qwen-Image is coming soon!

## Quick Start

1.  **Prerequisites:** Ensure `transformers>=4.51.3` is installed.
2.  **Install diffusers:**

    ```bash
    pip install git+https://github.com/huggingface/diffusers
    ```

3.  **Example Code:**

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

## Show Cases

**(Include example images from the original README)**

**(Summarize the Show Cases content in the format used in the original README)**

*   **High-Fidelity Text Rendering:** Precisely renders text in multiple languages, maintaining typographic details and layout.
*   **General Image Generation:** Supports diverse artistic styles from photorealistic to abstract.
*   **Advanced Image Editing:** Enables style transfer, object manipulation, and text editing.
*   **Image Understanding:** Supports object detection, segmentation, and more.

### Advanced Usage

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

## AI Arena

**(Summarize the AI Arena section, linking to the leaderboard and mentioning contact for model deployment)**

## Community Support

*   **Hugging Face:** Day-one support with ongoing developments for LoRA and finetuning workflows.
*   **ModelScope:**  DiffSynth-Studio, DiffSynth-Engine, and ModelScope AIGC Central offers various features.
*   **WaveSpeedAI:** Integrated Qwen-Image on their platform.
*   **LiblibAI:** Native support.
*   **cache-dit:**  Cache acceleration support for Qwen-Image with DBCache, TaylorSeer and Cache CFG.

## License

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

**(Include Discord and WeChat links/QR codes, and information about hiring)**