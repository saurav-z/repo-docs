# Qwen-Image: Unleash Stunning Visual Creation with a Powerful Image Foundation Model

**Qwen-Image** is a state-of-the-art 20B parameter image foundation model revolutionizing image generation and editing, with exceptional text rendering capabilities. [Explore the original repo](https://github.com/QwenLM/Qwen-Image) to learn more!

## Key Features

*   **Unmatched Text Rendering:** Generate images with exceptionally clear and accurate text, especially supporting Chinese and English.
*   **Versatile Image Generation:** Create a wide range of images from photorealistic scenes to artistic styles (anime, paintings, etc.) with ease.
*   **Advanced Image Editing:** Perform sophisticated edits like style transfer, object manipulation, and human pose adjustments.
*   **Comprehensive Image Understanding:** Supports object detection, segmentation, and more for intelligent editing capabilities.
*   **Multi-Language Support**: Built-in support for English and Chinese, with a tool to enhance prompts.

## Quick Start

1.  **Requirements:** Ensure you have `transformers>=4.51.3` and the latest version of `diffusers`.
2.  **Install `diffusers`:**

    ```bash
    pip install git+https://github.com/huggingface/diffusers
    ```

3.  **Generate Images with a Python Snippet:**

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

## Showcases

**(Include example images and short descriptions here. The original README provides good examples.)**

## Advanced Usage

### Prompt Enhancement

For optimized prompts:

```python
from tools.prompt_utils import rewrite
prompt = rewrite(prompt)
```

Or run from the command line:

```bash
cd src
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx python examples/generate_w_prompt_enhance.py
```

## Deploy Qwen-Image

### Multi-GPU API Server

Configure with environment variables, then run:

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

Evaluate and compare Qwen-Image against state-of-the-art APIs on the open benchmarking platform: [AI Arena](https://aiarena.alibaba-inc.com).

**(Include image of the AI Arena Leaderboard)**

## Community Support

*   **Hugging Face:** (Link to Hugging Face model page)
*   **ModelScope:** (Links to ModelScope resources)
*   **WaveSpeedAI:** (Link to WaveSpeedAI model page)
*   **LiblibAI:** (Link to LiblibAI community page)
*   **cache-dit:** (Link to example)

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

**(Include links to Discord, WeChat, GitHub issues, and the hiring email.)**

**(Include Star History Chart)**
```
[![Star History Chart](https://api.star-history.com/svg?repos=QwenLM/Qwen-Image&type=Date)](https://www.star-history.com/#QwenLM/Qwen-Image&Date)