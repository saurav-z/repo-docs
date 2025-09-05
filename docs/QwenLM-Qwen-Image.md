<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/qwen_image_logo.png" width="400"/>
<p>

<p align="center">
    💜 <a href="https://chat.qwen.ai/">Qwen Chat</a> |
    🤗 <a href="https://huggingface.co/Qwen/Qwen-Image">HuggingFace(T2I)</a> |
    🤗 <a href="https://huggingface.co/Qwen/Qwen-Image-Edit">HuggingFace(Edit)</a> | 🤖 <a href="https://modelscope.cn/models/Qwen/Qwen-Image">ModelScope-T2I</a> | 🤖 <a href="https://modelscope.cn/models/Qwen/Qwen-Image-Edit">ModelScope-Edit</a> | 📑 <a href="https://arxiv.org/abs/2508.02324">Tech Report</a> | 📑 <a href="https://qwenlm.github.io/blog/qwen-image/">Blog(T2I)</a> | 📑 <a href="https://qwenlm.github.io/blog/qwen-image-edit/">Blog(Edit)</a>
<br>
🖥️ <a href="https://huggingface.co/spaces/Qwen/Qwen-Image">T2I Demo</a> | 🖥️ <a href="https://huggingface.co/spaces/Qwen/Qwen-Image-Edit">Edit Demo</a> | 💬 <a href="https://github.com/QwenLM/Qwen-Image/blob/main/assets/wechat.png">WeChat (微信)</a> | 🫨 <a href="https://discord.gg/CV4E9rpNSD">Discord</a>
</p>

## Qwen-Image: Unleash Your Creative Vision with Cutting-Edge Image Generation and Editing

**Qwen-Image, a powerful 20B MMDiT image foundation model, excels at complex text rendering and precise image editing, offering unparalleled creative control.**  [See the original repository here](https://github.com/QwenLM/Qwen-Image).

### Key Features

*   **Exceptional Text Rendering:** Generate images with accurate and detailed text, including support for English and Chinese.
*   **Advanced Image Editing:**  Perform style transfer, object manipulation, and intricate detail enhancement with ease.
*   **Image Understanding Capabilities:** Leverage built-in features like object detection and semantic segmentation for intelligent image manipulation.
*   **User-Friendly Demos:**  Explore the capabilities of Qwen-Image through interactive demos on [Hugging Face](https://huggingface.co/spaces/Qwen/Qwen-Image) and [Qwen Chat](https://chat.qwen.ai/).
*   **Community Support:** Benefit from extensive community support via Hugging Face, ModelScope, WaveSpeedAI, LiblibAI, and cache-dit.

### News and Updates

*   **2025.08.19:**  Updated recommendations for using the latest diffusers commits for Qwen-Image-Edit, for the best results.
*   **2025.08.18:** Qwen-Image-Edit is open-sourced!
*   **2025.08.09:**  Support for LoRA models like MajicBeauty LoRA is available for generating realistic beauty images.
*   **2025.08.05:** Native support in ComfyUI & Qwen Chat integration. Technical Report released on Arxiv.
*   **2025.08.04:** Released Qwen-Image weights on Hugging Face and ModelScope.

### Quick Start

1.  **Prerequisites:**  Ensure `transformers>=4.51.3` and the latest `diffusers` version are installed.
    ```bash
    pip install git+https://github.com/huggingface/diffusers
    ```

2.  **Text to Image Example:**

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
        "en": ", Ultra HD, 4K, cinematic composition.",
        "zh": ", 超清，4K，电影级构图."
    }

    # Generate image
    prompt = '''A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup," with a neon light beside it displaying "通义千问". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the poster is written "π≈3.1415926-53589793-23846264-33832795-02384197".'''
    negative_prompt = " "

    # Aspect ratios
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

### Advanced Usage

*   **Prompt Enhancement for Text-to-Image:**
    ```python
    from tools.prompt_utils import rewrite
    prompt = rewrite(prompt)
    ```
    or run:
    ```bash
    cd src
    DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx python examples/generate_w_prompt_enhance.py
    ```

*   **Prompt Enhancement for Image Edit:**
    ```python
    from tools.prompt_utils import polish_edit_prompt
    prompt = polish_edit_prompt(prompt, pil_image)
    ```

### Deploy Qwen-Image

Qwen-Image supports Multi-GPU API Server for local deployment.
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

Qwen-Image is evaluated on the open benchmarking platform [AI Arena](https://aiarena.alibaba-inc.com).  View the latest leaderboard at [AI Arena Learboard](https://aiarena.alibaba-inc.com/corpora/arena/leaderboard?arenaType=text2image).  Contact weiyue.wy@alibaba-inc.com to deploy your model.

### License

Qwen-Image is licensed under the Apache 2.0 License.

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

### Get Involved

*   Join our [Discord](https://discord.gg/z3GAxXZ9Ce).
*   Connect via [WeChat groups](assets/wechat.png).
*   Report issues and contribute via pull requests on [GitHub](https://github.com/QwenLM/Qwen-Image).
*   Contact fulai.hr@alibaba-inc.com if you're interested in FTE or research intern positions.
*  Star History:
[![Star History Chart](https://api.star-history.com/svg?repos=QwenLM/Qwen-Image&type=Date)](https://www.star-history.com/#QwenLM/Qwen-Image&Date)