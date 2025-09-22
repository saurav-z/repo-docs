<p align="center">
  <img src="./assets/logo.png"  height=100 alt="HunyuanImage-2.1 Logo">
</p>

<div align="center">

# HunyuanImage-2.1: Generate Stunning 2K Images with Advanced AI

</div>

<p align="center">
  Unleash the power of AI and create breathtaking high-resolution images with **HunyuanImage-2.1**, a cutting-edge text-to-image model. 
  <br/>
  <a href="https://github.com/Tencent-Hunyuan/HunyuanImage-2.1">View the Code on GitHub</a> |
  <a href="https://huggingface.co/tencent/HunyuanImage-2.1">Hugging Face Demo</a> |
  <a href="https://hunyuan.tencent.com/modelSquare/home/play?modelId=286&from=/visual">Official Website (Try the Model!)</a>
</p>

-----

HunyuanImage-2.1 empowers you to generate incredible, high-resolution (2K) images from text descriptions. This repository provides PyTorch model definitions, pre-trained weights, and inference code, allowing you to create stunning visuals with ease.

## Key Features

*   ✨ **Ultra-High Resolution:** Generate images at 2K (2048 x 2048) resolution, perfect for detailed and visually rich outputs.
*   🌐 **Multilingual Support:**  Works seamlessly with both Chinese and English prompts, broadening creative possibilities.
*   🧠 **Advanced Architecture:** Leverages a sophisticated multi-modal DiT (Diffusion Transformer) backbone with dual-stream processing for superior image quality.
*   ✍️ **Glyph-Aware Text Rendering:** Incorporates ByT5's text rendering capabilities for accurate and aesthetically pleasing text within images.
*   📐 **Flexible Aspect Ratios:** Supports a wide range of aspect ratios, including 1:1, 16:9, 9:16, 4:3, 3:4, 3:2, and 2:3.
*   💡 **Intelligent Prompt Enhancement:** Automatically refines your prompts to maximize descriptive detail and achieve exceptional visual results.

## Latest Updates

*   **September 18, 2025:** ✨ Try the [PromptEnhancer-32B model](https://huggingface.co/PromptEnhancer/PromptEnhancer-32B) for higher-quality prompt enhancement!​.
*   **September 18, 2025:** ✨ [ComfyUI workflow of HunyuanImage-2.1](https://github.com/KimbingNg/ComfyUI-HunyuanImage2.1) is available now!
*   **September 16, 2025:** 👑 Achieved Top1 on the Arena leaderboard for text-to-image open-source models!  [Leaderboard](https://artificialanalysis.ai/text-to-image/arena/leaderboard-text)
*   **September 12, 2025:** 🚀 Released FP8 quantized models, enabling 2K image generation with just 24GB GPU memory!
*   **September 8, 2025:** 🚀 Inference code and model weights for HunyuanImage-2.1 released.

<div align="center">
  <img src="./assets/demo.jpg" width=100% alt="HunyuanImage 2.1 Demo">
</div>

## How to Get Started

### System Requirements

*   **Hardware:** NVIDIA GPU with CUDA support.  Minimum: 24 GB GPU memory for 2048x2048 image generation (measured with model CPU offloading and FP8 quantization enabled).
*   **Operating System:** Linux.

### Installation

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/Tencent-Hunyuan/HunyuanImage-2.1.git
    cd HunyuanImage-2.1
    ```

2.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    pip install flash-attn==2.7.3 --no-build-isolation
    ```

### Download Pretrained Models

Details are available [here](ckpts/checkpoints-download.md).

### Usage

1.  **Prompt Enhancement (Highly Recommended)**

    For the best results, utilize detailed and descriptive prompts. We strongly recommend using the [PromptEnhancer-32B model](https://huggingface.co/PromptEnhancer/PromptEnhancer-32B) to optimize your prompts.

2.  **Text-to-Image Generation**

    HunyuanImage-2.1 is optimized for 2K image generation (e.g., 2048x2048 for 1:1, 2560x1536 for 16:9, etc.).  Generating at 1K resolution may result in artifacts.  We recommend using the full generation pipeline (prompt enhancement and refiner) for optimal image quality.

    ```python
    import os
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    import torch
    from hyimage.diffusion.pipelines.hunyuanimage_pipeline import HunyuanImagePipeline

    # Supported model_name: hunyuanimage-v2.1, hunyuanimage-v2.1-distilled
    model_name = "hunyuanimage-v2.1"
    pipe = HunyuanImagePipeline.from_pretrained(model_name=model_name, use_fp8=True)
    pipe = pipe.to("cuda")

    # The input prompt
    prompt = "A cute, cartoon-style anthropomorphic penguin plush toy with fluffy fur, standing in a painting studio, wearing a red knitted scarf and a red beret with the word “Tencent” on it, holding a paintbrush with a focused expression as it paints an oil painting of the Mona Lisa, rendered in a photorealistic photographic style."

    # Generate with different aspect ratios
    aspect_ratios = {
        "16:9": (2560, 1536),
        "4:3": (2304, 1792),
        "1:1": (2048, 2048),
        "3:4": (1792, 2304),
        "9:16": (1536, 2560),
    }

    width, height = aspect_ratios["1:1"]

    image = pipe(
        prompt=prompt,
        width=width,
        height=height,
        # disable the reprompt if you already use the prompt enhancement to enhance the prompt
        use_reprompt=False,  # Enable prompt enhancement (which may result in higher GPU memory usage)
        use_refiner=True,   # Enable refiner model
        # For the distilled model, use 8 steps for faster inference.
        # For the non-distilled model, use 50 steps for better quality.
        num_inference_steps=8 if "distilled" in model_name else 50,
        guidance_scale=3.25 if "distilled" in model_name else 3.5,
        shift=4 if "distilled" in model_name else 5,
        seed=649151,
    )

    image.save("generated_image.png")
    ```
## Prompt Enhanced Demo

<p align="center">
  <img src="./assets/reprompt.jpg" width=100% alt="Human Evaluation with Other Models">
</p>

## Performance Benchmarks

### SSAE Evaluation

<p align="center">
<table>
<thead>
<tr>
    <th rowspan="2">Model</th>  <th rowspan="2">Open Source</th> <th rowspan="2">Mean Image Accuracy</th> <th rowspan="2">Global Accuracy</th> <th colspan="4" style="text-align: center;">Primary Subject</th> <th colspan="3" style="text-align: center;">Secondary Subject</th> <th colspan="2" style="text-align: center;">Scene</th> <th colspan="3" style="text-align: center;">Other</th>
</tr>
<tr>
    <th>Noun</th> <th>Key Attributes</th> <th>Other Attributes</th> <th>Action</th> <th>Noun</th> <th>Attributes</th> <th>Action</th> <th>Noun</th> <th>Attributes</th> <th>Shot</th> <th>Style</th> <th>Composition</th>
</tr>
</thead>
<tbody>
<tr>
    <td>FLUX-dev</td> <td>✅</td> <td>0.7122</td> <td>0.6995</td> <td>0.7965</td> <td>0.7824</td> <td>0.5993</td> <td>0.5777</td> <td>0.7950</td> <td>0.6826</td> <td>0.6923</td> <td>0.8453</td> <td>0.8094</td> <td>0.6452</td> <td>0.7096</td> <td>0.6190</td>
</tr>
<tr>
    <td>Seedream-3.0</td> <td>❌</td> <td>0.8827</td> <td>0.8792</td> <td>0.9490</td> <td>0.9311</td> <td>0.8242</td> <td>0.8177</td> <td>0.9747</td> <td>0.9103</td> <td>0.8400</td> <td>0.9489</td> <td>0.8848</td> <td>0.7582</td> <td>0.8726</td> <td>0.7619</td>
</tr>
<tr>
    <td>Qwen-Image</td> <td>✅</td> <td>0.8854</td> <td>0.8828</td> <td>0.9502</td> <td>0.9231</td> <td>0.8351</td> <td>0.8161</td> <td>0.9938</td> <td>0.9043</td> <td>0.8846</td> <td>0.9613</td> <td>0.8978</td> <td>0.7634</td> <td>0.8548</td> <td>0.8095</td>
</tr>
<tr>
    <td>GPT-Image</td>  <td>❌</td> <td> 0.8952</td> <td>0.8929</td> <td>0.9448</td> <td>0.9289</td> <td>0.8655</td> <td>0.8445</td> <td>0.9494</td> <td>0.9283</td> <td>0.8800</td> <td>0.9432</td> <td>0.9017</td> <td>0.7253</td> <td>0.8582</td> <td>0.7143</td>
</tr>
<tr>
    <td><strong>HunyuanImage 2.1</strong></td> <td>✅</td> <td><strong>0.8888</strong></td> <td><strong>0.8832</strong></td> <td>0.9339</td> <td>0.9341</td> <td>0.8363</td> <td>0.8342</td> <td>0.9627</td> <td>0.8870</td> <td>0.9615</td> <td>0.9448</td> <td>0.9254</td> <td>0.7527</td> <td>0.8689</td> <td>0.7619</td>
</tr>
</tbody>
</table>
</p>

### GSB Evaluation

<p align="center">
  <img src="./assets/gsb.png" width=70% alt="Human Evaluation with Other Models">
</p>

## Contact & Community

*   Join our [Discord server](https://discord.gg/ehjWMqF5wY) and [WeChat groups](assets/WECHAT.md) to connect, collaborate, and ask questions.
*   Contribute to the project by opening issues or submitting pull requests on [GitHub](https://github.com/Tencent-Hunyuan/HunyuanImage-2.1).

## BibTeX

```bibtex
@misc{HunyuanImage-2.1,
  title={HunyuanImage 2.1: An Efficient Diffusion Model for High-Resolution (2K) Text-to-Image Generation},
  author={Tencent Hunyuan Team},
  year={2025},
  howpublished={\url{https://github.com/Tencent-Hunyuan/HunyuanImage-2.1}},
}
```

## Acknowledgements

We appreciate the contributions of [Qwen](https://huggingface.co/Qwen), [FLUX](https://github.com/black-forest-labs/flux), [diffusers](https://github.com/huggingface/diffusers), and [HuggingFace](https://huggingface.co) to open-source research.

## GitHub Star History

<a href="https://star-history.com/#Tencent-Hunyuan/HunyuanImage-2.1&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=Tencent-Hunyuan/HunyuanImage-2.1&type=Date1&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=Tencent-Hunyuan/HunyuanImage-2.1&type=Date1" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=Tencent-Hunyuan/HunyuanImage-2.1&type=Date1" />
 </picture>
</a>