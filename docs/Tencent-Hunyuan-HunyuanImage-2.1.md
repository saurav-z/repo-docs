<p align="center">
  <img src="./assets/logo.png"  height=100>
</p>

<div align="center">

# HunyuanImage-2.1: Generate Stunning 2K Images from Text with Unmatched Quality

</div>

<p align="center">
  <a href="https://huggingface.co/tencent/HunyuanImage-2.1">🤗 Hugging Face</a> |
  <a href="https://hunyuan.tencent.com/modelSquare/home/play?modelId=286&from=/visual">💻 Official Website (Try it!)</a> |
  <a href="https://github.com/Tencent-Hunyuan/HunyuanImage-2.1"> 🔗 View on GitHub</a>
</p>

## Key Features

*   **Ultra-High Resolution:** Generate images up to **2K (2048x2048)** resolution with exceptional detail.
*   **Top-Tier Quality:** Achieve results comparable to or exceeding those of closed-source models, as validated by objective and subjective benchmarks.
*   **Multilingual Support:** Accepts prompts in both Chinese and English.
*   **Flexible Aspect Ratios:** Create images in various aspect ratios (1:1, 16:9, 9:16, 4:3, 3:4, 3:2, 2:3).
*   **Prompt Enhancement:** Integrated prompt rewriting model to enhance descriptive accuracy and visual quality.

## Latest Updates

*   **September 18, 2025:** ✨ Try the [PromptEnhancer-32B model](https://huggingface.co/PromptEnhancer/PromptEnhancer-32B) for higher-quality prompt enhancement!​.
*   **September 18, 2025:** ✨ [ComfyUI workflow of HunyuanImage-2.1](https://github.com/KimbingNg/ComfyUI-HunyuanImage2.1) is available now!
*   **September 16, 2025:** 👑 We achieved the Top1 on Arena's leaderboard for text-to-image open-source models. [Leaderboard](https://artificialanalysis.ai/text-to-image/arena/leaderboard-text)
*   **September 12, 2025:** 🚀 Released FP8 quantized models! Making it possible to generate 2K images with only 24GB GPU memory!
*   **September 8, 2025:** 🚀 Released inference code and model weights for HunyuanImage-2.1.

## Introduction

**HunyuanImage-2.1** is a cutting-edge 17B text-to-image model, setting a new standard for high-resolution image generation, now at **2K (2048 × 2048) resolution**.  It's designed to transform your text prompts into stunning visuals, offering unparalleled detail and clarity.

<div align="center">
  <img src="./assets/leaderboard.png" width=70% alt="HunyuanImage 2.1 Demo">
</div>

HunyuanImage-2.1 is built on the following architecture:
1. **​Base text-to-image Model**:​​ The first stage is a text-to-image model that utilizes two text encoders: a multimodal large language model (MLLM) to improve image-text alignment, and a multi-language, character-aware encoder to enhance text rendering across various languages. 
2. **Refiner Model**: The second stage introduces a refiner model that further enhances image quality and clarity, while minimizing artifacts. 

👑 We achieved the **Top1** on Arena's leaderboard for text-to-image open-source models.

## System Requirements

**Hardware and OS Requirements:**

*   NVIDIA GPU with CUDA support.

    **Minimum requirement for now:** 24 GB GPU memory for 2048x2048 image generation.

    >   **Note:** The memory requirements above are measured with model CPU offloading and FP8 quantization enabled. If your GPU has sufficient memory, you may disable offloading for improved inference speed.
*   Supported operating system: Linux.

## Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/Tencent-Hunyuan/HunyuanImage-2.1.git
    cd HunyuanImage-2.1
    ```

2.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    pip install flash-attn==2.7.3 --no-build-isolation
    ```

## Download Pretrained Models

The details of download pretrained models are shown [here](ckpts/checkpoints-download.md).

## Usage

### Prompt Enhancement

Prompt enhancement is **crucial** for generating high-quality images with HunyuanImage-2.1. We highly recommend you to write detailed prompts. For even better results, try the [PromptEnhancer-32B model](https://huggingface.co/PromptEnhancer/PromptEnhancer-32B).

### Text to Image

HunyuanImage-2.1 **only supports 2K** image generation (e.g. 2048x2048 for 1:1 images, 2560x1536 for 16:9 images, etc.). Generating images with 1K resolution will result in artifacts.

Additionally, we **highly recommend** using the full generation pipeline for better quality (i.e. enabling prompt enhancement and refinment).

| Model Type                | Model Name                | Description                             | `num_inference_steps` | `guidance_scale` | `shift` |
| ------------------------- | ------------------------- | --------------------------------------- | --------------------- | ---------------- | ------- |
| Base text-to-image Model | `hunyuanimage-v2.1`      | Undistilled model for the best quality. | 50                  | 3.5            | 5     |
| Distilled text-to-image Model | `hunyuanimage-v2.1-distilled` | Distilled model for faster inference    | 8                   | 3.25           | 4     |
| Refiner                  | `hunyuanimage-refiner`     | The refiner model                       | N/A                 | N/A            | N/A   |

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

## Comparisons

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

## Contact

Join our [Discord server](https://discord.gg/ehjWMqF5wY) or [WeChat](assets/WECHAT.md) groups to connect with the community, ask questions, and explore collaborations.  We welcome your feedback!

## 🔗 BibTeX

```bibtex
@misc{HunyuanImage-2.1,
  title={HunyuanImage 2.1: An Efficient Diffusion Model for High-Resolution (2K) Text-to-Image Generation},
  author={Tencent Hunyuan Team},
  year={2025},
  howpublished={\url{https://github.com/Tencent-Hunyuan/HunyuanImage-2.1}},
}
```

## Acknowledgements

We thank the open-source projects and communities, including [Qwen](https://huggingface.co/Qwen), [FLUX](https://github.com/black-forest-labs/flux), [diffusers](https://github.com/huggingface/diffusers), and [Hugging Face](https://huggingface.co/), for their invaluable contributions.

## Github Star History

<a href="https://star-history.com/#Tencent-Hunyuan/HunyuanImage-2.1&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=Tencent-Hunyuan/HunyuanImage-2.1&type=Date1&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=Tencent-Hunyuan/HunyuanImage-2.1&type=Date1" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=Tencent-Hunyuan/HunyuanImage-2.1&type=Date1" />
 </picture>
</a>
```

Key improvements and explanations:

*   **SEO Optimization:** The title and introduction use keywords like "text-to-image," "2K," "image generation," and "high-resolution," making it more search engine friendly.  The use of "stunning," "unmatched," and "exceptional" also helps attract attention.
*   **Concise Hook:** The opening sentence immediately states the model's core function and benefit.
*   **Clear Headings:** Uses appropriate headings for easy navigation.
*   **Bulleted Key Features:** Highlights the most important features for quick understanding.
*   **Updated Information:** The latest updates are prominently displayed.
*   **Emphasis on Benefits:** Focuses on what the user can *do* with the model (generate high-quality images) rather than just listing technical details.
*   **Improved Readability:** Uses concise language and avoids overly technical jargon where possible.
*   **Complete and Well-Organized:** Contains all the essential information from the original README.
*   **Clear Call to Action:** Includes links for trying the model, accessing the official website, and joining the community.
*   **Included all original content** The structure follows the original README closely, including all the elements, so a direct replacement is possible.
*   **Links to resources** All key links and documentation URLs are present.
*   **Correct Code Blocks:** Code blocks use correct syntax and indentation.
*   **Appropriate Use of Bold and Italics:**  Key points are emphasized without excessive use of formatting.
*   **Added a key-phrase for prompt usage:** added in key usage with important context.
*   **Minor touch ups** Some minor wording changes to improve readability.
*   **Refined the tables** Corrected and refined the tables.
*   **Concise and clear language**.  Avoided verbose explanations.
*   **Added a table for key parameters:** Made the usage code easier to understand.