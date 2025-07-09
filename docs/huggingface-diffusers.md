---
title: Diffusers: Generate Images, Audio, and More with State-of-the-Art Diffusion Models
description: Build and run cutting-edge diffusion models for image generation, audio creation, and 3D molecule structures with Hugging Face Diffusers. 
keywords: diffusion models, image generation, text-to-image, audio generation, deep learning, Hugging Face, PyTorch, AI, machine learning
---

<p align="center">
    <br>
    <img src="https://raw.githubusercontent.com/huggingface/diffusers/main/docs/source/en/imgs/diffusers_library.jpg" width="400"/>
    <br>
<p>

[![License](https://img.shields.io/github/license/huggingface/datasets.svg?color=blue)](https://github.com/huggingface/diffusers/blob/main/LICENSE)
[![GitHub release](https://img.shields.io/github/release/huggingface/diffusers.svg)](https://github.com/huggingface/diffusers/releases)
[![PyPI monthly downloads](https://static.pepy.tech/badge/diffusers/month)](https://pepy.tech/project/diffusers)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/diffuserslib.svg?style=social&label=Follow%20%40diffuserslib)](https://twitter.com/diffuserslib)

**Harness the power of diffusion models with Hugging Face's Diffusers library, your go-to resource for creating stunning AI-generated content.**

## Key Features

*   **Versatile Pipelines:** Utilize pre-built diffusion pipelines for tasks like text-to-image, image-to-image, inpainting, and more.
*   **Modular Components:** Build your own diffusion systems using interchangeable noise schedulers and pretrained models.
*   **State-of-the-Art Models:** Access a wide range of cutting-edge pretrained models for image, audio, and 3D molecule generation.
*   **Ease of Use:** Designed for usability, simplicity, and customizability, with a focus on a user-friendly experience.
*   **Optimized Performance:** Benefit from optimization guides to enhance speed and reduce memory consumption.
*   **Community Driven:** Contribute to the open-source community and join the discussion on our Discord channel.

## What is Diffusers?

🤗 Diffusers is a powerful Python library by Hugging Face, providing a comprehensive toolkit for working with diffusion models.  Whether you're a beginner or an experienced researcher, Diffusers simplifies the process of generating images, audio, and even 3D structures from text prompts or other inputs.  It offers flexible, modular components for both inference and training, empowering you to experiment and build your own cutting-edge AI models.

## Getting Started

### Installation

Install 🤗 Diffusers using pip or conda.  Ensure you have PyTorch or Flax installed.

**With pip (PyTorch):**

```bash
pip install --upgrade diffusers[torch]
```

**With conda (PyTorch):**

```sh
conda install -c conda-forge diffusers
```

**Apple Silicon (M1/M2) Support:** Refer to the [Apple Silicon guide](https://huggingface.co/docs/diffusers/optimization/mps) for optimization.

### Quickstart

Generate an image from text with just a few lines of code:

```python
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipeline.to("cuda")
pipeline("An image of a squirrel in Picasso style").images[0]
```

**Explore the library's components to build custom diffusion systems:**

```python
from diffusers import DDPMScheduler, UNet2DModel
from PIL import Image
import torch

scheduler = DDPMScheduler.from_pretrained("google/ddpm-cat-256")
model = UNet2DModel.from_pretrained("google/ddpm-cat-256").to("cuda")
scheduler.set_timesteps(50)

sample_size = model.config.sample_size
noise = torch.randn((1, 3, sample_size, sample_size), device="cuda")
input = noise

for t in scheduler.timesteps:
    with torch.no_grad():
        noisy_residual = model(input, t).sample
        prev_noisy_sample = scheduler.step(noisy_residual, t, input).prev_sample
        input = prev_noisy_sample

image = (input / 2 + 0.5).clamp(0, 1)
image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
image = Image.fromarray((image * 255).round().astype("uint8"))
image
```

Dive into the [Quickstart](https://huggingface.co/docs/diffusers/quicktour) to kickstart your diffusion model journey!

## Documentation

Explore the documentation for a comprehensive understanding of the library:

| Documentation                                                                | What You'll Learn                                                                                                                                                                                                                            |
| :--------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [Tutorial](https://huggingface.co/docs/diffusers/tutorials/tutorial_overview) | Learn how to use the library's features for building your own diffusion system.                                                                                                                                                   |
| [Loading](https://huggingface.co/docs/diffusers/using-diffusers/loading)    | Guides for how to load and configure all the components (pipelines, models, and schedulers) of the library, as well as how to use different schedulers.                                                                                |
| [Pipelines for inference](https://huggingface.co/docs/diffusers/using-diffusers/overview_techniques) | Guides for how to use pipelines for different inference tasks, batched generation, controlling generated outputs and randomness, and how to contribute a pipeline to the library.                                   |
| [Optimization](https://huggingface.co/docs/diffusers/optimization/fp16)      | Guides for how to optimize your diffusion model to run faster and consume less memory.                                                                                                                                              |
| [Training](https://huggingface.co/docs/diffusers/training/overview)        | Guides for how to train a diffusion model for different tasks with different training techniques.                                                                                                                                           |

## Contribution

We welcome contributions! Review the [Contribution guide](https://github.com/huggingface/diffusers/blob/main/CONTRIBUTING.md) for details on how to contribute.  Explore [issues](https://github.com/huggingface/diffusers/issues) labeled as "Good first issue" or "New pipeline/model" to get started.

Join the discussion and connect with the community on our public [Discord channel](https://discord.gg/G7tWnz98XR)!

## Popular Tasks & Pipelines

| Task                          | Pipeline                                                                                       | 🤗 Hub                                                                                                 |
| :---------------------------- | :--------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------- |
| Unconditional Image Generation | [DDPM](https://huggingface.co/docs/diffusers/api/pipelines/ddpm)                              | [google/ddpm-ema-church-256](https://huggingface.co/google/ddpm-ema-church-256)                          |
| Text-to-Image                 | [Stable Diffusion Text-to-Image](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img) | [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) |
| Text-to-Image                 | [unCLIP](https://huggingface.co/docs/diffusers/api/pipelines/unclip)                          | [kakaobrain/karlo-v1-alpha](https://huggingface.co/kakaobrain/karlo-v1-alpha)                          |
| Text-to-Image                 | [DeepFloyd IF](https://huggingface.co/docs/diffusers/api/pipelines/deepfloyd_if)                | [DeepFloyd/IF-I-XL-v1.0](https://huggingface.co/DeepFloyd/IF-I-XL-v1.0)                                |
| Text-to-Image                 | [Kandinsky](https://huggingface.co/docs/diffusers/api/pipelines/kandinsky)                    | [kandinsky-community/kandinsky-2-2-decoder](https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder) |
| Text-guided Image-to-Image    | [ControlNet](https://huggingface.co/docs/diffusers/api/pipelines/controlnet)                  | [lllyasviel/sd-controlnet-canny](https://huggingface.co/lllyasviel/sd-controlnet-canny)                  |
| Text-guided Image-to-Image    | [InstructPix2Pix](https://huggingface.co/docs/diffusers/api/pipelines/pix2pix)              | [timbrooks/instruct-pix2pix](https://huggingface.co/timbrooks/instruct-pix2pix)                          |
| Text-guided Image-to-Image    | [Stable Diffusion Image-to-Image](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/img2img) | [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) |
| Text-guided Image Inpainting  | [Stable Diffusion Inpainting](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/inpaint) | [runwayml/stable-diffusion-inpainting](https://huggingface.co/runwayml/stable-diffusion-inpainting)      |
| Image Variation               | [Stable Diffusion Image Variation](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/image_variation) | [lambdalabs/sd-image-variations-diffusers](https://huggingface.co/lambdalabs/sd-image-variations-diffusers)   |
| Super Resolution              | [Stable Diffusion Upscale](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/upscale)   | [stabilityai/stable-diffusion-x4-upscaler](https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler)    |
| Super Resolution              | [Stable Diffusion Latent Upscale](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/latent_upscale) | [stabilityai/sd-x2-latent-upscaler](https://huggingface.co/stabilityai/sd-x2-latent-upscaler)              |

## Libraries Using Diffusers

(List of popular libraries using Diffusers as in original README, truncated for brevity)

*   TaskMatrix
*   InvokeAI
*   InstantID
*   ...and over 14,000 other repositories!

## Credits

(Credits as in original README)

## Citation

```bibtex
@misc{von-platen-etal-2022-diffusers,
  author = {Patrick von Platen and Suraj Patil and Anton Lozhkov and Pedro Cuenca and Nathan Lambert and Kashif Rasul and Mishig Davaadorj and Dhruv Nair and Sayak Paul and William Berman and Yiyi Xu and Steven Liu and Thomas Wolf},
  title = {Diffusers: State-of-the-art diffusion models},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/huggingface/diffusers}}
}
```

For more information, check out the [Diffusers GitHub repository](https://github.com/huggingface/diffusers).