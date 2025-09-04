---
title: "🤗 Diffusers: Your Toolkit for Cutting-Edge Diffusion Models"
description: "Generate images, audio, and more with state-of-the-art diffusion models using 🤗 Diffusers, a versatile and user-friendly library.  Get started today!"
keywords: diffusion models, image generation, text-to-image, audio generation, machine learning, deep learning, AI, Hugging Face, diffusers library, Stable Diffusion, DDPM, UNet, pipelines, schedulers, models
---

<p align="center">
    <br>
    <img src="https://raw.githubusercontent.com/huggingface/diffusers/main/docs/source/en/imgs/diffusers_library.jpg" width="400"/>
    <br>
<p>

<p align="center">
    <a href="https://github.com/huggingface/diffusers/blob/main/LICENSE"><img alt="GitHub" src="https://img.shields.io/github/license/huggingface/datasets.svg?color=blue"></a>
    <a href="https://github.com/huggingface/diffusers/releases"><img alt="GitHub release" src="https://img.shields.io/github/release/huggingface/diffusers.svg"></a>
    <a href="https://pepy.tech/project/diffusers"><img alt="GitHub release" src="https://static.pepy.tech/badge/diffusers/month"></a>
    <a href="CODE_OF_CONDUCT.md"><img alt="Contributor Covenant" src="https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg"></a>
    <a href="https://twitter.com/diffuserslib"><img alt="X account" src="https://img.shields.io/twitter/url/https/twitter.com/diffuserslib.svg?style=social&label=Follow%20%40diffuserslib"></a>
</p>

**🤗 Diffusers empowers you to create stunning AI-generated content using the latest diffusion models.**  This library provides a modular and accessible framework for both beginners and experts, allowing you to explore the exciting world of generative AI.  [Check out the original repository](https://github.com/huggingface/diffusers) for more details.

## Key Features

*   **State-of-the-art Pipelines:** Quickly generate images, audio, and more with pre-built pipelines, ready to use with just a few lines of code.
*   **Flexible Schedulers:** Experiment with different noise schedulers to control diffusion speed and output quality, fine-tuning your results.
*   **Modular Models:** Utilize pre-trained models as building blocks, combined with schedulers, to design custom end-to-end diffusion systems.
*   **Usability-Focused Design:**  Built with a focus on user experience, making it easy to get started and experiment.
*   **Extensive Model Hub Integration:** Access a vast library of pre-trained models on the Hugging Face Hub, with thousands of checkpoints to choose from.
*   **Comprehensive Documentation:** Get clear explanations and guides to help you work with the library.

## Installation

Install 🤗 Diffusers in a virtual environment using pip or conda:

### PyTorch

```bash
pip install --upgrade diffusers[torch]
```

### Conda

```sh
conda install -c conda-forge diffusers
```

### Apple Silicon (M1/M2) Support

Refer to the [How to use Stable Diffusion in Apple Silicon](https://huggingface.co/docs/diffusers/optimization/mps) guide.

## Quickstart

Generate an image from text using a pre-trained model:

```python
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipeline.to("cuda")
pipeline("An image of a squirrel in Picasso style").images[0]
```

Build your own diffusion system:

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

Explore the [Quickstart](https://huggingface.co/docs/diffusers/quicktour) to begin your diffusion journey.

## Documentation Navigation

| Documentation                                                                | What You Can Learn                                                                                                                                                         |
| :--------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Tutorial](https://huggingface.co/docs/diffusers/tutorials/tutorial_overview) | Learn the basics of using models, schedulers, and training your own diffusion models.                                                                                    |
| [Loading](https://huggingface.co/docs/diffusers/using-diffusers/loading)    | Learn how to load and configure pipelines, models, and schedulers.                                                                                                       |
| [Pipelines for Inference](https://huggingface.co/docs/diffusers/using-diffusers/overview_techniques)   | Learn about inference tasks, batched generation, controlling output, and contributing your own pipelines.              |
| [Optimization](https://huggingface.co/docs/diffusers/optimization/fp16)  | Guides for optimizing diffusion models to reduce running time and memory consumption.   |
| [Training](https://huggingface.co/docs/diffusers/training/overview) | Guides for training diffusion models for different tasks with various training techniques.                                                                |


## Contribution

We welcome contributions!  Check out our [Contribution guide](https://github.com/huggingface/diffusers/blob/main/CONTRIBUTING.md) and explore the [issues](https://github.com/huggingface/diffusers/issues) to get involved.  Find opportunities via:

*   [Good first issues](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22)
*   [New model/pipeline](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22New+pipeline%2Fmodel%22)
*   [New scheduler](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22New+scheduler%22)

Join the discussion on our Discord channel: <a href="https://discord.gg/G7tWnz98XR"><img alt="Join us on Discord" src="https://img.shields.io/discord/823813159592001537?color=5865F2&logo=discord&logoColor=white"></a>.

## Popular Tasks & Pipelines

| Task                       | Pipeline                                                                              | 🤗 Hub                                                                                                 |
| :------------------------- | :------------------------------------------------------------------------------------ | :---------------------------------------------------------------------------------------------------- |
| Unconditional Image Generation | DDPM                                                                               | [google/ddpm-ema-church-256](https://huggingface.co/google/ddpm-ema-church-256)                         |
| Text-to-Image              | Stable Diffusion Text-to-Image                                                        | [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) |
| Text-to-Image              | unCLIP                                                                                | [kakaobrain/karlo-v1-alpha](https://huggingface.co/kakaobrain/karlo-v1-alpha)                          |
| Text-to-Image              | DeepFloyd IF                                                                          | [DeepFloyd/IF-I-XL-v1.0](https://huggingface.co/DeepFloyd/IF-I-XL-v1.0)                                |
| Text-to-Image              | Kandinsky                                                                             | [kandinsky-community/kandinsky-2-2-decoder](https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder) |
| Text-guided Image-to-Image | ControlNet                                                                          | [lllyasviel/sd-controlnet-canny](https://huggingface.co/lllyasviel/sd-controlnet-canny)               |
| Text-guided Image-to-Image | InstructPix2Pix                                                                       | [timbrooks/instruct-pix2pix](https://huggingface.co/timbrooks/instruct-pix2pix)                       |
| Text-guided Image-to-Image | Stable Diffusion Image-to-Image                                                        | [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) |
| Text-guided Image Inpainting | Stable Diffusion Inpainting                                                           | [runwayml/stable-diffusion-inpainting](https://huggingface.co/runwayml/stable-diffusion-inpainting)    |
| Image Variation            | Stable Diffusion Image Variation                                                        | [lambdalabs/sd-image-variations-diffusers](https://huggingface.co/lambdalabs/sd-image-variations-diffusers) |
| Super Resolution           | Stable Diffusion Upscale                                                              | [stabilityai/stable-diffusion-x4-upscaler](https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler) |
| Super Resolution           | Stable Diffusion Latent Upscale                                                       | [stabilityai/sd-x2-latent-upscaler](https://huggingface.co/stabilityai/sd-x2-latent-upscaler)        |

## Popular Libraries Using 🧨 Diffusers

*   [Microsoft/TaskMatrix](https://github.com/microsoft/TaskMatrix)
*   [invoke-ai/InvokeAI](https://github.com/invoke-ai/InvokeAI)
*   [InstantID/InstantID](https://github.com/InstantID/InstantID)
*   [apple/ml-stable-diffusion](https://github.com/apple/ml-stable-diffusion)
*   [Sanster/lama-cleaner](https://github.com/Sanster/lama-cleaner)
*   [IDEA-Research/Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything)
*   [ashawkey/stable-dreamfusion](https://github.com/ashawkey/stable-dreamfusion)
*   [deep-floyd/IF](https://github.com/deep-floyd/IF)
*   [bentoml/BentoML](https://github.com/bentoml/BentoML)
*   [bmaltais/kohya_ss](https://github.com/bmaltais/kohya_ss)
*   and over 14,000 other amazing GitHub repositories!

Thank you for using our library!

## Credits

This library is built upon the research of many contributors. Special thanks to:

*   @CompVis (latent diffusion models)
*   @hojonathanho and @pesser (DDPM implementation)
*   @ermongroup (DDIM implementation)
*   @yang-song (Score-VE and Score-VP implementations)
*   @heejkoo (diffusion models overview)
*   @crowsonkb and @rromb (useful discussions and insights)

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