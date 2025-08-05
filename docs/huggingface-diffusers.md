<p align="center">
    <br>
    <img src="https://raw.githubusercontent.com/huggingface/diffusers/main/docs/source/en/imgs/diffusers_library.jpg" width="400"/>
    <br>
<p>

# 🤗 Diffusers: The Ultimate Library for Diffusion Models

**Unleash the power of generative AI with 🤗 Diffusers, the leading open-source library for state-of-the-art diffusion models.  **

[Go to the original repository on GitHub](https://github.com/huggingface/diffusers)

<p align="center">
    <a href="https://github.com/huggingface/diffusers/blob/main/LICENSE"><img alt="GitHub" src="https://img.shields.io/github/license/huggingface/datasets.svg?color=blue"></a>
    <a href="https://github.com/huggingface/diffusers/releases"><img alt="GitHub release" src="https://img.shields.io/github/release/huggingface/diffusers.svg"></a>
    <a href="https://pepy.tech/project/diffusers"><img alt="GitHub release" src="https://static.pepy.tech/badge/diffusers/month"></a>
    <a href="CODE_OF_CONDUCT.md"><img alt="Contributor Covenant" src="https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg"></a>
    <a href="https://twitter.com/diffuserslib"><img alt="X account" src="https://img.shields.io/twitter/url/https/twitter.com/diffuserslib.svg?style=social&label=Follow%20%40diffuserslib"></a>
</p>

🤗 Diffusers is your go-to resource for generating images, audio, 3D structures, and more using cutting-edge diffusion models.  It provides a modular and flexible framework suitable for both quick inference and in-depth training. The library prioritizes:

*   **Usability:** Easy-to-use APIs for rapid prototyping and deployment.
*   **Simplicity:**  Clear and concise code for ease of understanding and modification.
*   **Customizability:**  Highly adaptable for advanced users who want to tweak and optimize.

## Key Features

*   **Diffusion Pipelines:** Ready-to-use pipelines for various generative tasks, allowing you to generate results with minimal code.
*   **Flexible Schedulers:**  A wide selection of noise schedulers for fine-grained control over diffusion processes, enabling you to adjust speed and output quality.
*   **Pretrained Models:** Modular building blocks, including models and schedulers, that you can combine to create custom diffusion systems.

## Installation

Install 🤗 Diffusers in a virtual environment using pip or conda. Make sure you have PyTorch and/or Flax installed.

### PyTorch

```bash
pip install --upgrade diffusers[torch]
```

### Flax

```bash
pip install --upgrade diffusers[flax]
```

### Apple Silicon (M1/M2) Support

Consult the [How to use Stable Diffusion in Apple Silicon](https://huggingface.co/docs/diffusers/optimization/mps) guide for optimal performance.

## Quickstart

Get started generating outputs in seconds!

```python
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipeline.to("cuda")
pipeline("An image of a squirrel in Picasso style").images[0]
```

Build your own diffusion systems using the model and scheduler toolbox:

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

Explore the [Quickstart](https://huggingface.co/docs/diffusers/quicktour) for an introductory guide.

## Documentation Overview

| **Documentation**                                                   | **Learn**                                                                                                                                                                                               |
|---------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Tutorial](https://huggingface.co/docs/diffusers/tutorials/tutorial_overview)                                                            | Learn to use models and schedulers to create and train your own diffusion systems.  |
| [Loading](https://huggingface.co/docs/diffusers/using-diffusers/loading)                                                             | Guides on loading and configuring pipelines, models, and schedulers, along with scheduler usage.                                |
| [Pipelines for inference](https://huggingface.co/docs/diffusers/using-diffusers/overview_techniques)                                             | Learn to use pipelines, batch generation, output control, and contribution.                                        |
| [Optimization](https://huggingface.co/docs/diffusers/optimization/fp16)                                                        | Techniques for optimizing your diffusion models for speed and efficiency.                                                                                                  |
| [Training](https://huggingface.co/docs/diffusers/training/overview) | Learn to train your models for various tasks with different techniques.                                                                                                                               |

## Contribute

We welcome contributions from the community! Check out the [Contribution guide](https://github.com/huggingface/diffusers/blob/main/CONTRIBUTING.md).

*   Explore [Good first issues](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22).
*   Contribute exciting new diffusion models and pipelines through [New model/pipeline](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22New+pipeline%2Fmodel%22).
*   Contribute new schedulers via [New scheduler](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22New+scheduler%22).

Join our Discord community for discussions and support: <a href="https://discord.gg/G7tWnz98XR"><img alt="Join us on Discord" src="https://img.shields.io/discord/823813159592001537?color=5865F2&logo=discord&logoColor=white"></a>.

## Popular Tasks & Pipelines

| Task                         | Pipeline                                                                     | 🤗 Hub                                                                                                     |
| ---------------------------- | ---------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| Unconditional Image Generation | [DDPM](https://huggingface.co/docs/diffusers/api/pipelines/ddpm)          | [google/ddpm-ema-church-256](https://huggingface.co/google/ddpm-ema-church-256)                           |
| Text-to-Image                | [Stable Diffusion Text-to-Image](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img) | [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) |
| Text-to-Image                | [unCLIP](https://huggingface.co/docs/diffusers/api/pipelines/unclip)        | [kakaobrain/karlo-v1-alpha](https://huggingface.co/kakaobrain/karlo-v1-alpha)                             |
| Text-to-Image                | [DeepFloyd IF](https://huggingface.co/docs/diffusers/api/pipelines/deepfloyd_if)        | [DeepFloyd/IF-I-XL-v1.0](https://huggingface.co/DeepFloyd/IF-I-XL-v1.0)                             |
| Text-to-Image                | [Kandinsky](https://huggingface.co/docs/diffusers/api/pipelines/kandinsky)        | [kandinsky-community/kandinsky-2-2-decoder](https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder)                             |
| Text-guided Image-to-Image   | [ControlNet](https://huggingface.co/docs/diffusers/api/pipelines/controlnet) | [lllyasviel/sd-controlnet-canny](https://huggingface.co/lllyasviel/sd-controlnet-canny)                 |
| Text-guided Image-to-Image   | [InstructPix2Pix](https://huggingface.co/docs/diffusers/api/pipelines/pix2pix) | [timbrooks/instruct-pix2pix](https://huggingface.co/timbrooks/instruct-pix2pix)                           |
| Text-guided Image-to-Image   | [Stable Diffusion Image-to-Image](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/img2img) | [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) |
| Text-guided Image Inpainting | [Stable Diffusion Inpainting](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/inpaint) | [runwayml/stable-diffusion-inpainting](https://huggingface.co/runwayml/stable-diffusion-inpainting)     |
| Image Variation              | [Stable Diffusion Image Variation](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/image_variation) | [lambdalabs/sd-image-variations-diffusers](https://huggingface.co/lambdalabs/sd-image-variations-diffusers) |
| Super Resolution             | [Stable Diffusion Upscale](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/upscale)     | [stabilityai/stable-diffusion-x4-upscaler](https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler) |
| Super Resolution             | [Stable Diffusion Latent Upscale](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/latent_upscale)     | [stabilityai/sd-x2-latent-upscaler](https://huggingface.co/stabilityai/sd-x2-latent-upscaler) |

## Used by Amazing Libraries

*   [TaskMatrix](https://github.com/microsoft/TaskMatrix)
*   [InvokeAI](https://github.com/invoke-ai/InvokeAI)
*   [InstantID](https://github.com/InstantID/InstantID)
*   [ml-stable-diffusion](https://github.com/apple/ml-stable-diffusion)
*   [lama-cleaner](https://github.com/Sanster/lama-cleaner)
*   [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything)
*   [stable-dreamfusion](https://github.com/ashawkey/stable-dreamfusion)
*   [IF](https://github.com/deep-floyd/IF)
*   [BentoML](https://github.com/bentoml/BentoML)
*   [kohya_ss](https://github.com/bmaltais/kohya_ss)
*   ... and +14,000 other GitHub repositories!

Thank you for being a part of the 🤗 Diffusers community!

## Credits

This library builds upon the work of many researchers and developers. We acknowledge and thank the following for their valuable contributions:

*   @CompVis' latent diffusion models library
*   @hojonathanho's original DDPM implementation
*   @ermongroup's DDIM implementation
*   @yang-song's Score-VE and Score-VP implementations

Special thanks to @heejkoo and @crowsonkb/ @rromb for their discussions and insights.

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