# 🤗 Diffusers: Generate Images, Audio, and More with Diffusion Models

**Unleash the power of diffusion models with 🤗 Diffusers, your go-to library for creating stunning AI-generated content.** ([Original Repo](https://github.com/huggingface/diffusers))

[![License](https://img.shields.io/github/license/huggingface/datasets.svg?color=blue)](https://github.com/huggingface/diffusers/blob/main/LICENSE)
[![GitHub release](https://img.shields.io/github/release/huggingface/diffusers.svg)](https://github.com/huggingface/diffusers/releases)
[![Pypi](https://static.pepy.tech/badge/diffusers/month)](https://pepy.tech/project/diffusers)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/diffuserslib.svg?style=social&label=Follow%20%40diffuserslib)](https://twitter.com/diffuserslib)

🤗 Diffusers provides a modular and user-friendly toolkit for working with state-of-the-art diffusion models. Whether you're a beginner or an expert, this library offers the building blocks you need to generate amazing results. Built with a focus on usability, simplicity, and customizability, Diffusers empowers you to:

**Key Features:**

*   **Diffusion Pipelines:** Easily run pre-trained diffusion models with just a few lines of code for tasks like text-to-image generation.
*   **Flexible Schedulers:** Experiment with different noise schedulers to control the speed and quality of your generated outputs.
*   **Modular Models:** Use pre-trained models as components and combine them with schedulers to create custom diffusion systems.
*   **Extensive Model Support:** Access a vast selection of pre-trained models on the Hugging Face Hub (30,000+ checkpoints).

## Installation

Get started by installing 🤗 Diffusers in a virtual environment using either `pip` or `conda`:

### PyTorch

```bash
pip install --upgrade diffusers[torch]
```

### Conda

```sh
conda install -c conda-forge diffusers
```

### Apple Silicon (M1/M2)

See the [Apple Silicon guide](https://huggingface.co/docs/diffusers/optimization/mps) for optimized performance.

## Quickstart: Image Generation from Text

Generate images from text using a pre-trained model:

```python
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipeline.to("cuda")
pipeline("An image of a squirrel in Picasso style").images[0]
```

Or, build your own diffusion system:

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

Explore the [Quickstart](https://huggingface.co/docs/diffusers/quicktour) to dive deeper.

## Documentation & Tutorials

Learn how to use and customize the library:

| Documentation                                                              | What can I learn?                                                                                                                                                                                               |
| -------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Tutorial](https://huggingface.co/docs/diffusers/tutorials/tutorial_overview)                                                             | Learn the library's basics, including how to build your own diffusion system with models and schedulers and how to train your own diffusion model.                             |
| [Loading](https://huggingface.co/docs/diffusers/using-diffusers/loading)                                                               | How to load and configure all components of the library (pipelines, models, and schedulers).                                                                                                                                  |
| [Pipelines for inference](https://huggingface.co/docs/diffusers/using-diffusers/overview_techniques)                                              | Use pipelines for different inference tasks, batch generation, controlling outputs and randomness, and how to contribute a pipeline to the library.                                        |
| [Optimization](https://huggingface.co/docs/diffusers/optimization/fp16)                                                         | Guides on optimizing diffusion models for faster performance and reduced memory consumption.                                                                                                                                       |
| [Training](https://huggingface.co/docs/diffusers/training/overview) | Learn how to train a diffusion model for different tasks with different training techniques.                                                                                                                          |

## Contribute

We welcome contributions! Check out the [Contribution guide](https://github.com/huggingface/diffusers/blob/main/CONTRIBUTING.md) and explore [issues](https://github.com/huggingface/diffusers/issues) marked as "good first issue," "New model/pipeline," or "New scheduler" to get involved.

Join the community on Discord: <a href="https://discord.gg/G7tWnz98XR"><img alt="Join us on Discord" src="https://img.shields.io/discord/823813159592001537?color=5865F2&logo=discord&logoColor=white"></a>

## Popular Tasks & Pipelines

| Task                           | Pipeline                                                                                     | 🤗 Hub                                                                                                   |
| ------------------------------ | -------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| Unconditional Image Generation | [DDPM](https://huggingface.co/docs/diffusers/api/pipelines/ddpm)                             | [google/ddpm-ema-church-256](https://huggingface.co/google/ddpm-ema-church-256)                         |
| Text-to-Image                  | [Stable Diffusion Text-to-Image](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img) | [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) |
| Text-to-Image                  | [unCLIP](https://huggingface.co/docs/diffusers/api/pipelines/unclip)                            | [kakaobrain/karlo-v1-alpha](https://huggingface.co/kakaobrain/karlo-v1-alpha)                           |
| Text-to-Image                  | [DeepFloyd IF](https://huggingface.co/docs/diffusers/api/pipelines/deepfloyd_if)              | [DeepFloyd/IF-I-XL-v1.0](https://huggingface.co/DeepFloyd/IF-I-XL-v1.0)                               |
| Text-to-Image                  | [Kandinsky](https://huggingface.co/docs/diffusers/api/pipelines/kandinsky)                     | [kandinsky-community/kandinsky-2-2-decoder](https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder)     |
| Text-guided Image-to-Image     | [ControlNet](https://huggingface.co/docs/diffusers/api/pipelines/controlnet)                   | [lllyasviel/sd-controlnet-canny](https://huggingface.co/lllyasviel/sd-controlnet-canny)                     |
| Text-guided Image-to-Image     | [InstructPix2Pix](https://huggingface.co/docs/diffusers/api/pipelines/pix2pix)               | [timbrooks/instruct-pix2pix](https://huggingface.co/timbrooks/instruct-pix2pix)                         |
| Text-guided Image-to-Image     | [Stable Diffusion Image-to-Image](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/img2img) | [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) |
| Text-guided Image Inpainting   | [Stable Diffusion Inpainting](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/inpaint)   | [runwayml/stable-diffusion-inpainting](https://huggingface.co/runwayml/stable-diffusion-inpainting)       |
| Image Variation                | [Stable Diffusion Image Variation](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/image_variation) | [lambdalabs/sd-image-variations-diffusers](https://huggingface.co/lambdalabs/sd-image-variations-diffusers)     |
| Super Resolution               | [Stable Diffusion Upscale](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/upscale)  | [stabilityai/stable-diffusion-x4-upscaler](https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler)  |
| Super Resolution               | [Stable Diffusion Latent Upscale](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/latent_upscale) | [stabilityai/sd-x2-latent-upscaler](https://huggingface.co/stabilityai/sd-x2-latent-upscaler)              |

## Used By Many

Popular libraries using 🧨 Diffusers:

*   [Microsoft TaskMatrix](https://github.com/microsoft/TaskMatrix)
*   [InvokeAI](https://github.com/invoke-ai/InvokeAI)
*   [InstantID](https://github.com/InstantID/InstantID)
*   [Apple ml-stable-diffusion](https://github.com/apple/ml-stable-diffusion)
*   [Lama Cleaner](https://github.com/Sanster/lama-cleaner)
*   [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything)
*   [Stable Dreamfusion](https://github.com/ashawkey/stable-dreamfusion)
*   [DeepFloyd IF](https://github.com/deep-floyd/IF)
*   [BentoML](https://github.com/bentoml/BentoML)
*   [Kohya_ss](https://github.com/bmaltais/kohya_ss)
*   ...and over 14,000 other GitHub repositories 💪

## Credits

This library builds upon the work of many researchers and developers. We would like to thank:

*   @CompVis for the latent diffusion models library.
*   @hojonathanho for the original DDPM implementation.
*   @pesser for the PyTorch translation.
*   @ermongroup for the DDIM implementation.
*   @yang-song for the Score-VE and Score-VP implementations.

And, we thank @heejkoo and @crowsonkb and @rromb.

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