# 🤗 Diffusers: Generate Images, Audio, and More with State-of-the-Art Diffusion Models

**Unleash the power of diffusion models with 🤗 Diffusers, your go-to library for generating stunning visuals, audio, and even 3D molecular structures!** ([Original Repository](https://github.com/huggingface/diffusers))

<p align="center">
    <br>
    <img src="https://raw.githubusercontent.com/huggingface/diffusers/main/docs/source/en/imgs/diffusers_library.jpg" width="400"/>
    <br>
<p>

[![License](https://img.shields.io/github/license/huggingface/datasets.svg?color=blue)](https://github.com/huggingface/diffusers/blob/main/LICENSE)
[![GitHub release](https://img.shields.io/github/release/huggingface/diffusers.svg)](https://github.com/huggingface/diffusers/releases)
[![Pypi](https://static.pepy.tech/badge/diffusers/month)](https://pepy.tech/project/diffusers)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/diffuserslib.svg?style=social&label=Follow%20%40diffuserslib)](https://twitter.com/diffuserslib)

Built with a focus on **usability, simplicity, and customizability**, 🤗 Diffusers provides the building blocks for both beginners and experts.

## Key Features:

*   **Pre-trained Diffusion Pipelines:** Easily generate images, audio, and other media with just a few lines of code.
*   **Modular Components:** Access interchangeable noise schedulers and pretrained models for fine-grained control and customization.
*   **Extensive Model Support:** Access a vast library of pretrained models from the Hugging Face Hub, and build your own end-to-end diffusion systems.
*   **Flexible and Customizable:** Design and train your own diffusion models for specific tasks.
*   **Active Community:** Benefit from a vibrant community and readily available resources.

## Installation

Install 🤗 Diffusers in a virtual environment using either pip or conda:

### PyTorch

```bash
pip install --upgrade diffusers[torch]
```

### Conda

```sh
conda install -c conda-forge diffusers
```

### Apple Silicon (M1/M2)

Refer to the [How to use Stable Diffusion in Apple Silicon](https://huggingface.co/docs/diffusers/optimization/mps) guide for optimal performance on Apple Silicon.

## Quickstart

Generate images from text using a pre-trained model:

```python
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipeline.to("cuda")
pipeline("An image of a squirrel in Picasso style").images[0]
```

Or, build your diffusion system from scratch:

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

Explore the [Quickstart](https://huggingface.co/docs/diffusers/quicktour) for a fast-paced introduction!

## Documentation

Explore the documentation to master your skills:

| Documentation                                                                 | What Can I Learn?                                                                                                                                                                                                                                                        |
| :---------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Tutorial](https://huggingface.co/docs/diffusers/tutorials/tutorial_overview) | Learn to use the library's key features, like models and schedulers to build your diffusion system, and train your diffusion model.                                                                                                                                  |
| [Loading](https://huggingface.co/docs/diffusers/using-diffusers/loading)     | Learn how to load and configure components (pipelines, models, and schedulers) and how to use different schedulers.                                                                                                                                                     |
| [Pipelines for Inference](https://huggingface.co/docs/diffusers/using-diffusers/overview_techniques)           | Learn how to use pipelines for different inference tasks, batched generation, controlling generated outputs and randomness, and how to contribute a pipeline to the library.               |
| [Optimization](https://huggingface.co/docs/diffusers/optimization/fp16)         | Learn to optimize your diffusion model to run faster and consume less memory.                                                                                                                                                                                                      |
| [Training](https://huggingface.co/docs/diffusers/training/overview)           | Learn to train a diffusion model for different tasks with different training techniques.                                                                                                                                                                                           |

## Contribution

We welcome contributions!  Check out the [Contribution guide](https://github.com/huggingface/diffusers/blob/main/CONTRIBUTING.md) for details.

*   Find [Good first issues](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22) to get started.
*   Contribute [New pipelines/models](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22New+pipeline%2Fmodel%22).
*   Contribute [New schedulers](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22New+scheduler%22).

Join the community on Discord: <a href="https://discord.gg/G7tWnz98XR"><img alt="Join us on Discord" src="https://img.shields.io/discord/823813159592001537?color=5865F2&logo=discord&logoColor=white"></a>

## Popular Tasks & Pipelines

| Task                         | Pipeline                                                                                                           | 🤗 Hub                                                                                                                                   |
| :--------------------------- | :----------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------- |
| Unconditional Image Generation | [DDPM](https://huggingface.co/docs/diffusers/api/pipelines/ddpm)                                                   | [google/ddpm-ema-church-256](https://huggingface.co/google/ddpm-ema-church-256)                                                        |
| Text-to-Image                | [Stable Diffusion Text-to-Image](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img)   | [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)                     |
| Text-to-Image                | [unCLIP](https://huggingface.co/docs/diffusers/api/pipelines/unclip)                                               | [kakaobrain/karlo-v1-alpha](https://huggingface.co/kakaobrain/karlo-v1-alpha)                                                          |
| Text-to-Image                | [DeepFloyd IF](https://huggingface.co/docs/diffusers/api/pipelines/deepfloyd_if)                                 | [DeepFloyd/IF-I-XL-v1.0](https://huggingface.co/DeepFloyd/IF-I-XL-v1.0)                                                                 |
| Text-to-Image                | [Kandinsky](https://huggingface.co/docs/diffusers/api/pipelines/kandinsky)                                           | [kandinsky-community/kandinsky-2-2-decoder](https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder)                         |
| Text-guided Image-to-Image    | [ControlNet](https://huggingface.co/docs/diffusers/api/pipelines/controlnet)                                       | [lllyasviel/sd-controlnet-canny](https://huggingface.co/lllyasviel/sd-controlnet-canny)                                                   |
| Text-guided Image-to-Image    | [InstructPix2Pix](https://huggingface.co/docs/diffusers/api/pipelines/pix2pix)                                   | [timbrooks/instruct-pix2pix](https://huggingface.co/timbrooks/instruct-pix2pix)                                                        |
| Text-guided Image-to-Image    | [Stable Diffusion Image-to-Image](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/img2img) | [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)                     |
| Text-guided Image Inpainting   | [Stable Diffusion Inpainting](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/inpaint)   | [runwayml/stable-diffusion-inpainting](https://huggingface.co/runwayml/stable-diffusion-inpainting)                                    |
| Image Variation              | [Stable Diffusion Image Variation](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/image_variation) | [lambdalabs/sd-image-variations-diffusers](https://huggingface.co/lambdalabs/sd-image-variations-diffusers)                            |
| Super Resolution             | [Stable Diffusion Upscale](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/upscale)          | [stabilityai/stable-diffusion-x4-upscaler](https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler)                           |
| Super Resolution             | [Stable Diffusion Latent Upscale](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/latent_upscale) | [stabilityai/sd-x2-latent-upscaler](https://huggingface.co/stabilityai/sd-x2-latent-upscaler)                                          |

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
*   And many more!

Thank you for using 🤗 Diffusers!

## Credits

This library builds upon the work of many researchers. We are particularly grateful to:

*   [@CompVis](https://github.com/CompVis) for their latent diffusion models library ([https://github.com/CompVis/latent-diffusion](https://github.com/CompVis/latent-diffusion)).
*   [@hojonathanho](https://github.com/hojonathanho) for the original DDPM implementation ([https://github.com/hojonathanho/diffusion](https://github.com/hojonathanho/diffusion)) and [@pesser](https://github.com/pesser) for the PyTorch translation ([https://github.com/pesser/pytorch_diffusion](https://github.com/pesser/pytorch_diffusion)).
*   [@ermongroup](https://github.com/ermongroup) for their DDIM implementation ([https://github.com/ermongroup/ddim](https://github.com/ermongroup/ddim)).
*   [@yang-song](https://github.com/yang-song) for the Score-VE and Score-VP implementations ([https://github.com/yang-song/score_sde_pytorch](https://github.com/yang-song/score_sde_pytorch)).

We also thank [@heejkoo](https://github.com/heejkoo) for the overview of diffusion models ([https://github.com/heejkoo/Awesome-Diffusion-Models](https://github.com/heejkoo/Awesome-Diffusion-Models)) and [@crowsonkb](https://github.com/crowsonkb) and [@rromb](https://github.com/rromb) for their insights.

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