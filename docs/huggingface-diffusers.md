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

## 🤗 Diffusers: Unleash the Power of Diffusion Models

**🤗 Diffusers** is your comprehensive toolkit for working with cutting-edge diffusion models to generate stunning images, audio, and more. ([See the original repo](https://github.com/huggingface/diffusers))

**Key Features:**

*   **Intuitive Pipelines:** Easily generate outputs with state-of-the-art diffusion pipelines using just a few lines of code.
*   **Flexible Schedulers:** Experiment with interchangeable noise schedulers to control diffusion speed and image quality.
*   **Modular Models:** Build custom diffusion systems using pretrained models as building blocks.
*   **Usability-Focused Design:** Prioritizing ease of use, simplicity, and customizability.
*   **Wide Range of Applications:** Generate images, audio, and even 3D molecular structures.

## Installation

Install 🤗 Diffusers in a virtual environment using pip or conda.  Ensure you have PyTorch and/or Flax installed.

### PyTorch

```bash
pip install --upgrade diffusers[torch]
```

### Flax

```bash
pip install --upgrade diffusers[flax]
```

### Apple Silicon (M1/M2) Support

Refer to the [How to use Stable Diffusion in Apple Silicon](https://huggingface.co/docs/diffusers/optimization/mps) guide.

## Quickstart

Generate images from text effortlessly:

```python
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipeline.to("cuda")
pipeline("An image of a squirrel in Picasso style").images[0]
```

Or build your own diffusion systems:

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

Explore the [Quickstart](https://huggingface.co/docs/diffusers/quicktour) to begin your diffusion journey!

## Documentation

Explore the key documentation sections to learn more:

| Documentation                                                  | What you'll learn                                                                                                                                            |
| :------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Tutorial](https://huggingface.co/docs/diffusers/tutorials/tutorial_overview)                            | Learn to use models and schedulers to build your own diffusion system, plus training.                                           |
| [Loading](https://huggingface.co/docs/diffusers/using-diffusers/loading)                             | Discover how to load and configure pipelines, models, and schedulers.                                                           |
| [Pipelines for Inference](https://huggingface.co/docs/diffusers/using-diffusers/overview_techniques) | Learn how to use pipelines for inference, generation, and how to contribute your own pipelines.                                    |
| [Optimization](https://huggingface.co/docs/diffusers/optimization/fp16)                          | Optimize diffusion models for faster performance and reduced memory consumption.                                                  |
| [Training](https://huggingface.co/docs/diffusers/training/overview) | Learn to train diffusion models for various tasks using different techniques.                                                                             |

## Contribution

We welcome contributions from the open-source community!  See the [Contribution guide](https://github.com/huggingface/diffusers/blob/main/CONTRIBUTING.md) and contribute to our [issues](https://github.com/huggingface/diffusers/issues), including [Good first issues](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22), [New model/pipeline](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22New+pipeline%2Fmodel%22), and [New scheduler](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22New+scheduler%22).

Join our Discord channel: <a href="https://discord.gg/G7tWnz98XR"><img alt="Join us on Discord" src="https://img.shields.io/discord/823813159592001537?color=5865F2&logo=discord&logoColor=white"></a>

## Popular Tasks & Pipelines

| Task                            | Pipeline                                                                                                           | 🤗 Hub                                                                                                            |
| :------------------------------ | :----------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------- |
| Unconditional Image Generation  | [DDPM](https://huggingface.co/docs/diffusers/api/pipelines/ddpm)                                                  | [google/ddpm-ema-church-256](https://huggingface.co/google/ddpm-ema-church-256)                                |
| Text-to-Image                   | [Stable Diffusion Text-to-Image](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img) | [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) |
| Text-to-Image                   | [unCLIP](https://huggingface.co/docs/diffusers/api/pipelines/unclip)                                               | [kakaobrain/karlo-v1-alpha](https://huggingface.co/kakaobrain/karlo-v1-alpha)                                  |
| Text-to-Image                   | [DeepFloyd IF](https://huggingface.co/docs/diffusers/api/pipelines/deepfloyd_if)                                    | [DeepFloyd/IF-I-XL-v1.0](https://huggingface.co/DeepFloyd/IF-I-XL-v1.0)                                          |
| Text-to-Image                   | [Kandinsky](https://huggingface.co/docs/diffusers/api/pipelines/kandinsky)                                         | [kandinsky-community/kandinsky-2-2-decoder](https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder)      |
| Text-guided Image-to-Image      | [ControlNet](https://huggingface.co/docs/diffusers/api/pipelines/controlnet)                                        | [lllyasviel/sd-controlnet-canny](https://huggingface.co/lllyasviel/sd-controlnet-canny)                         |
| Text-guided Image-to-Image      | [InstructPix2Pix](https://huggingface.co/docs/diffusers/api/pipelines/pix2pix)                                     | [timbrooks/instruct-pix2pix](https://huggingface.co/timbrooks/instruct-pix2pix)                                   |
| Text-guided Image-to-Image      | [Stable Diffusion Image-to-Image](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/img2img)  | [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) |
| Text-guided Image Inpainting  | [Stable Diffusion Inpainting](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/inpaint)         | [runwayml/stable-diffusion-inpainting](https://huggingface.co/runwayml/stable-diffusion-inpainting)            |
| Image Variation                 | [Stable Diffusion Image Variation](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/image_variation) | [lambdalabs/sd-image-variations-diffusers](https://huggingface.co/lambdalabs/sd-image-variations-diffusers)      |
| Super Resolution                | [Stable Diffusion Upscale](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/upscale)             | [stabilityai/stable-diffusion-x4-upscaler](https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler)     |
| Super Resolution                | [Stable Diffusion Latent Upscale](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/latent_upscale) | [stabilityai/sd-x2-latent-upscaler](https://huggingface.co/stabilityai/sd-x2-latent-upscaler)                     |

## Popular Libraries Using 🤗 Diffusers

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
*   And over 14,000 other amazing GitHub repositories!

## Credits

This library is built on the research and implementations of many contributors. Special thanks to the following:

*   [CompVis](https://github.com/CompVis/latent-diffusion)
*   [hojonathanho](https://github.com/hojonathanho/diffusion)
*   [pesser](https://github.com/pesser/pytorch_diffusion)
*   [ermongroup](https://github.com/ermongroup/ddim)
*   [yang-song](https://github.com/yang-song/score_sde_pytorch)
*   [heejkoo](https://github.com/heejkoo/Awesome-Diffusion-Models)
*   [crowsonkb](https://github.com/crowsonkb/)
*   [rromb](https://github.com/rromb/)

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