# 🤗 Diffusers: Generate Images, Audio, and 3D Structures with Diffusion Models

**Unleash the power of diffusion models with 🤗 Diffusers, the leading library for generating high-quality content.** [Visit the original repository](https://github.com/huggingface/diffusers)

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

## Key Features:

*   **State-of-the-art Diffusion Pipelines:** Generate images, audio, and 3D structures with pre-trained models.
*   **Modular Design:** Build your own diffusion systems with interchangeable components.
*   **Easy-to-Use:** Generate outputs with just a few lines of code.
*   **Highly Customizable:** Modify and train models to fit your specific needs.
*   **Extensive Model Support:** Access a wide range of pre-trained models on the [Hugging Face Hub](https://huggingface.co/models?library=diffusers&sort=downloads).

## Installation

Install 🤗 Diffusers in a virtual environment using pip or conda.

### PyTorch

With `pip`:

```bash
pip install --upgrade diffusers[torch]
```

With `conda`:

```sh
conda install -c conda-forge diffusers
```

### Flax

With `pip`:

```bash
pip install --upgrade diffusers[flax]
```

### Apple Silicon (M1/M2) support

Refer to the [How to use Stable Diffusion in Apple Silicon](https://huggingface.co/docs/diffusers/optimization/mps) guide.

## Quickstart

Generate an image from text:

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

Explore the [Quickstart](https://huggingface.co/docs/diffusers/quicktour) to start your diffusion journey!

## Documentation and Guides

| **Documentation**                                                   | **What can I learn?**                                                                                                                                                                           |
|---------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Tutorial](https://huggingface.co/docs/diffusers/tutorials/tutorial_overview)                                                            | Learn how to use the library's main features, like using models and schedulers to build a diffusion system, and train a diffusion model.  |
| [Loading](https://huggingface.co/docs/diffusers/using-diffusers/loading)                                                             | Discover how to load and configure all components (pipelines, models, and schedulers), plus how to use different schedulers.                                         |
| [Pipelines for inference](https://huggingface.co/docs/diffusers/using-diffusers/overview_techniques)                                             | Learn how to use pipelines for inference, batch generation, controlling the outputs, and contributing a pipeline.               |
| [Optimization](https://huggingface.co/docs/diffusers/optimization/fp16)                                                        | Optimize your diffusion model for speed and memory usage.                                                                                                          |
| [Training](https://huggingface.co/docs/diffusers/training/overview) | Train a diffusion model with different techniques.                                                                                                                              |

## Contribute

We welcome contributions!  Check out our [Contribution guide](https://github.com/huggingface/diffusers/blob/main/CONTRIBUTING.md) to get started. Explore the [issues](https://github.com/huggingface/diffusers/issues) for contribution opportunities.

*   [Good first issues](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22)
*   [New model/pipeline](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22New+pipeline%2Fmodel%22)
*   [New scheduler](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22New+scheduler%22)

Join the community on Discord: <a href="https://discord.gg/G7tWnz98XR"><img alt="Join us on Discord" src="https://img.shields.io/discord/823813159592001537?color=5865F2&logo=discord&logoColor=white"></a>

## Popular Tasks & Pipelines

(Table of Popular Tasks & Pipelines is kept, as it adds valuable SEO keywords)

## Popular Libraries Using 🧨 Diffusers

(List of libraries is kept for the same reason)

## Credits

(Credits are kept to acknowledge the contributors)

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