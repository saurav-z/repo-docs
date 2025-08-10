# 🤗 Diffusers: The Go-To Library for Diffusion Models

**Unleash the power of AI-generated content with 🤗 Diffusers, your one-stop-shop for cutting-edge diffusion models for images, audio, and more!**  Explore the original repo: [https://github.com/huggingface/diffusers](https://github.com/huggingface/diffusers)

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

🤗 Diffusers provides a modular and versatile toolkit for both beginners and experienced users, prioritizing:

*   **State-of-the-Art Pipelines:** Easily generate content with pre-built pipelines for various tasks like text-to-image, image-to-image, and more.
*   **Flexible Schedulers:** Experiment with different diffusion speeds and output qualities using a range of interchangeable noise schedulers.
*   **Customizable Models:** Utilize pretrained models as building blocks to create your own diffusion systems, tailoring them to your specific needs.

## Key Features

*   **Easy-to-use:** Get started generating AI content with just a few lines of code.
*   **Extensive Model Support:** Access a vast library of pretrained diffusion models from the Hugging Face Hub.
*   **Modular Design:** Build and customize diffusion systems with interchangeable components.
*   **Optimization Guides:** Learn how to optimize your models for faster performance and reduced memory consumption.
*   **Training Support:** Train your own diffusion models for unique tasks.

## Installation

Install 🤗 Diffusers in a virtual environment using `pip` or `conda`:

### PyTorch

```bash
pip install --upgrade diffusers[torch]
```

```sh
conda install -c conda-forge diffusers
```

### Flax

```bash
pip install --upgrade diffusers[flax]
```

### Apple Silicon (M1/M2) support

Refer to the [Apple Silicon guide](https://huggingface.co/docs/diffusers/optimization/mps) for installation instructions.

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

For more information, see the [Quickstart](https://huggingface.co/docs/diffusers/quicktour).

## Documentation and Resources

| **Section**                                                         | **Description**                                                                                                                                                     |
| :------------------------------------------------------------------ | :---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Tutorial](https://huggingface.co/docs/diffusers/tutorials/tutorial_overview)                                                                           | Basic usage of the library, using models and schedulers, and training your own diffusion model.                                                                   |
| [Loading](https://huggingface.co/docs/diffusers/using-diffusers/loading)                                                                         | Loading and configuration of all components (pipelines, models, and schedulers) and using different schedulers.                                                         |
| [Pipelines for inference](https://huggingface.co/docs/diffusers/using-diffusers/overview_techniques)                                                | Using pipelines for inference tasks, batch generation, controlling outputs, and contributing a pipeline.                                                                 |
| [Optimization](https://huggingface.co/docs/diffusers/optimization/fp16)                                                                   | Optimize diffusion models for speed and memory usage.                                                                                                                |
| [Training](https://huggingface.co/docs/diffusers/training/overview) | Train diffusion models for different tasks using different training techniques.                                                                                          |

## Contribute

We welcome contributions from the open-source community!  Consult the [Contribution guide](https://github.com/huggingface/diffusers/blob/main/CONTRIBUTING.md) and explore [issues](https://github.com/huggingface/diffusers/issues).

*   [Good first issues](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22)
*   [New model/pipeline](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22New+pipeline%2Fmodel%22)
*   [New scheduler](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22New+scheduler%22)

Join our public Discord channel for discussions! <a href="https://discord.gg/G7tWnz98XR"><img alt="Join us on Discord" src="https://img.shields.io/discord/823813159592001537?color=5865F2&logo=discord&logoColor=white"></a>.

## Popular Tasks & Pipelines

**(Table of Pipelines - Keep as is)**

## Used by Leading Libraries

**(List of libraries - Keep as is)**

## Credits

**(Keep as is)**

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