# 🤗 Diffusers: Generate Images, Audio, and More with State-of-the-Art Diffusion Models

**🤗 Diffusers** is the leading open-source library providing a modular toolkit for working with cutting-edge diffusion models. [Explore the Diffusers library on GitHub](https://github.com/huggingface/diffusers).

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

## Key Features

*   **Pre-trained Pipelines:** Easily generate images, audio, and 3D structures using ready-to-use diffusion pipelines.
*   **Modular Design:** Build custom diffusion systems with interchangeable noise schedulers and pre-trained models.
*   **Extensive Model Support:** Access a vast collection of pre-trained models from the Hugging Face Hub.
*   **Optimization Tools:** Optimize your models for faster performance and reduced memory consumption.
*   **Training Capabilities:** Train your own diffusion models for diverse tasks and datasets.

## Why Choose 🤗 Diffusers?

*   **Usability-Focused:** Diffusers prioritizes ease of use, making it accessible for both beginners and experts.
*   **Simple and Customizable:** The library emphasizes simplicity and customizability, allowing you to tailor solutions to your specific needs.
*   **Community-Driven:** Benefit from a vibrant open-source community and contribute to the ongoing development of diffusion models.

## Installation

Install 🤗 Diffusers in a virtual environment using pip or conda.

### PyTorch

```bash
pip install --upgrade diffusers[torch]
```

### Conda

```sh
conda install -c conda-forge diffusers
```

### Apple Silicon (M1/M2) Support

See the [Apple Silicon guide](https://huggingface.co/docs/diffusers/optimization/mps) for details.

## Quickstart: Generate an Image from Text

```python
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipeline.to("cuda")
pipeline("An image of a squirrel in Picasso style").images[0]
```

## Dive Deeper: Build Your Own Diffusion System

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

## Documentation

*   [Tutorial](https://huggingface.co/docs/diffusers/tutorials/tutorial_overview): Get started with the library's core features.
*   [Loading](https://huggingface.co/docs/diffusers/using-diffusers/loading): Learn how to load and configure components.
*   [Pipelines for inference](https://huggingface.co/docs/diffusers/using-diffusers/overview_techniques): Explore inference tasks and pipeline techniques.
*   [Optimization](https://huggingface.co/docs/diffusers/optimization/fp16): Optimize your models for efficiency.
*   [Training](https://huggingface.co/docs/diffusers/training/overview): Train diffusion models for various applications.

## Contribution

We welcome contributions! See our [Contribution Guide](https://github.com/huggingface/diffusers/blob/main/CONTRIBUTING.md) and explore [issues](https://github.com/huggingface/diffusers/issues) to contribute. Join the discussion on our [Discord](https://discord.gg/G7tWnz98XR).

## Popular Tasks & Pipelines

*(Table of popular tasks and pipelines - keep as is)*

## Used By

*(List of libraries that utilize diffusers - keep as is)*

## Credits

*(List of credits - keep as is)*

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