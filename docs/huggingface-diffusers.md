# 🤗 Diffusers: Generate Images, Audio, and More with State-of-the-Art Diffusion Models

[Go to the original repository](https://github.com/huggingface/diffusers)

<p align="center">
    <br>
    <img src="https://raw.githubusercontent.com/huggingface/diffusers/main/docs/source/en/imgs/diffusers_library.jpg" width="400"/>
    <br>
<p>

**🤗 Diffusers** is your go-to library for utilizing and experimenting with cutting-edge diffusion models to generate stunning images, captivating audio, and even 3D molecular structures.  Built with a focus on usability, simplicity, and customizability, it's the perfect toolbox for both quick inference and in-depth model training.

## Key Features

*   **Pre-trained Pipelines:** Easily generate outputs with ready-to-use diffusion pipelines.
*   **Flexible Schedulers:** Experiment with various noise schedulers for diverse diffusion speeds and output qualities.
*   **Modular Models:** Utilize pre-trained models as building blocks to create custom diffusion systems.
*   **Extensive Community:** Join a vibrant community and access a wide range of pre-trained models on the Hugging Face Hub.
*   **Apple Silicon Support:** Optimized for Apple Silicon (M1/M2) for faster performance.

## Installation

Install 🤗 Diffusers in a virtual environment using either pip or Conda.

### PyTorch

```bash
pip install --upgrade diffusers[torch]
```

### Flax

```bash
pip install --upgrade diffusers[flax]
```

## Quickstart

Generate images from text in just a few lines of code using the `from_pretrained` method:

```python
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipeline.to("cuda")
pipeline("An image of a squirrel in Picasso style").images[0]
```

Or, build your own diffusion system by combining models and schedulers:

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

## Documentation Overview

| **Documentation**                                                  | **What You Can Learn**                                                                                                                                                                                                      |
| ----------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Tutorial](https://huggingface.co/docs/diffusers/tutorials/tutorial_overview)                                                          | Learn the library's core features, including building diffusion systems and training your own models.                                                                                    |
| [Loading](https://huggingface.co/docs/diffusers/using-diffusers/loading)                                                            | Guides on loading and configuring pipelines, models, and schedulers, including using different schedulers.                                                                               |
| [Pipelines for inference](https://huggingface.co/docs/diffusers/using-diffusers/overview_techniques)                                            | Learn to use pipelines for various inference tasks, batch generation, controlling outputs, and contributing pipelines.                                                                        |
| [Optimization](https://huggingface.co/docs/diffusers/optimization/fp16)                                                       | Optimize your models for faster execution and lower memory consumption.                                                                                                                                |
| [Training](https://huggingface.co/docs/diffusers/training/overview) | Train diffusion models for various tasks using diverse training techniques.                                                                                                                                                          |

## Contribution

We welcome contributions from the open-source community! Check out the [Contribution guide](https://github.com/huggingface/diffusers/blob/main/CONTRIBUTING.md) and explore [issues](https://github.com/huggingface/diffusers/issues) to contribute.

-   See [Good first issues](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22) for general opportunities.
-   See [New model/pipeline](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22New+pipeline%2Fmodel%22) to contribute new diffusion models / diffusion pipelines.
-   See [New scheduler](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22New+scheduler%22) for new scheduler implementations.

Join our Discord channel for discussions and support: <a href="https://discord.gg/G7tWnz98XR"><img alt="Join us on Discord" src="https://img.shields.io/discord/823813159592001537?color=5865F2&logo=discord&logoColor=white"></a>

## Popular Tasks & Pipelines

(Table of Popular Tasks & Pipelines - as in original README)

## Popular Libraries Using 🧨 Diffusers

(List of popular libraries - as in original README)

## Credits

(Credit section - as in original README)

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