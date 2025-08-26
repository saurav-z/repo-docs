# 🤗 Diffusers: Generate Images, Audio, and More with State-of-the-Art Diffusion Models

**Unleash the power of AI with 🤗 Diffusers, your go-to library for creating stunning visuals, audio, and 3D molecular structures using cutting-edge diffusion models.** [Explore the original repo](https://github.com/huggingface/diffusers).

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

Built with a focus on usability, simplicity, and customizability, 🤗 Diffusers empowers you to leverage the latest advancements in diffusion models.

**Key Features:**

*   **Pre-trained Pipelines:** Quickly generate images, audio, and more with ready-to-use diffusion pipelines.
*   **Modular Components:**  Utilize interchangeable noise schedulers and pretrained models for ultimate flexibility and customization.
*   **Easy to Use:** Get started generating outputs with just a few lines of code.
*   **Extensive Model Library:**  Access a vast collection of pre-trained models on the [Hugging Face Hub](https://huggingface.co/models?library=diffusers&sort=downloads).
*   **Optimization Guides:** Optimize your models for faster performance and reduced memory consumption.
*   **Training Support:** Train your own diffusion models using a variety of techniques.

## Installation

Install 🤗 Diffusers in a virtual environment using pip or conda:

**PyTorch:**

```bash
pip install --upgrade diffusers[torch]
```

**Flax:**

```bash
pip install --upgrade diffusers[flax]
```

**Conda:**
```sh
conda install -c conda-forge diffusers
```

**Apple Silicon (M1/M2) Support:**  Refer to the [Apple Silicon guide](https://huggingface.co/docs/diffusers/optimization/mps) for optimized performance.

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

## Documentation & Resources

| Documentation                                                   | What You'll Learn                                                                                                                                                                           |
|---------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Tutorial](https://huggingface.co/docs/diffusers/tutorials/tutorial_overview)                                                            | Comprehensive guide to the library's key features, building custom diffusion systems, and training your own models.  |
| [Loading](https://huggingface.co/docs/diffusers/using-diffusers/loading)                                                             | Instructions for loading and configuring pipelines, models, and schedulers.                                       |
| [Pipelines for inference](https://huggingface.co/docs/diffusers/using-diffusers/overview_techniques)                                             | Guides for inference tasks, batch generation, controlling outputs, and contributing pipelines.               |
| [Optimization](https://huggingface.co/docs/diffusers/optimization/fp16)                                                        | Optimize your diffusion model for faster performance.                                                                                                          |
| [Training](https://huggingface.co/docs/diffusers/training/overview) | Guides for training diffusion models for different tasks.                                                                                               |

## Contribution

Join the community and contribute to 🤗 Diffusers!  See the [Contribution guide](https://github.com/huggingface/diffusers/blob/main/CONTRIBUTING.md) and explore open [issues](https://github.com/huggingface/diffusers/issues) to get started.

*   [Good first issues](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22)
*   [New model/pipeline](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22New+pipeline%2Fmodel%22)
*   [New scheduler](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22New+scheduler%22)

Connect with us on Discord: <a href="https://discord.gg/G7tWnz98XR"><img alt="Join us on Discord" src="https://img.shields.io/discord/823813159592001537?color=5865F2&logo=discord&logoColor=white"></a>.

## Popular Tasks & Pipelines

*(Table of Popular Tasks and Pipelines - same as in the original README)*

## Used by Amazing Libraries

*(List of popular libraries that utilize Diffusers - same as in the original README)*

## Credits

*(List of credits - same as in the original README)*

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