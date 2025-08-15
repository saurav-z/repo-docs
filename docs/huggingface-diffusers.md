# 🤗 Diffusers: Unleash the Power of Diffusion Models

**Generate stunning images, audio, and 3D structures with the leading library for diffusion models!** Access the [original repo](https://github.com/huggingface/diffusers) for the latest updates and contributions.

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

🤗 Diffusers is the premier library for working with state-of-the-art diffusion models. Whether you're a beginner or an expert, you can quickly generate high-quality content. Built with a focus on:

*   **Usability:** Easy to learn and use for both inference and training.
*   **Simplicity:** Easy to understand and get started.
*   **Customizability:** Easily tweak and adapt to your needs.

## Key Features of 🤗 Diffusers:

*   **Pre-trained Diffusion Pipelines:** Generate images, audio, and more with just a few lines of code using ready-to-use pipelines.
*   **Flexible Schedulers:** Experiment with various noise schedulers to control diffusion speed and image quality.
*   **Modular Building Blocks:** Access pre-trained models, schedulers, and pipelines as interchangeable components for creating your own diffusion systems.

## Installation

Install 🤗 Diffusers in a virtual environment.

### PyTorch

```bash
pip install --upgrade diffusers[torch]
```

### Flax

```bash
pip install --upgrade diffusers[flax]
```

### Conda (Community Maintained)
```sh
conda install -c conda-forge diffusers
```
### Apple Silicon (M1/M2) Support
Refer to the [MPS Guide](https://huggingface.co/docs/diffusers/optimization/mps) for details.

## Quickstart: Generate an Image from Text

Here's how easy it is to generate an image:

```python
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipeline.to("cuda")
pipeline("An image of a squirrel in Picasso style").images[0]
```

**Explore the documentation for advanced use cases!**

## Explore the Documentation

| **Section**                                                   | **What to Learn**                                                                                                                                                                           |
|---------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Tutorial](https://huggingface.co/docs/diffusers/tutorials/tutorial_overview)                                                            | Build diffusion systems, and train your own diffusion models with models and schedulers.  |
| [Loading](https://huggingface.co/docs/diffusers/using-diffusers/loading)                                                             | Guides for loading and configuring all components.                                         |
| [Pipelines for inference](https://huggingface.co/docs/diffusers/using-diffusers/overview_techniques)                                             | Guides for different inference tasks and generating outputs.               |
| [Optimization](https://huggingface.co/docs/diffusers/optimization/fp16)                                                        | Optimize your model for speed and memory usage.                                                                                                         |
| [Training](https://huggingface.co/docs/diffusers/training/overview) | Train diffusion models for various tasks and using different techniques.                                                                                               |

## Contribute

We welcome contributions from the open-source community! Check out our [Contribution guide](https://github.com/huggingface/diffusers/blob/main/CONTRIBUTING.md).

*   [Good first issues](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22)
*   [New model/pipeline](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22New+pipeline%2Fmodel%22)
*   [New scheduler](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22New+scheduler%22)

Join our Discord: <a href="https://discord.gg/G7tWnz98XR"><img alt="Join us on Discord" src="https://img.shields.io/discord/823813159592001537?color=5865F2&logo=discord&logoColor=white"></a>

## Popular Tasks & Pipelines

*(Table of pipelines will stay the same)*

## Used by Popular Libraries

*(List of libraries will stay the same)*

## Credits

*(Credits section will stay the same)*

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