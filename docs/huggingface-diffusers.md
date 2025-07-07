<p align="center">
    <br>
    <img src="https://raw.githubusercontent.com/huggingface/diffusers/main/docs/source/en/imgs/diffusers_library.jpg" width="400"/>
    <br>
<p>

# 🤗 Diffusers: Your Gateway to Cutting-Edge Diffusion Models

**Unleash the power of AI with 🤗 Diffusers, the leading library for generating images, audio, and 3D structures using state-of-the-art diffusion models.** ([View on GitHub](https://github.com/huggingface/diffusers))

<p align="center">
    <a href="https://github.com/huggingface/diffusers/blob/main/LICENSE"><img alt="GitHub" src="https://img.shields.io/github/license/huggingface/datasets.svg?color=blue"></a>
    <a href="https://github.com/huggingface/diffusers/releases"><img alt="GitHub release" src="https://img.shields.io/github/release/huggingface/diffusers.svg"></a>
    <a href="https://pepy.tech/project/diffusers"><img alt="GitHub release" src="https://static.pepy.tech/badge/diffusers/month"></a>
    <a href="CODE_OF_CONDUCT.md"><img alt="Contributor Covenant" src="https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg"></a>
    <a href="https://twitter.com/diffuserslib"><img alt="X account" src="https://img.shields.io/twitter/url/https/twitter.com/diffuserslib.svg?style=social&label=Follow%20%40diffuserslib"></a>
</p>

## Key Features

*   **Ready-to-Use Pipelines:** Generate images from text, modify existing images, and more with pre-built diffusion pipelines.
*   **Modular Design:**  Combine interchangeable noise schedulers and pretrained models for complete control over the diffusion process.
*   **Extensive Model Support:** Access a vast collection of pretrained diffusion models from the Hugging Face Hub and other sources.
*   **Easy to Use:** Designed with a focus on usability and simplicity, making it accessible for both beginners and experts.
*   **Customizable:** Build and train your own diffusion models with a flexible and customizable framework.
*   **Optimized Performance:** Optimize models for faster inference and reduced memory usage with guides on techniques like mixed precision.
*   **Community Driven:** Benefit from open-source contributions and support from the vibrant Hugging Face community.

## Installation

Install 🤗 Diffusers with either `pip` or `conda` in a virtual environment:

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

**Note:**  Refer to the documentation for detailed installation instructions, including support for Apple Silicon (M1/M2) devices.

## Quickstart

Get started generating images in seconds:

```python
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipeline.to("cuda")
pipeline("An image of a squirrel in Picasso style").images[0]
```

## Documentation

*   **Tutorial:** Learn the fundamentals of using models and schedulers.
*   **Loading:** Understand how to load and configure pipelines, models, and schedulers.
*   **Pipelines for Inference:** Explore techniques for various inference tasks and controlling outputs.
*   **Optimization:** Learn how to optimize your diffusion model for speed and memory efficiency.
*   **Training:** Find guides for training diffusion models for different tasks.

## Contribution

We welcome contributions from the open-source community! Check out the [Contribution guide](https://github.com/huggingface/diffusers/blob/main/CONTRIBUTING.md) and the [issues](https://github.com/huggingface/diffusers/issues) to get started.

Join the discussion on our public Discord channel: <a href="https://discord.gg/G7tWnz98XR"><img alt="Join us on Discord" src="https://img.shields.io/discord/823813159592001537?color=5865F2&logo=discord&logoColor=white"></a>.

## Popular Tasks & Pipelines

| **Task**                 | **Pipeline**                                                                  | **🤗 Hub**                                                                                                  |
| :----------------------- | :---------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------- |
| Unconditional Image Generation | DDPM                                                                        | [google/ddpm-ema-church-256](https://huggingface.co/google/ddpm-ema-church-256)                            |
| Text-to-Image            | Stable Diffusion Text-to-Image                                                | [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) |
| Text-to-Image            | unCLIP                                                                        | [kakaobrain/karlo-v1-alpha](https://huggingface.co/kakaobrain/karlo-v1-alpha)                               |
| Text-to-Image            | DeepFloyd IF                                                                  | [DeepFloyd/IF-I-XL-v1.0](https://huggingface.co/DeepFloyd/IF-I-XL-v1.0)                                      |
| Text-to-Image            | Kandinsky                                                                     | [kandinsky-community/kandinsky-2-2-decoder](https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder)  |
| Text-guided Image-to-Image | ControlNet                                                                    | [lllyasviel/sd-controlnet-canny](https://huggingface.co/lllyasviel/sd-controlnet-canny)                     |
| Text-guided Image-to-Image | InstructPix2Pix                                                               | [timbrooks/instruct-pix2pix](https://huggingface.co/timbrooks/instruct-pix2pix)                             |
| Text-guided Image-to-Image | Stable Diffusion Image-to-Image                                               | [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) |
| Text-guided Image Inpainting | Stable Diffusion Inpainting                                                      | [runwayml/stable-diffusion-inpainting](https://huggingface.co/runwayml/stable-diffusion-inpainting)        |
| Image Variation          | Stable Diffusion Image Variation                                               | [lambdalabs/sd-image-variations-diffusers](https://huggingface.co/lambdalabs/sd-image-variations-diffusers)  |
| Super Resolution         | Stable Diffusion Upscale                                                      | [stabilityai/stable-diffusion-x4-upscaler](https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler)  |
| Super Resolution         | Stable Diffusion Latent Upscale                                               | [stabilityai/sd-x2-latent-upscaler](https://huggingface.co/stabilityai/sd-x2-latent-upscaler)                 |

## Used by Many Libraries

*   [TaskMatrix](https://github.com/microsoft/TaskMatrix)
*   [InvokeAI](https://github.com/invoke-ai/InvokeAI)
*   [InstantID](https://github.com/InstantID/InstantID)
*   [ml-stable-diffusion](https://github.com/apple/ml-stable-diffusion)
*   [lama-cleaner](https://github.com/Sanster/lama-cleaner)
*   [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything)
*   [stable-dreamfusion](https://github.com/ashawkey/stable-dreamfusion)
*   [deep-floyd/IF](https://github.com/deep-floyd/IF)
*   [BentoML](https://github.com/bentoml/BentoML)
*   [kohya_ss](https://github.com/bmaltais/kohya_ss)
*   +14,000 other amazing GitHub repositories 💪

## Credits

(Credits section included)

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