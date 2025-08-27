# 🤗 Diffusers: Generate Images, Audio, and More with State-of-the-Art Diffusion Models

**Unleash the power of diffusion models with 🤗 Diffusers, the leading library for generating images, audio, and even 3D structures.**  ([Original Repository](https://github.com/huggingface/diffusers))

<p align="center">
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

Built for both beginners and experts, 🤗 Diffusers offers a modular and user-friendly experience, prioritizing:

*   **Simplified Inference:** Run state-of-the-art diffusion pipelines with minimal code.
*   **Flexible Schedulers:** Experiment with various noise schedulers for optimal results.
*   **Customizable Models:** Utilize pretrained models as building blocks for creating your own diffusion systems.

## Key Features

*   **State-of-the-Art Pipelines:** Access ready-to-use pipelines for text-to-image, image-to-image, and more.
*   **Modular Design:** Easily swap out noise schedulers and model components.
*   **Pretrained Models:** Leverage a vast library of pretrained models from the Hugging Face Hub.
*   **Easy Installation:** Simple setup with PyPI or Conda.
*   **Optimization Support:** Guides and resources for optimizing your models on different hardware.
*   **Active Community:** Join a thriving community for support and collaboration.

## Installation

Install 🤗 Diffusers in a virtual environment using either `pip` or `conda`.

**Using pip:**

```bash
pip install --upgrade diffusers[torch]  # Or diffusers[flax]
```

**Using conda:**

```sh
conda install -c conda-forge diffusers
```

**Apple Silicon (M1/M2) Support:** See the [official guide](https://huggingface.co/docs/diffusers/optimization/mps) for optimized performance.

## Quickstart

Generate images quickly:

```python
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipeline.to("cuda")
pipeline("An image of a squirrel in Picasso style").images[0]
```

Or build your own diffusion system:

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

*   [Tutorial](https://huggingface.co/docs/diffusers/tutorials/tutorial_overview): Learn the basics of using the library.
*   [Loading](https://huggingface.co/docs/diffusers/using-diffusers/loading): How to load and configure the components.
*   [Pipelines for inference](https://huggingface.co/docs/diffusers/using-diffusers/overview_techniques): Use pipelines for different inference tasks.
*   [Optimization](https://huggingface.co/docs/diffusers/optimization/fp16): Optimize your models.
*   [Training](https://huggingface.co/docs/diffusers/training/overview): Train your own diffusion model.

## Contribution

Contribute to 🤗 Diffusers!  See the [Contribution guide](https://github.com/huggingface/diffusers/blob/main/CONTRIBUTING.md) and explore [issues](https://github.com/huggingface/diffusers/issues) to get started. Join our [Discord channel](https://discord.gg/G7tWnz98XR) to connect with the community.

## Popular Tasks & Pipelines

| Task                       | Pipeline                                                                                               | 🤗 Hub                                                                                                  |
| -------------------------- | ------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------- |
| Unconditional Image Generation | DDPM                                                                                                   | [google/ddpm-ema-church-256](https://huggingface.co/google/ddpm-ema-church-256)                              |
| Text-to-Image              | Stable Diffusion Text-to-Image                                                                        | [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) |
| Text-to-Image              | unCLIP                                                                                                 | [kakaobrain/karlo-v1-alpha](https://huggingface.co/kakaobrain/karlo-v1-alpha)                              |
| Text-to-Image              | DeepFloyd IF                                                                                            | [DeepFloyd/IF-I-XL-v1.0](https://huggingface.co/DeepFloyd/IF-I-XL-v1.0)                                    |
| Text-to-Image              | Kandinsky                                                                                              | [kandinsky-community/kandinsky-2-2-decoder](https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder) |
| Text-guided Image-to-Image   | ControlNet                                                                                             | [lllyasviel/sd-controlnet-canny](https://huggingface.co/lllyasviel/sd-controlnet-canny)                    |
| Text-guided Image-to-Image   | InstructPix2Pix                                                                                        | [timbrooks/instruct-pix2pix](https://huggingface.co/timbrooks/instruct-pix2pix)                             |
| Text-guided Image-to-Image   | Stable Diffusion Image-to-Image                                                                        | [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) |
| Text-guided Image Inpainting | Stable Diffusion Inpainting                                                                           | [runwayml/stable-diffusion-inpainting](https://huggingface.co/runwayml/stable-diffusion-inpainting)        |
| Image Variation            | Stable Diffusion Image Variation                                                                        | [lambdalabs/sd-image-variations-diffusers](https://huggingface.co/lambdalabs/sd-image-variations-diffusers) |
| Super Resolution           | Stable Diffusion Upscale                                                                               | [stabilityai/stable-diffusion-x4-upscaler](https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler) |
| Super Resolution           | Stable Diffusion Latent Upscale                                                                          | [stabilityai/sd-x2-latent-upscaler](https://huggingface.co/stabilityai/sd-x2-latent-upscaler)             |

## Libraries Using 🧨 Diffusers

(Listing from original README)

## Credits

(Listing from original README)

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