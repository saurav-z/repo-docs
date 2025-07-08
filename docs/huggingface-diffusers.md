<p align="center">
    <br>
    <img src="https://raw.githubusercontent.com/huggingface/diffusers/main/docs/source/en/imgs/diffusers_library.jpg" width="400"/>
    <br>
<p>

# 🤗 Diffusers: Unleash the Power of Diffusion Models

🤗 Diffusers is your one-stop Python library for cutting-edge diffusion models, enabling you to generate stunning images, audio, and more with just a few lines of code. [Check out the original repository](https://github.com/huggingface/diffusers) for the latest updates and more information!

<p align="center">
    <a href="https://github.com/huggingface/diffusers/blob/main/LICENSE"><img alt="GitHub" src="https://img.shields.io/github/license/huggingface/datasets.svg?color=blue"></a>
    <a href="https://github.com/huggingface/diffusers/releases"><img alt="GitHub release" src="https://img.shields.io/github/release/huggingface/diffusers.svg"></a>
    <a href="https://pepy.tech/project/diffusers"><img alt="GitHub release" src="https://static.pepy.tech/badge/diffusers/month"></a>
    <a href="CODE_OF_CONDUCT.md"><img alt="Contributor Covenant" src="https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg"></a>
    <a href="https://twitter.com/diffuserslib"><img alt="X account" src="https://img.shields.io/twitter/url/https/twitter.com/diffuserslib.svg?style=social&label=Follow%20%40diffuserslib"></a>
</p>

## Key Features

*   **State-of-the-Art Diffusion Pipelines:** Generate images, audio, and more with pre-trained pipelines.
*   **Flexible Schedulers:** Experiment with diverse noise schedulers for optimal results.
*   **Modular Components:** Utilize pretrained models as building blocks for custom diffusion systems.
*   **Easy to Use:** Get started with powerful generative AI with minimal code.
*   **Extensive Model Support:** Access a vast library of pretrained models on the Hugging Face Hub.

## Installation

Install 🤗 Diffusers in a virtual environment using pip or conda:

### PyTorch

```bash
pip install --upgrade diffusers[torch]
```

or

```sh
conda install -c conda-forge diffusers
```

### Flax

```bash
pip install --upgrade diffusers[flax]
```

### Apple Silicon (M1/M2) Support

Refer to the [Apple Silicon guide](https://huggingface.co/docs/diffusers/optimization/mps) for optimization.

## Quickstart: Generate an Image

Create an image from text using a pretrained model:

```python
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipeline.to("cuda")
pipeline("An image of a squirrel in Picasso style").images[0]
```

## Building Custom Diffusion Systems

Explore the building blocks: models and schedulers:

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

## Documentation Navigation

*   **Tutorial:** Learn the core concepts and build your own diffusion systems.
*   **Loading:** Guides for loading and configuring components.
*   **Pipelines:** Use pipelines for various inference tasks.
*   **Optimization:** Improve speed and reduce memory usage.
*   **Training:** Train your own diffusion models.

## Contribute

Join the community!  Check the [Contribution guide](https://github.com/huggingface/diffusers/blob/main/CONTRIBUTING.md) and look for [issues](https://github.com/huggingface/diffusers/issues). Connect with us on [Discord](https://discord.gg/G7tWnz98XR).

## Popular Tasks & Pipelines

| Task                     | Pipeline                                                                 | Hugging Face Hub                                                                           |
| ------------------------ | ------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------- |
| Unconditional Image Generation | DDPM                                                                     | [google/ddpm-ema-church-256](https://huggingface.co/google/ddpm-ema-church-256)            |
| Text-to-Image          | Stable Diffusion Text-to-Image                                         | [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) |
| Text-to-Image          | unCLIP                                                                   | [kakaobrain/karlo-v1-alpha](https://huggingface.co/kakaobrain/karlo-v1-alpha)            |
| Text-to-Image          | DeepFloyd IF                                                                 | [DeepFloyd/IF-I-XL-v1.0](https://huggingface.co/DeepFloyd/IF-I-XL-v1.0)                    |
| Text-to-Image          | Kandinsky                                                                | [kandinsky-community/kandinsky-2-2-decoder](https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder) |
| Text-guided Image-to-Image | ControlNet                                                               | [lllyasviel/sd-controlnet-canny](https://huggingface.co/lllyasviel/sd-controlnet-canny) |
| Text-guided Image-to-Image | InstructPix2Pix                                                            | [timbrooks/instruct-pix2pix](https://huggingface.co/timbrooks/instruct-pix2pix)          |
| Text-guided Image-to-Image | Stable Diffusion Image-to-Image                                         | [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) |
| Text-guided Image Inpainting | Stable Diffusion Inpainting                                               | [runwayml/stable-diffusion-inpainting](https://huggingface.co/runwayml/stable-diffusion-inpainting) |
| Image Variation          | Stable Diffusion Image Variation                                        | [lambdalabs/sd-image-variations-diffusers](https://huggingface.co/lambdalabs/sd-image-variations-diffusers)  |
| Super Resolution         | Stable Diffusion Upscale                                                | [stabilityai/stable-diffusion-x4-upscaler](https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler)   |
| Super Resolution         | Stable Diffusion Latent Upscale                                         | [stabilityai/sd-x2-latent-upscaler](https://huggingface.co/stabilityai/sd-x2-latent-upscaler)  |

## Popular Libraries Using Diffusers

(List of example libraries, consider linking to them if possible)

*   Microsoft TaskMatrix
*   InvokeAI
*   InstantID
*   Apple ml-stable-diffusion
*   Lama Cleaner
*   IDEA-Research Grounded-Segment-Anything
*   Stable Dreamfusion
*   DeepFloyd IF
*   BentoML
*   Kohya_ss

And over 14,000 other amazing repositories!

## Credits

Thank you to the researchers and developers whose work made this library possible.

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