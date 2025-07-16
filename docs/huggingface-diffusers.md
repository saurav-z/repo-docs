# 🤗 Diffusers: Generate Images, Audio, and More with State-of-the-Art Diffusion Models

**Unleash the power of diffusion models with 🤗 Diffusers, the premier library for generating stunning content.**

[![License](https://img.shields.io/github/license/huggingface/datasets.svg?color=blue)](https://github.com/huggingface/diffusers/blob/main/LICENSE)
[![Release](https://img.shields.io/github/release/huggingface/diffusers.svg)](https://github.com/huggingface/diffusers/releases)
[![Downloads](https://static.pepy.tech/badge/diffusers/month)](https://pepy.tech/project/diffusers)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/diffuserslib.svg?style=social&label=Follow%20%40diffuserslib)](https://twitter.com/diffuserslib)

[See the original repo](https://github.com/huggingface/diffusers)

🤗 Diffusers is your one-stop-shop for working with cutting-edge diffusion models, including image generation, audio synthesis, and even 3D molecule creation.  Built with a focus on *usability*, *simplicity*, and *customizability*, the library offers a modular approach for both inference and training, making it perfect for beginners and experts alike.

**Key Features:**

*   **Ready-to-Use Pipelines:** Generate outputs with ease using pre-built diffusion pipelines, perfect for quick results.
*   **Flexible Schedulers:** Experiment with different noise schedulers to control the speed and quality of your diffusion process.
*   **Modular Models:** Build custom diffusion systems by combining pretrained models and schedulers.
*   **Wide Range of Applications:** Explore text-to-image, image-to-image, inpainting, and more.
*   **Extensive Community Support:** Benefit from a thriving community and comprehensive documentation.

## Installation

Get started quickly by installing 🤗 Diffusers in a virtual environment using either `pip` or `conda`. Make sure you have PyTorch or Flax installed.

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

### Apple Silicon (M1/M2) Support

Refer to the [How to use Stable Diffusion in Apple Silicon](https://huggingface.co/docs/diffusers/optimization/mps) guide.

## Quickstart

Generate images from text in just a few lines of code, using a pretrained model from the [Hugging Face Hub](https://huggingface.co/models?library=diffusers&sort=downloads):

```python
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipeline.to("cuda")
pipeline("An image of a squirrel in Picasso style").images[0]
```

Or, build your own diffusion system using the building blocks:

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

Visit the [Quickstart](https://huggingface.co/docs/diffusers/quicktour) for a rapid introduction to diffusion models.

## Documentation Navigation

| **Documentation Section**                                         | **Learn About**                                                                                                                                                                                               |
|-------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Tutorial](https://huggingface.co/docs/diffusers/tutorials/tutorial_overview)                                                                     | Using core features like models and schedulers to create and train diffusion models.                                                                                                                                   |
| [Loading](https://huggingface.co/docs/diffusers/using-diffusers/loading)                                                                      | Loading and configuring pipelines, models, and schedulers, plus how to use various schedulers.                                                                                                                         |
| [Pipelines for Inference](https://huggingface.co/docs/diffusers/using-diffusers/overview_techniques)                                                        | Using pipelines for diverse inference tasks, batch generation, controlling outputs, and contributing pipelines.                                                                                                        |
| [Optimization](https://huggingface.co/docs/diffusers/optimization/fp16)                                                                       | Optimizing diffusion models for faster execution and lower memory consumption.                                                                                                                                          |
| [Training](https://huggingface.co/docs/diffusers/training/overview)                                                                       | Training diffusion models for diverse tasks, including different training methods.                                                                                                                                      |

## Contribute

We welcome contributions from the open-source community!  Check out our [Contribution guide](https://github.com/huggingface/diffusers/blob/main/CONTRIBUTING.md) and explore:

*   [Good first issues](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22)
*   [New model/pipeline](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22New+pipeline%2Fmodel%22)
*   [New scheduler](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22New+scheduler%22)

Join our Discord for discussions and support: <a href="https://discord.gg/G7tWnz98XR"><img alt="Join us on Discord" src="https://img.shields.io/discord/823813159592001537?color=5865F2&logo=discord&logoColor=white"></a>.

## Popular Tasks & Pipelines

| Task                        | Pipeline                                                                                                | Hugging Face Hub                                                                                        |
|-----------------------------|---------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|
| Unconditional Image Generation | [DDPM](https://huggingface.co/docs/diffusers/api/pipelines/ddpm)                                     | [google/ddpm-ema-church-256](https://huggingface.co/google/ddpm-ema-church-256)                        |
| Text-to-Image               | [Stable Diffusion Text-to-Image](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img) | [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) |
| Text-to-Image               | [unCLIP](https://huggingface.co/docs/diffusers/api/pipelines/unclip)                                   | [kakaobrain/karlo-v1-alpha](https://huggingface.co/kakaobrain/karlo-v1-alpha)                           |
| Text-to-Image               | [DeepFloyd IF](https://huggingface.co/docs/diffusers/api/pipelines/deepfloyd_if)                           | [DeepFloyd/IF-I-XL-v1.0](https://huggingface.co/DeepFloyd/IF-I-XL-v1.0)                                 |
| Text-to-Image               | [Kandinsky](https://huggingface.co/docs/diffusers/api/pipelines/kandinsky)                                | [kandinsky-community/kandinsky-2-2-decoder](https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder) |
| Text-guided Image-to-Image  | [ControlNet](https://huggingface.co/docs/diffusers/api/pipelines/controlnet)                            | [lllyasviel/sd-controlnet-canny](https://huggingface.co/lllyasviel/sd-controlnet-canny)                   |
| Text-guided Image-to-Image  | [InstructPix2Pix](https://huggingface.co/docs/diffusers/api/pipelines/pix2pix)                          | [timbrooks/instruct-pix2pix](https://huggingface.co/timbrooks/instruct-pix2pix)                         |
| Text-guided Image-to-Image  | [Stable Diffusion Image-to-Image](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/img2img) | [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) |
| Text-guided Image Inpainting| [Stable Diffusion Inpainting](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/inpaint) | [runwayml/stable-diffusion-inpainting](https://huggingface.co/runwayml/stable-diffusion-inpainting)          |
| Image Variation             | [Stable Diffusion Image Variation](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/image_variation) | [lambdalabs/sd-image-variations-diffusers](https://huggingface.co/lambdalabs/sd-image-variations-diffusers) |
| Super Resolution            | [Stable Diffusion Upscale](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/upscale)    | [stabilityai/stable-diffusion-x4-upscaler](https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler)    |
| Super Resolution            | [Stable Diffusion Latent Upscale](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/latent_upscale) | [stabilityai/sd-x2-latent-upscaler](https://huggingface.co/stabilityai/sd-x2-latent-upscaler)            |

## Used by these Amazing Projects

*   [Microsoft/TaskMatrix](https://github.com/microsoft/TaskMatrix)
*   [invoke-ai/InvokeAI](https://github.com/invoke-ai/InvokeAI)
*   [InstantID/InstantID](https://github.com/InstantID/InstantID)
*   [apple/ml-stable-diffusion](https://github.com/apple/ml-stable-diffusion)
*   [Sanster/lama-cleaner](https://github.com/Sanster/lama-cleaner)
*   [IDEA-Research/Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything)
*   [ashawkey/stable-dreamfusion](https://github.com/ashawkey/stable-dreamfusion)
*   [deep-floyd/IF](https://github.com/deep-floyd/IF)
*   [bentoml/BentoML](https://github.com/bentoml/BentoML)
*   [bmaltais/kohya_ss](https://github.com/bmaltais/kohya_ss)
*   And over 14,000 other GitHub repositories!

Thank you for using Diffusers!

## Credits

This library builds upon the work of numerous researchers and developers. We are particularly grateful for the contributions of:

*   [CompVis](https://github.com/CompVis/latent-diffusion)' latent diffusion models library
*   [hojonathanho](https://github.com/hojonathanho/diffusion) and [pesser](https://github.com/pesser/pytorch_diffusion)'s original DDPM implementation
*   [ermongroup](https://github.com/ermongroup/ddim)'s DDIM implementation
*   [yang-song](https://github.com/yang-song/score_sde_pytorch)'s Score-VE and Score-VP implementations

We also thank @heejkoo and @crowsonkb and @rromb for their insights and discussions.

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