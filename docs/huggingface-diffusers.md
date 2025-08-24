# 🤗 Diffusers: Generate Images, Audio, and More with State-of-the-Art Diffusion Models

**Create stunning AI-generated content** with the 🤗 Diffusers library, your comprehensive toolbox for exploring and utilizing diffusion models. Find the original repository [here](https://github.com/huggingface/diffusers).

<p align="center">
    <br>
    <img src="https://raw.githubusercontent.com/huggingface/diffusers/main/docs/source/en/imgs/diffusers_library.jpg" width="400"/>
    <br>
<p>

[![License](https://img.shields.io/github/license/huggingface/datasets.svg?color=blue)](https://github.com/huggingface/diffusers/blob/main/LICENSE)
[![GitHub release](https://img.shields.io/github/release/huggingface/diffusers.svg)](https://github.com/huggingface/diffusers/releases)
[![PyPI monthly downloads](https://static.pepy.tech/badge/diffusers/month)](https://pepy.tech/project/diffusers)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/diffuserslib.svg?style=social&label=Follow%20%40diffuserslib)](https://twitter.com/diffuserslib)

## Key Features

*   **Wide Range of Applications:** Generate images, audio, and even 3D structures of molecules using pre-trained diffusion models.
*   **User-Friendly Pipelines:** Leverage ready-to-use pipelines for easy inference and quick results.
*   **Modular Architecture:** Build custom diffusion systems by combining interchangeable schedulers, models, and pipelines.
*   **Extensive Model Support:** Access a vast library of pre-trained models from the Hugging Face Hub.
*   **Customization Focus:** Designed with customizability and user contribution in mind.

## Core Components

*   **Diffusion Pipelines:** Ready-to-use pipelines for common tasks like text-to-image generation.
*   **Schedulers:** Implement diverse sampling methods (e.g., DDIM, PNDM) for varying speeds and image quality.
*   **Models:** Utilize pre-trained models (e.g., UNet) as building blocks for creating custom diffusion systems.

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

**Important:** Refer to the official documentation for [PyTorch](https://pytorch.org/get-started/locally/) and [Flax](https://flax.readthedocs.io/en/latest/#installation) for detailed installation instructions.  Also, check out the [Apple Silicon (M1/M2) support](https://huggingface.co/docs/diffusers/optimization/mps) guide.

## Quickstart

Generate images from text with ease:

```python
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipeline.to("cuda")
pipeline("An image of a squirrel in Picasso style").images[0]
```

Or build your own custom diffusion system:

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

Explore the [Quickstart](https://huggingface.co/docs/diffusers/quicktour) to start your diffusion journey.

## Documentation Navigation

| Documentation                                                         | What you can learn                                                                                                                                                                                                                                  |
| --------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Tutorial](https://huggingface.co/docs/diffusers/tutorials/tutorial_overview) | A beginner's guide to using the library's essential features, including building your own diffusion system and training your own diffusion model.                                                                                              |
| [Loading](https://huggingface.co/docs/diffusers/using-diffusers/loading) | Instructions on loading and configuring all library components (pipelines, models, and schedulers), as well as how to use different schedulers.                                                                                                   |
| [Pipelines for inference](https://huggingface.co/docs/diffusers/using-diffusers/overview_techniques) | Guides on using pipelines for various inference tasks, batch generation, controlling outputs and randomness, and how to contribute a pipeline.                                                                                         |
| [Optimization](https://huggingface.co/docs/diffusers/optimization/fp16) | How to optimize your diffusion model for faster execution and reduced memory consumption.                                                                                                                                                           |
| [Training](https://huggingface.co/docs/diffusers/training/overview) | Guides for training a diffusion model for various tasks with different training techniques.                                                                                                                                                           |

## Contribution

We welcome contributions! Check out the [Contribution guide](https://github.com/huggingface/diffusers/blob/main/CONTRIBUTING.md) for details. Explore the [issues](https://github.com/huggingface/diffusers/issues) to find areas to contribute.

*   [Good first issues](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22)
*   [New model/pipeline](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22New+pipeline%2Fmodel%22)
*   [New scheduler](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22New+scheduler%22)

Join our public Discord channel: <a href="https://discord.gg/G7tWnz98XR"><img alt="Join us on Discord" src="https://img.shields.io/discord/823813159592001537?color=5865F2&logo=discord&logoColor=white"></a> for discussions and support.

## Popular Tasks & Pipelines

| Task                       | Pipeline                                                                         | 🤗 Hub                                                                                             |
| -------------------------- | -------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| Unconditional Image Generation | [DDPM](https://huggingface.co/docs/diffusers/api/pipelines/ddpm)              | [google/ddpm-ema-church-256](https://huggingface.co/google/ddpm-ema-church-256)                      |
| Text-to-Image              | [Stable Diffusion Text-to-Image](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img) | [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) |
| Text-to-Image              | [unCLIP](https://huggingface.co/docs/diffusers/api/pipelines/unclip)               | [kakaobrain/karlo-v1-alpha](https://huggingface.co/kakaobrain/karlo-v1-alpha)                      |
| Text-to-Image              | [DeepFloyd IF](https://huggingface.co/docs/diffusers/api/pipelines/deepfloyd_if)     | [DeepFloyd/IF-I-XL-v1.0](https://huggingface.co/DeepFloyd/IF-I-XL-v1.0)                          |
| Text-to-Image              | [Kandinsky](https://huggingface.co/docs/diffusers/api/pipelines/kandinsky)          | [kandinsky-community/kandinsky-2-2-decoder](https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder) |
| Text-guided Image-to-Image | [ControlNet](https://huggingface.co/docs/diffusers/api/pipelines/controlnet)      | [lllyasviel/sd-controlnet-canny](https://huggingface.co/lllyasviel/sd-controlnet-canny)                |
| Text-guided Image-to-Image | [InstructPix2Pix](https://huggingface.co/docs/diffusers/api/pipelines/pix2pix)   | [timbrooks/instruct-pix2pix](https://huggingface.co/timbrooks/instruct-pix2pix)                  |
| Text-guided Image-to-Image | [Stable Diffusion Image-to-Image](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/img2img) | [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) |
| Text-guided Image Inpainting| [Stable Diffusion Inpainting](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/inpaint) | [runwayml/stable-diffusion-inpainting](https://huggingface.co/runwayml/stable-diffusion-inpainting)    |
| Image Variation            | [Stable Diffusion Image Variation](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/image_variation) | [lambdalabs/sd-image-variations-diffusers](https://huggingface.co/lambdalabs/sd-image-variations-diffusers)    |
| Super Resolution          | [Stable Diffusion Upscale](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/upscale) | [stabilityai/stable-diffusion-x4-upscaler](https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler)     |
| Super Resolution          | [Stable Diffusion Latent Upscale](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/latent_upscale) | [stabilityai/sd-x2-latent-upscaler](https://huggingface.co/stabilityai/sd-x2-latent-upscaler)        |

## Libraries Using 🤗 Diffusers (partial list)

*   [TaskMatrix](https://github.com/microsoft/TaskMatrix)
*   [InvokeAI](https://github.com/invoke-ai/InvokeAI)
*   [InstantID](https://github.com/InstantID/InstantID)
*   [ml-stable-diffusion](https://github.com/apple/ml-stable-diffusion)
*   [lama-cleaner](https://github.com/Sanster/lama-cleaner)
*   [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything)
*   [stable-dreamfusion](https://github.com/ashawkey/stable-dreamfusion)
*   [IF](https://github.com/deep-floyd/IF)
*   [BentoML](https://github.com/bentoml/BentoML)
*   [kohya\_ss](https://github.com/bmaltais/kohya_ss)
*   And many more!

Thank you for using us ❤️.

## Credits

This library is built upon the work of many researchers. We thank the following, whose implementations have aided our development:

*   @CompVis' latent diffusion models [here](https://github.com/CompVis/latent-diffusion)
*   @hojonathanho's original DDPM [here](https://github.com/hojonathanho/diffusion) and @pesser's PyTorch translation [here](https://github.com/pesser/pytorch_diffusion)
*   @ermongroup's DDIM [here](https://github.com/ermongroup/ddim)
*   @yang-song's Score-VE and Score-VP [here](https://github.com/yang-song/score_sde_pytorch)

We also thank @heejkoo for the overview of papers [here](https://github.com/heejkoo/Awesome-Diffusion-Models) and @crowsonkb and @rromb for useful discussions and insights.

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