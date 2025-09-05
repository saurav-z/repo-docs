# 🤗 Diffusers: Generate Images, Audio, and 3D Structures with Diffusion Models

**Unlock the power of cutting-edge diffusion models for creative generation with the 🤗 Diffusers library!** [Link to Original Repo](https://github.com/huggingface/diffusers)

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

🤗 Diffusers is the premier library for working with pretrained diffusion models, offering a comprehensive toolkit for both inference and training.  This library prioritizes usability, simplicity, and customizability to empower users of all levels.

**Key Features:**

*   **Diffusion Pipelines:** Easily generate images, audio, and 3D structures with pre-built pipelines.
*   **Flexible Schedulers:** Experiment with different noise schedulers to control the speed and quality of your diffusion process.
*   **Modular Models:** Leverage pre-trained models as building blocks to create your own custom diffusion systems.
*   **Extensive Model Support:** Access a vast range of pre-trained models (30,000+ checkpoints) on the Hugging Face Hub.
*   **Optimized for Performance:**  Optimization guides are available to help you run models faster and use less memory.

## Installation

Install 🤗 Diffusers in a virtual environment using either `pip` or `conda`. Ensure you have PyTorch installed (refer to PyTorch's official documentation).

### PyTorch

```bash
pip install --upgrade diffusers[torch]
```

### Conda

```sh
conda install -c conda-forge diffusers
```

### Apple Silicon (M1/M2) Support

Follow the guide on [How to use Stable Diffusion in Apple Silicon](https://huggingface.co/docs/diffusers/optimization/mps) for optimized performance.

## Quickstart

Get started quickly with image generation:

```python
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipeline.to("cuda")
pipeline("An image of a squirrel in Picasso style").images[0]
```

Build your own systems using models and schedulers:

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

Explore the [Quickstart](https://huggingface.co/docs/diffusers/quicktour) for a guided introduction.

## Documentation Overview

| **Documentation Section**                                                   | **What You'll Learn**                                                                                                                                                                           |
|---------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Tutorial](https://huggingface.co/docs/diffusers/tutorials/tutorial_overview)                                                            | Learn the basics of using models and schedulers to build diffusion systems and train your own models. |
| [Loading](https://huggingface.co/docs/diffusers/using-diffusers/loading)                                                             | Learn how to load and configure pipelines, models, and schedulers and use different schedulers.                                       |
| [Pipelines for inference](https://huggingface.co/docs/diffusers/using-diffusers/overview_techniques)                                             | Learn to use pipelines for inference, including batch generation, output control, and contributing a pipeline.               |
| [Optimization](https://huggingface.co/docs/diffusers/optimization/fp16)                                                        | Optimize your diffusion models for speed and memory efficiency.                                                                                                          |
| [Training](https://huggingface.co/docs/diffusers/training/overview) | Learn how to train diffusion models for various tasks using different techniques.                                                                                               |

## Contribute

Join the community and contribute to 🤗 Diffusers!

*   Follow the [Contribution guide](https://github.com/huggingface/diffusers/blob/main/CONTRIBUTING.md).
*   Explore [Good first issues](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22).
*   Contribute a [New pipeline/model](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22New+pipeline%2Fmodel%22).
*   Add a [New scheduler](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22New+scheduler%22).

Connect with the community on our [Discord](https://discord.gg/G7tWnz98XR).

## Popular Tasks & Pipelines

| Task                         | Pipeline                                                                                              | 🤗 Hub                                                                                               |
| :--------------------------- | :---------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------- |
| Unconditional Image Generation | [DDPM](https://huggingface.co/docs/diffusers/api/pipelines/ddpm)                                | [google/ddpm-ema-church-256](https://huggingface.co/google/ddpm-ema-church-256)                                  |
| Text-to-Image                | [Stable Diffusion Text-to-Image](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img) | [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) |
| Text-to-Image                | [unCLIP](https://huggingface.co/docs/diffusers/api/pipelines/unclip)                                 | [kakaobrain/karlo-v1-alpha](https://huggingface.co/kakaobrain/karlo-v1-alpha)                               |
| Text-to-Image                | [DeepFloyd IF](https://huggingface.co/docs/diffusers/api/pipelines/deepfloyd_if)                             | [DeepFloyd/IF-I-XL-v1.0](https://huggingface.co/DeepFloyd/IF-I-XL-v1.0)                                   |
| Text-to-Image                | [Kandinsky](https://huggingface.co/docs/diffusers/api/pipelines/kandinsky)                             | [kandinsky-community/kandinsky-2-2-decoder](https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder) |
| Text-guided Image-to-Image   | [ControlNet](https://huggingface.co/docs/diffusers/api/pipelines/controlnet)                           | [lllyasviel/sd-controlnet-canny](https://huggingface.co/lllyasviel/sd-controlnet-canny)                    |
| Text-guided Image-to-Image   | [InstructPix2Pix](https://huggingface.co/docs/diffusers/api/pipelines/pix2pix)                         | [timbrooks/instruct-pix2pix](https://huggingface.co/timbrooks/instruct-pix2pix)                          |
| Text-guided Image-to-Image   | [Stable Diffusion Image-to-Image](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/img2img) | [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) |
| Text-guided Image Inpainting  | [Stable Diffusion Inpainting](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/inpaint) | [runwayml/stable-diffusion-inpainting](https://huggingface.co/runwayml/stable-diffusion-inpainting)          |
| Image Variation               | [Stable Diffusion Image Variation](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/image_variation) | [lambdalabs/sd-image-variations-diffusers](https://huggingface.co/lambdalabs/sd-image-variations-diffusers)  |
| Super Resolution              | [Stable Diffusion Upscale](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/upscale)      | [stabilityai/stable-diffusion-x4-upscaler](https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler)  |
| Super Resolution              | [Stable Diffusion Latent Upscale](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/latent_upscale)    | [stabilityai/sd-x2-latent-upscaler](https://huggingface.co/stabilityai/sd-x2-latent-upscaler)           |

## Used by

The library is used by a lot of amazing projects 💪, including (but not limited to):

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
*   And many more (14,000+ GitHub repositories)

Thank you for using 🤗 Diffusers!

## Credits

We appreciate the work of many authors for making this library possible:

*   [CompVis](https://github.com/CompVis/latent-diffusion) for latent diffusion models.
*   [hojonathanho](https://github.com/hojonathanho/diffusion) for the original DDPM implementation.
*   [pesser](https://github.com/pesser/pytorch_diffusion) for the PyTorch translation of DDPM.
*   [ermongroup](https://github.com/ermongroup/ddim) for the DDIM implementation.
*   [yang-song](https://github.com/yang-song/score_sde_pytorch) for Score-VE and Score-VP implementations.

We also thank @heejkoo and @crowsonkb and @rromb for their contributions and insights.

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