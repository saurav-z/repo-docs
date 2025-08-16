<!--
Copyright 2022 - The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

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

# 🤗 Diffusers: Generate Images, Audio, and 3D Structures with Diffusion Models

**Unleash the power of diffusion models with 🤗 Diffusers, the leading library for generating cutting-edge content, including images, audio, and even 3D molecular structures.**  This library provides a modular and user-friendly toolkit for both inference and training, whether you're a beginner or an experienced researcher.  [Explore the original repository](https://github.com/huggingface/diffusers).

**Key Features:**

*   **State-of-the-Art Pipelines:** Quickly generate images, audio, and more with pre-trained diffusion models using intuitive pipelines.
*   **Flexible Schedulers:** Experiment with various noise schedulers to control the speed and quality of your diffusion processes.
*   **Modular Building Blocks:** Utilize pre-trained models and schedulers to construct custom diffusion systems tailored to your specific needs.
*   **Easy to Use:**  Prioritizes usability, simplicity, and customizability over performance and abstraction.

## Installation

Install 🤗 Diffusers in a virtual environment using either PyPI or Conda. Ensure you have PyTorch or Flax installed.

### PyTorch

Using `pip`:

```bash
pip install --upgrade diffusers[torch]
```

Using `conda`:

```sh
conda install -c conda-forge diffusers
```

### Flax

Using `pip`:

```bash
pip install --upgrade diffusers[flax]
```

### Apple Silicon (M1/M2) Support

Refer to the [Apple Silicon guide](https://huggingface.co/docs/diffusers/optimization/mps) for optimized performance.

## Quickstart

Generate stunning outputs effortlessly with 🤗 Diffusers. Load a pre-trained diffusion model (browse the [Hub](https://huggingface.co/models?library=diffusers&sort=downloads) for 30,000+ checkpoints) using the `from_pretrained` method:

```python
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipeline.to("cuda")
pipeline("An image of a squirrel in Picasso style").images[0]
```

Alternatively, build your own diffusion system using models and schedulers:

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

Start your diffusion journey today with the [Quickstart](https://huggingface.co/docs/diffusers/quicktour)!

## Documentation Overview

| Documentation                                                   | What You Can Learn                                                                                                                                                                                                                                                                                                                                                                     |
| :-------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Tutorial](https://huggingface.co/docs/diffusers/tutorials/tutorial_overview) | A comprehensive introduction to core features, including building and training your own diffusion models, and using models and schedulers to build your own diffusion system.                                                                                                                                                                                               |
| [Loading](https://huggingface.co/docs/diffusers/using-diffusers/loading)   | Learn to load and configure all library components (pipelines, models, and schedulers) and utilize different schedulers effectively.                                                                                                                                                                                                                                         |
| [Pipelines for inference](https://huggingface.co/docs/diffusers/using-diffusers/overview_techniques)    | Explore inference tasks, batch generation, output control, randomness management, and pipeline contribution guidelines.                                                                                                                                                                                                                   |
| [Optimization](https://huggingface.co/docs/diffusers/optimization/fp16)     | Optimize your diffusion models for faster performance and reduced memory consumption.                                                                                                                                                                                                                                                                           |
| [Training](https://huggingface.co/docs/diffusers/training/overview)       | Learn to train diffusion models for various tasks with diverse training techniques.                                                                                                                                                                                                                                                                                |

## Contribution

We welcome contributions from the open-source community!  See our [Contribution guide](https://github.com/huggingface/diffusers/blob/main/CONTRIBUTING.md). Find opportunities through [issues](https://github.com/huggingface/diffusers/issues), including:

*   [Good first issues](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22)
*   [New model/pipeline](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22New+pipeline%2Fmodel%22)
*   [New scheduler](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22New+scheduler%22)

Join the discussion on our public Discord channel: <a href="https://discord.gg/G7tWnz98XR"><img alt="Join us on Discord" src="https://img.shields.io/discord/823813159592001537?color=5865F2&logo=discord&logoColor=white"></a>.

## Popular Tasks & Pipelines

| Task                          | Pipeline                                                                                                                   | 🤗 Hub                                                                                                    |
| :---------------------------- | :------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------- |
| Unconditional Image Generation | [DDPM](https://huggingface.co/docs/diffusers/api/pipelines/ddpm)                                                      | [google/ddpm-ema-church-256](https://huggingface.co/google/ddpm-ema-church-256)                                         |
| Text-to-Image                | [Stable Diffusion Text-to-Image](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img)              | [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)              |
| Text-to-Image                | [unCLIP](https://huggingface.co/docs/diffusers/api/pipelines/unclip)                                                      | [kakaobrain/karlo-v1-alpha](https://huggingface.co/kakaobrain/karlo-v1-alpha)                                       |
| Text-to-Image                | [DeepFloyd IF](https://huggingface.co/docs/diffusers/api/pipelines/deepfloyd_if)                                            | [DeepFloyd/IF-I-XL-v1.0](https://huggingface.co/DeepFloyd/IF-I-XL-v1.0)                                         |
| Text-to-Image                | [Kandinsky](https://huggingface.co/docs/diffusers/api/pipelines/kandinsky)                                                | [kandinsky-community/kandinsky-2-2-decoder](https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder)               |
| Text-guided Image-to-Image    | [ControlNet](https://huggingface.co/docs/diffusers/api/pipelines/controlnet)                                              | [lllyasviel/sd-controlnet-canny](https://huggingface.co/lllyasviel/sd-controlnet-canny)                                 |
| Text-guided Image-to-Image    | [InstructPix2Pix](https://huggingface.co/docs/diffusers/api/pipelines/pix2pix)                                           | [timbrooks/instruct-pix2pix](https://huggingface.co/timbrooks/instruct-pix2pix)                                |
| Text-guided Image-to-Image    | [Stable Diffusion Image-to-Image](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/img2img)          | [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)          |
| Text-guided Image Inpainting  | [Stable Diffusion Inpainting](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/inpaint)             | [runwayml/stable-diffusion-inpainting](https://huggingface.co/runwayml/stable-diffusion-inpainting)            |
| Image Variation               | [Stable Diffusion Image Variation](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/image_variation) | [lambdalabs/sd-image-variations-diffusers](https://huggingface.co/lambdalabs/sd-image-variations-diffusers) |
| Super Resolution              | [Stable Diffusion Upscale](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/upscale)               | [stabilityai/stable-diffusion-x4-upscaler](https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler)      |
| Super Resolution              | [Stable Diffusion Latent Upscale](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/latent_upscale)    | [stabilityai/sd-x2-latent-upscaler](https://huggingface.co/stabilityai/sd-x2-latent-upscaler)                |

## Libraries Using 🧨 Diffusers

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
*   And over 14,000 other amazing GitHub repositories!

Thank you for using 🤗 Diffusers!

## Credits

This library is built upon the work of numerous researchers and developers. We are especially grateful to the following implementations:

*   @CompVis' latent diffusion models ([https://github.com/CompVis/latent-diffusion](https://github.com/CompVis/latent-diffusion))
*   @hojonathanho's original DDPM implementation ([https://github.com/hojonathanho/diffusion](https://github.com/hojonathanho/diffusion)) and @pesser's PyTorch translation ([https://github.com/pesser/pytorch_diffusion](https://github.com/pesser/pytorch_diffusion))
*   @ermongroup's DDIM implementation ([https://github.com/ermongroup/ddim](https://github.com/ermongroup/ddim))
*   @yang-song's Score-VE and Score-VP implementations ([https://github.com/yang-song/score_sde_pytorch](https://github.com/yang-song/score_sde_pytorch))

We also acknowledge @heejkoo for their helpful overview of diffusion model resources ([https://github.com/heejkoo/Awesome-Diffusion-Models](https://github.com/heejkoo/Awesome-Diffusion-Models)) and @crowsonkb and @rromb for their valuable discussions.

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