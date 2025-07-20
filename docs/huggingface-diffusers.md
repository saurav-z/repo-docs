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

# 🤗 Diffusers: Generate Amazing Content with State-of-the-Art Diffusion Models

**[Go to the original repository on GitHub](https://github.com/huggingface/diffusers)**

🤗 Diffusers is the leading library for working with cutting-edge diffusion models, offering a simple yet powerful way to generate images, audio, and even 3D structures.  Designed for usability and customization, it provides everything you need for both inference and training.

**Key Features:**

*   **Diffusion Pipelines:**  Ready-to-use pipelines for generating content with just a few lines of code.
*   **Flexible Schedulers:**  Choose from various noise schedulers to control diffusion speed and output quality.
*   **Modular Models:**  Use pretrained models as building blocks to create your own customized diffusion systems.

## Installation

Install 🤗 Diffusers in a virtual environment using either PyPI or Conda. Ensure you have installed PyTorch and/or Flax based on your framework preference.

### PyTorch

```bash
pip install --upgrade diffusers[torch]
```

or with `conda`:

```sh
conda install -c conda-forge diffusers
```

### Flax

```bash
pip install --upgrade diffusers[flax]
```

### Apple Silicon (M1/M2) support

Refer to the [How to use Stable Diffusion in Apple Silicon](https://huggingface.co/docs/diffusers/optimization/mps) guide.

## Quickstart

Generating outputs is easy with 🤗 Diffusers.  Load a pretrained model and generate an image from text:

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

Explore the [Quickstart](https://huggingface.co/docs/diffusers/quicktour) to begin your diffusion journey!

## Documentation Navigation

| Documentation                                                                 | What You'll Learn                                                                                                                                                                                                |
| :---------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Tutorial](https://huggingface.co/docs/diffusers/tutorials/tutorial_overview)                                                      | Learn to use the library's key features, build custom diffusion systems, and train your own models.                                                                         |
| [Loading](https://huggingface.co/docs/diffusers/using-diffusers/loading)                                                           | Guides for loading and configuring pipelines, models, and schedulers; explore different schedulers.                                                                               |
| [Pipelines for inference](https://huggingface.co/docs/diffusers/using-diffusers/overview_techniques)                                            |  How to use pipelines for inference tasks, batch generation, controlling outputs, and contributing pipelines.                                                                 |
| [Optimization](https://huggingface.co/docs/diffusers/optimization/fp16)                                                         | Guides to optimize your diffusion models for faster performance and reduced memory usage.                                                                                       |
| [Training](https://huggingface.co/docs/diffusers/training/overview)                                                              | Tutorials on training diffusion models for various tasks, including different training techniques.                                                                          |

## Contribution

We welcome contributions! See our [Contribution guide](https://github.com/huggingface/diffusers/blob/main/CONTRIBUTING.md).

Find opportunities in:

*   [Good first issues](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22)
*   [New model/pipeline](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22New+pipeline%2Fmodel%22)
*   [New scheduler](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22New+scheduler%22)

Join our Discord: <a href="https://discord.gg/G7tWnz98XR"><img alt="Join us on Discord" src="https://img.shields.io/discord/823813159592001537?color=5865F2&logo=discord&logoColor=white"></a>.

## Popular Tasks & Pipelines

| Task                      | Pipeline                                                                    | 🤗 Hub                                                                                              |
| :------------------------ | :-------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------- |
| Unconditional Image Generation | DDPM | [google/ddpm-ema-church-256](https://huggingface.co/google/ddpm-ema-church-256)  |
| Text-to-Image               | Stable Diffusion Text-to-Image                                            | [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) |
| Text-to-Image               | unCLIP                                                                     | [kakaobrain/karlo-v1-alpha](https://huggingface.co/kakaobrain/karlo-v1-alpha)                   |
| Text-to-Image               | DeepFloyd IF                                                              | [DeepFloyd/IF-I-XL-v1.0](https://huggingface.co/DeepFloyd/IF-I-XL-v1.0)                           |
| Text-to-Image               | Kandinsky                                                                  | [kandinsky-community/kandinsky-2-2-decoder](https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder)     |
| Text-guided Image-to-Image  | ControlNet                                                                 | [lllyasviel/sd-controlnet-canny](https://huggingface.co/lllyasviel/sd-controlnet-canny)              |
| Text-guided Image-to-Image  | InstructPix2Pix                                                            | [timbrooks/instruct-pix2pix](https://huggingface.co/timbrooks/instruct-pix2pix)                    |
| Text-guided Image-to-Image  | Stable Diffusion Image-to-Image                                            | [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) |
| Text-guided Image Inpainting | Stable Diffusion Inpainting                                                  | [runwayml/stable-diffusion-inpainting](https://huggingface.co/runwayml/stable-diffusion-inpainting)      |
| Image Variation             | Stable Diffusion Image Variation                                           | [lambdalabs/sd-image-variations-diffusers](https://huggingface.co/lambdalabs/sd-image-variations-diffusers)     |
| Super Resolution            | Stable Diffusion Upscale                                                   | [stabilityai/stable-diffusion-x4-upscaler](https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler)      |
| Super Resolution            | Stable Diffusion Latent Upscale                                            | [stabilityai/sd-x2-latent-upscaler](https://huggingface.co/stabilityai/sd-x2-latent-upscaler)     |

## Used by Amazing Libraries

Diffusers is used in many popular libraries, including:
*   [TaskMatrix](https://github.com/microsoft/TaskMatrix)
*   [InvokeAI](https://github.com/invoke-ai/InvokeAI)
*   [InstantID](https://github.com/InstantID/InstantID)
*   [ml-stable-diffusion](https://github.com/apple/ml-stable-diffusion)
*   [lama-cleaner](https://github.com/Sanster/lama-cleaner)
*   [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything)
*   [stable-dreamfusion](https://github.com/ashawkey/stable-dreamfusion)
*   [IF](https://github.com/deep-floyd/IF)
*   [BentoML](https://github.com/bentoml/BentoML)
*   [kohya_ss](https://github.com/bmaltais/kohya_ss)
*   and more than 14,000 other projects!

Thanks for using 🤗 Diffusers!

## Credits

We would like to thank the many authors and contributors whose research and implementations have made this library possible. Special thanks to:

*   @CompVis for their latent diffusion models library ([here](https://github.com/CompVis/latent-diffusion))
*   @hojonathanho for the original DDPM implementation ([here](https://github.com/hojonathanho/diffusion)) and @pesser for the PyTorch translation ([here](https://github.com/pesser/pytorch_diffusion))
*   @ermongroup for their DDIM implementation ([here](https://github.com/ermongroup/ddim))
*   @yang-song for their Score-VE and Score-VP implementations ([here](https://github.com/yang-song/score_sde_pytorch))

We also appreciate @heejkoo for the overview of diffusion models resources ([here](https://github.com/heejkoo/Awesome-Diffusion-Models)) and @crowsonkb and @rromb for their discussions.

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