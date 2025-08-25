<!-- Copyright 2022 - The HuggingFace Team. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. -->

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

# 🤗 Diffusers: Unleash the Power of Diffusion Models

**🤗 Diffusers is the premier open-source library for leveraging cutting-edge diffusion models to generate stunning images, audio, and 3D molecular structures.**  Dive into the world of generative AI with this comprehensive toolkit.  [See the original repo](https://github.com/huggingface/diffusers).

**Key Features:**

*   **Pre-trained Diffusion Pipelines:**  Easily generate content with state-of-the-art pipelines using just a few lines of code.
*   **Modular Components:** Access interchangeable noise schedulers for flexible diffusion speeds and output quality.
*   **Building Blocks for Customization:**  Utilize pre-trained models as modular components for building your own end-to-end diffusion systems.
*   **Usability-Focused Design:** Built with a focus on usability, simplicity, and customizability.

## Installation

Get started with 🤗 Diffusers by installing it in a virtual environment using either pip or Conda.  Ensure you have PyTorch and/or Flax installed, referring to their official documentation for details.

### PyTorch

*   **Pip:**

    ```bash
    pip install --upgrade diffusers[torch]
    ```

*   **Conda:**

    ```sh
    conda install -c conda-forge diffusers
    ```

### Flax

*   **Pip:**

    ```bash
    pip install --upgrade diffusers[flax]
    ```

### Apple Silicon (M1/M2) Support

Consult the  [How to use Stable Diffusion in Apple Silicon](https://huggingface.co/docs/diffusers/optimization/mps) guide for optimized performance on M1/M2 devices.

## Quickstart: Generate Images in Seconds

Generate images from text with just a few lines of code:

```python
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipeline.to("cuda")
pipeline("An image of a squirrel in Picasso style").images[0]
```

Alternatively, create your own diffusion systems by leveraging the models and schedulers toolbox:

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

Get started with the [Quickstart](https://huggingface.co/docs/diffusers/quicktour) today!

## Documentation & Learning Resources

Explore our comprehensive documentation to guide your diffusion journey:

| **Documentation Section**                                                   | **What You'll Learn**                                                                                                                                                                           |
|---------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Tutorial](https://huggingface.co/docs/diffusers/tutorials/tutorial_overview)                                                            | A crash course on using the library's key features including building and training your own diffusion systems.  |
| [Loading](https://huggingface.co/docs/diffusers/using-diffusers/loading)                                                             | Guides on loading, configuring and using the components of the library (pipelines, models, and schedulers).                                         |
| [Pipelines for inference](https://huggingface.co/docs/diffusers/using-diffusers/overview_techniques)                                             | Guides for using pipelines for various inference tasks, batch generation, controlling generated outputs and randomness. |
| [Optimization](https://huggingface.co/docs/diffusers/optimization/fp16)                                                        | Guides for optimizing your diffusion models for faster runtimes and lower memory consumption.                                                                                                          |
| [Training](https://huggingface.co/docs/diffusers/training/overview) | Guides for training diffusion models for various tasks.                                                                                               |

## Contribute

We welcome contributions from the open-source community!  Check out our [Contribution guide](https://github.com/huggingface/diffusers/blob/main/CONTRIBUTING.md) and browse existing [issues](https://github.com/huggingface/diffusers/issues) for ways to contribute.  Consider:

*   [Good first issues](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22)
*   [New model/pipeline](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22New+pipeline%2Fmodel%22)
*   [New scheduler](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22New+scheduler%22)

Join the conversation on our public Discord channel: <a href="https://discord.gg/G7tWnz98XR"><img alt="Join us on Discord" src="https://img.shields.io/discord/823813159592001537?color=5865F2&logo=discord&logoColor=white"></a>.

## Popular Tasks & Pipelines

Explore these popular tasks and pipelines:

| **Task**                    | **Pipeline**                                                     | **Hugging Face Hub**                                                                               |
| :-------------------------- | :--------------------------------------------------------------- | :------------------------------------------------------------------------------------------------- |
| Unconditional Image Generation | DDPM | [google/ddpm-ema-church-256](https://huggingface.co/google/ddpm-ema-church-256) |
| Text-to-Image               | Stable Diffusion Text-to-Image                                 | [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) |
| Text-to-Image               | unCLIP                                                                  | [kakaobrain/karlo-v1-alpha](https://huggingface.co/kakaobrain/karlo-v1-alpha)                                       |
| Text-to-Image               | DeepFloyd IF                                                       | [DeepFloyd/IF-I-XL-v1.0](https://huggingface.co/DeepFloyd/IF-I-XL-v1.0)                                       |
| Text-to-Image               | Kandinsky                                                        | [kandinsky-community/kandinsky-2-2-decoder](https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder)   |
| Text-guided Image-to-Image   | ControlNet                                                   | [lllyasviel/sd-controlnet-canny](https://huggingface.co/lllyasviel/sd-controlnet-canny)                                  |
| Text-guided Image-to-Image   | InstructPix2Pix                                                | [timbrooks/instruct-pix2pix](https://huggingface.co/timbrooks/instruct-pix2pix)                                     |
| Text-guided Image-to-Image   | Stable Diffusion Image-to-Image                              | [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) |
| Text-guided Image Inpainting   | Stable Diffusion Inpainting | [runwayml/stable-diffusion-inpainting](https://huggingface.co/runwayml/stable-diffusion-inpainting) |
| Image Variation             | Stable Diffusion Image Variation                              | [lambdalabs/sd-image-variations-diffusers](https://huggingface.co/lambdalabs/sd-image-variations-diffusers)             |
| Super Resolution             | Stable Diffusion Upscale                                         | [stabilityai/stable-diffusion-x4-upscaler](https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler)         |
| Super Resolution             | Stable Diffusion Latent Upscale                                 | [stabilityai/sd-x2-latent-upscaler](https://huggingface.co/stabilityai/sd-x2-latent-upscaler)                  |

## Used by Leading Projects

*   [Microsoft/TaskMatrix](https://github.com/microsoft/TaskMatrix)
*   [invoke-ai/InvokeAI](https://github.com/invoke-ai/InvokeAI)
*   [InstantID/InstantID](https://github.com/InstantID/InstantID)
*   [apple/ml-stable-diffusion](https://github.com/apple/ml-stable-diffusion)
*   [Sanster/lama-cleaner](https://github.com/Sanster/lama-cleaner)
*   [IDEA-Research/Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything)
*   [ashawkey/stable-dreamfusion](https://github.com/ashawkey/stable-dreamfusion)
*   [deep-floyd/IF](https://github.com/deep-floyd/IF)
*   [bentoml/BentoML](https://github.com/bentoml/BentoML)
*   [bmaltais/kohya\_ss](https://github.com/bmaltais/kohya_ss)
*   ...and +14,000 other amazing GitHub repositories!

Thank you for using 🤗 Diffusers.

## Credits

This library builds upon prior work and the contributions of numerous researchers and developers. We are especially grateful to the following implementations:

*   [@CompVis' latent diffusion models library](https://github.com/CompVis/latent-diffusion)
*   [@hojonathanho](https://github.com/hojonathanho/diffusion) original DDPM implementation
*   [@pesser](https://github.com/pesser/pytorch_diffusion)'s PyTorch translation
*   [@ermongroup's DDIM implementation](https://github.com/ermongroup/ddim)
*   [@yang-song's Score-VE and Score-VP implementations](https://github.com/yang-song/score_sde_pytorch)

We also thank @heejkoo for the helpful overview of papers, code, and resources on diffusion models, and @crowsonkb and @rromb for useful discussions and insights.

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