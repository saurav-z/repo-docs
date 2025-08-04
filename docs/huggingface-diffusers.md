# 🤗 Diffusers: Your Toolkit for State-of-the-Art Diffusion Models

Generate stunning images, audio, and 3D structures with ease using the Hugging Face 🤗 Diffusers library ([original repo](https://github.com/huggingface/diffusers)).  This comprehensive library empowers you to leverage the power of diffusion models for various creative applications.

<p align="center">
    <br>
    <img src="https://raw.githubusercontent.com/huggingface/diffusers/main/docs/source/en/imgs/diffusers_library.jpg" width="400"/>
    <br>
<p>

[![License](https://img.shields.io/github/license/huggingface/datasets.svg?color=blue)](https://github.com/huggingface/diffusers/blob/main/LICENSE)
[![GitHub release](https://img.shields.io/github/release/huggingface/diffusers.svg)](https://github.com/huggingface/diffusers/releases)
[![Monthly Downloads](https://static.pepy.tech/badge/diffusers/month)](https://pepy.tech/project/diffusers)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)
[![Follow on Twitter](https://img.shields.io/twitter/url/https/twitter.com/diffuserslib.svg?style=social&label=Follow%20%40diffuserslib)](https://twitter.com/diffuserslib)

## Key Features

*   **Versatile Pipelines:** Ready-to-use diffusion pipelines for image generation, text-to-image, and more, requiring only a few lines of code.
*   **Modular Components:** Build custom diffusion systems with interchangeable noise schedulers and pre-trained models.
*   **Extensive Model Support:** Access a vast collection of pre-trained models from the Hugging Face Hub for various tasks.
*   **Easy to Use:** Designed with a focus on simplicity and usability, making it accessible for beginners and experts.
*   **Highly Customizable:** Customize your diffusion models with the ability to choose schedulers and models, enabling advanced users to modify and create new models.
*   **Optimization Guides:** Learn about optimizing your diffusion models for faster inference and reduced memory consumption.
*   **Training Resources:** Train your own diffusion models with comprehensive guides on training and available training techniques.

## Installation

Install 🤗 Diffusers within a virtual environment using either `pip` or `conda`.

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

Refer to the [How to use Stable Diffusion in Apple Silicon](https://huggingface.co/docs/diffusers/optimization/mps) guide for details on optimizing performance on Apple Silicon.

## Quickstart: Generate Images with Ease

Get started generating outputs in just a few lines of code. Load any pre-trained diffusion model from the [Hugging Face Hub](https://huggingface.co/models?library=diffusers&sort=downloads) and start generating:

```python
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipeline.to("cuda")
pipeline("An image of a squirrel in Picasso style").images[0]
```

Alternatively, build your diffusion system by combining models and schedulers:

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

Explore the [Quickstart](https://huggingface.co/docs/diffusers/quicktour) documentation to quickly launch your diffusion journey.

## Navigating the Documentation

| Documentation                                                                           | What can I learn?                                                                                                                                                                                 |
| :-------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [Tutorial](https://huggingface.co/docs/diffusers/tutorials/tutorial_overview)                                                              | Learn how to use the library's important features to build a diffusion model and training your own diffusion model.   |
| [Loading](https://huggingface.co/docs/diffusers/using-diffusers/loading)                                                               | Discover how to load and configure all the library components, including pipelines, models, and schedulers, plus guidance on using different schedulers.                                                                        |
| [Pipelines for inference](https://huggingface.co/docs/diffusers/using-diffusers/overview_techniques)                                                     | Learn how to use pipelines for inference tasks, batched generation, controlling generated outputs, and how to contribute a pipeline to the library.               |
| [Optimization](https://huggingface.co/docs/diffusers/optimization/fp16)                                                     | Understand optimization techniques to run your diffusion model faster and consume less memory.                                                                                                                                         |
| [Training](https://huggingface.co/docs/diffusers/training/overview) | Guides on training a diffusion model with different techniques.                                                                                               |

## Contribution

Contribute to the 🤗 Diffusers community!  Check out our [Contribution guide](https://github.com/huggingface/diffusers/blob/main/CONTRIBUTING.md) for details on how to contribute.

Explore the [issues](https://github.com/huggingface/diffusers/issues) to find opportunities to contribute:
*   See [Good first issues](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22) for general opportunities.
*   See [New model/pipeline](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22New+pipeline%2Fmodel%22) to contribute new diffusion models / diffusion pipelines.
*   See [New scheduler](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22New+scheduler%22) to contribute to our scheduler capabilities.

Join the discussion on our public Discord channel: <a href="https://discord.gg/G7tWnz98XR"><img alt="Join us on Discord" src="https://img.shields.io/discord/823813159592001537?color=5865F2&logo=discord&logoColor=white"></a>.

## Popular Tasks & Pipelines

| Task                       | Pipeline                                                                                 | 🤗 Hub                                                                                                  |
| :------------------------- | :--------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------- |
| Unconditional Image Generation  | [DDPM](https://huggingface.co/docs/diffusers/api/pipelines/ddpm)                                                          | [google/ddpm-ema-church-256](https://huggingface.co/google/ddpm-ema-church-256)                                               |
| Text-to-Image              | [Stable Diffusion Text-to-Image](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img) | [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)              |
| Text-to-Image              | [unCLIP](https://huggingface.co/docs/diffusers/api/pipelines/unclip)                                                        | [kakaobrain/karlo-v1-alpha](https://huggingface.co/kakaobrain/karlo-v1-alpha)                                                 |
| Text-to-Image              | [DeepFloyd IF](https://huggingface.co/docs/diffusers/api/pipelines/deepfloyd_if)                                                      | [DeepFloyd/IF-I-XL-v1.0](https://huggingface.co/DeepFloyd/IF-I-XL-v1.0)                                         |
| Text-to-Image              | [Kandinsky](https://huggingface.co/docs/diffusers/api/pipelines/kandinsky)                                                       | [kandinsky-community/kandinsky-2-2-decoder](https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder)                                           |
| Text-guided Image-to-Image  | [ControlNet](https://huggingface.co/docs/diffusers/api/pipelines/controlnet)                                                  | [lllyasviel/sd-controlnet-canny](https://huggingface.co/lllyasviel/sd-controlnet-canny)                                            |
| Text-guided Image-to-Image  | [InstructPix2Pix](https://huggingface.co/docs/diffusers/api/pipelines/pix2pix)                                             | [timbrooks/instruct-pix2pix](https://huggingface.co/timbrooks/instruct-pix2pix)                                              |
| Text-guided Image-to-Image  | [Stable Diffusion Image-to-Image](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/img2img) | [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)              |
| Text-guided Image Inpainting | [Stable Diffusion Inpainting](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/inpaint)                    | [runwayml/stable-diffusion-inpainting](https://huggingface.co/runwayml/stable-diffusion-inpainting)                                         |
| Image Variation            | [Stable Diffusion Image Variation](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/image_variation)        | [lambdalabs/sd-image-variations-diffusers](https://huggingface.co/lambdalabs/sd-image-variations-diffusers)                                   |
| Super Resolution           | [Stable Diffusion Upscale](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/upscale)                             | [stabilityai/stable-diffusion-x4-upscaler](https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler)                               |
| Super Resolution           | [Stable Diffusion Latent Upscale](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/latent_upscale)       | [stabilityai/sd-x2-latent-upscaler](https://huggingface.co/stabilityai/sd-x2-latent-upscaler)                                     |

## Libraries using 🤗 Diffusers

-   [TaskMatrix](https://github.com/microsoft/TaskMatrix)
-   [InvokeAI](https://github.com/invoke-ai/InvokeAI)
-   [InstantID](https://github.com/InstantID/InstantID)
-   [ml-stable-diffusion](https://github.com/apple/ml-stable-diffusion)
-   [lama-cleaner](https://github.com/Sanster/lama-cleaner)
-   [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything)
-   [stable-dreamfusion](https://github.com/ashawkey/stable-dreamfusion)
-   [IF](https://github.com/deep-floyd/IF)
-   [BentoML](https://github.com/bentoml/BentoML)
-   [kohya\_ss](https://github.com/bmaltais/kohya_ss)
-   And over 14,000 other GitHub repositories.

Thank you for using 🤗 Diffusers.

## Credits

This library is built upon the foundational work of many researchers and developers. We extend our gratitude to:

*   @CompVis for the latent diffusion models library ([here](https://github.com/CompVis/latent-diffusion)).
*   @hojonathanho for the original DDPM implementation and @pesser for the PyTorch translation ([here](https://github.com/hojonathanho/diffusion) and [here](https://github.com/pesser/pytorch_diffusion)).
*   @ermongroup for the DDIM implementation ([here](https://github.com/ermongroup/ddim)).
*   @yang-song for the Score-VE and Score-VP implementations ([here](https://github.com/yang-song/score_sde_pytorch)).

We also thank @heejkoo for the overview of diffusion models and @crowsonkb and @rromb for discussions and insights.

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