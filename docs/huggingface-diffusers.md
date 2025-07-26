# 🤗 Diffusers: The Leading Library for Diffusion Models

Unlock the power of generative AI with **🤗 Diffusers**, a cutting-edge library for state-of-the-art pretrained diffusion models, available on [GitHub](https://github.com/huggingface/diffusers). Generate stunning images, audio, and even 3D molecular structures with ease.

<p align="center">
    <br>
    <img src="https://raw.githubusercontent.com/huggingface/diffusers/main/docs/source/en/imgs/diffusers_library.jpg" width="400"/>
    <br>
<p>

[![GitHub license](https://img.shields.io/github/license/huggingface/datasets.svg?color=blue)](https://github.com/huggingface/diffusers/blob/main/LICENSE)
[![GitHub release](https://img.shields.io/github/release/huggingface/diffusers.svg)](https://github.com/huggingface/diffusers/releases)
[![Pypi Downloads](https://static.pepy.tech/badge/diffusers/month)](https://pepy.tech/project/diffusers)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)
[![Follow on Twitter](https://img.shields.io/twitter/url/https/twitter.com/diffuserslib.svg?style=social&label=Follow%20%40diffuserslib)](https://twitter.com/diffuserslib)

**Key Features:**

*   **Intuitive Pipelines:** Run inference with pre-built pipelines for various tasks, requiring minimal code.
*   **Modular Design:** Access interchangeable schedulers and models for maximum flexibility and customization.
*   **Extensive Model Support:** Utilize a vast collection of pretrained models for diverse generative tasks, including image generation, audio synthesis, and 3D structure creation.
*   **Easy to Use:** The library is designed with a focus on usability, simplicity, and customizability.

**Get Started:**

## Installation

Install 🤗 Diffusers in a virtual environment using pip or conda.  Make sure you have [PyTorch](https://pytorch.org/get-started/locally/) and [Flax](https://flax.readthedocs.io/en/latest/#installation) installed.

### PyTorch

```bash
pip install --upgrade diffusers[torch]
```

### Flax

```bash
pip install --upgrade diffusers[flax]
```

### Apple Silicon (M1/M2) Support

For running models on Apple Silicon, refer to the  [How to use Stable Diffusion in Apple Silicon](https://huggingface.co/docs/diffusers/optimization/mps) guide.

## Quickstart: Generate Images from Text

Generate an image from text in a few lines of code:

```python
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipeline.to("cuda")
pipeline("An image of a squirrel in Picasso style").images[0]
```

**Explore the Toolbox**

You can also build your own diffusion system using models and schedulers:

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

Check out the [Quickstart](https://huggingface.co/docs/diffusers/quicktour) to begin your diffusion model journey!

## Documentation & Resources

| Documentation                                                        | What You Can Learn                                                                                                                                                                                                                   |
| :------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Tutorial](https://huggingface.co/docs/diffusers/tutorials/tutorial_overview)                                    | Learn how to use models and schedulers to build your own diffusion system and train your own diffusion model.                                                                                                       |
| [Loading](https://huggingface.co/docs/diffusers/using-diffusers/loading)                                          | Learn how to load and configure all the components (pipelines, models, and schedulers) of the library, and how to use different schedulers.                                                                         |
| [Pipelines for inference](https://huggingface.co/docs/diffusers/using-diffusers/overview_techniques)                                        | Learn how to use pipelines for inference tasks, batched generation, controlling generated outputs, and how to contribute a pipeline to the library.                                                               |
| [Optimization](https://huggingface.co/docs/diffusers/optimization/fp16)                                     | Optimize your diffusion model to run faster and consume less memory.                                                                                                                                                  |
| [Training](https://huggingface.co/docs/diffusers/training/overview) | Learn how to train a diffusion model for different tasks with different training techniques.                                                                                                                                          |

## Contribution

We welcome contributions from the community!  See our [Contribution guide](https://github.com/huggingface/diffusers/blob/main/CONTRIBUTING.md).

*   Explore [Good first issues](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22) for starting points.
*   Contribute to new models/pipelines with [New model/pipeline](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22New+pipeline%2Fmodel%22).
*   Add a [New scheduler](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22New+scheduler%22).

Join the discussion in our public Discord channel:  <a href="https://discord.gg/G7tWnz98XR"><img alt="Join us on Discord" src="https://img.shields.io/discord/823813159592001537?color=5865F2&logo=discord&logoColor=white"></a>.

## Popular Tasks & Pipelines

| Task                      | Pipeline                                                                                                            | 🤗 Hub                                                                                                        |
| :------------------------ | :------------------------------------------------------------------------------------------------------------------ | :------------------------------------------------------------------------------------------------------------- |
| Unconditional Image Generation | [DDPM](https://huggingface.co/docs/diffusers/api/pipelines/ddpm)                                                 | [google/ddpm-ema-church-256](https://huggingface.co/google/ddpm-ema-church-256)                                   |
| Text-to-Image           | [Stable Diffusion Text-to-Image](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img)    | [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)  |
| Text-to-Image           | [unCLIP](https://huggingface.co/docs/diffusers/api/pipelines/unclip)                                                | [kakaobrain/karlo-v1-alpha](https://huggingface.co/kakaobrain/karlo-v1-alpha)                                    |
| Text-to-Image           | [DeepFloyd IF](https://huggingface.co/docs/diffusers/api/pipelines/deepfloyd_if)                                         | [DeepFloyd/IF-I-XL-v1.0](https://huggingface.co/DeepFloyd/IF-I-XL-v1.0)                                         |
| Text-to-Image           | [Kandinsky](https://huggingface.co/docs/diffusers/api/pipelines/kandinsky)                                               | [kandinsky-community/kandinsky-2-2-decoder](https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder)       |
| Text-guided Image-to-Image  | [ControlNet](https://huggingface.co/docs/diffusers/api/pipelines/controlnet)                                           | [lllyasviel/sd-controlnet-canny](https://huggingface.co/lllyasviel/sd-controlnet-canny)                             |
| Text-guided Image-to-Image  | [InstructPix2Pix](https://huggingface.co/docs/diffusers/api/pipelines/pix2pix)                                       | [timbrooks/instruct-pix2pix](https://huggingface.co/timbrooks/instruct-pix2pix)                                   |
| Text-guided Image-to-Image  | [Stable Diffusion Image-to-Image](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/img2img)  | [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)  |
| Text-guided Image Inpainting | [Stable Diffusion Inpainting](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/inpaint)       | [runwayml/stable-diffusion-inpainting](https://huggingface.co/runwayml/stable-diffusion-inpainting)           |
| Image Variation         | [Stable Diffusion Image Variation](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/image_variation) | [lambdalabs/sd-image-variations-diffusers](https://huggingface.co/lambdalabs/sd-image-variations-diffusers)      |
| Super Resolution        | [Stable Diffusion Upscale](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/upscale)                 | [stabilityai/stable-diffusion-x4-upscaler](https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler)     |
| Super Resolution        | [Stable Diffusion Latent Upscale](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/latent_upscale)  | [stabilityai/sd-x2-latent-upscaler](https://huggingface.co/stabilityai/sd-x2-latent-upscaler)                   |

## Popular Libraries Using Diffusers

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

This library builds upon the work of many researchers.  We are especially grateful for the contributions of:

*   @CompVis' latent diffusion models library, [here](https://github.com/CompVis/latent-diffusion)
*   @hojonathanho's original DDPM implementation, [here](https://github.com/hojonathanho/diffusion) and @pesser's PyTorch translation, [here](https://github.com/pesser/pytorch_diffusion)
*   @ermongroup's DDIM implementation, [here](https://github.com/ermongroup/ddim)
*   @yang-song's Score-VE and Score-VP implementations, [here](https://github.com/yang-song/score_sde_pytorch)

We also thank @heejkoo for his overview of diffusion models, [here](https://github.com/heejkoo/Awesome-Diffusion-Models), and @crowsonkb and @rromb for their discussions.

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