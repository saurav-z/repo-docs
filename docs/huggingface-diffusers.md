<p align="center">
    <br>
    <img src="https://raw.githubusercontent.com/huggingface/diffusers/main/docs/source/en/imgs/diffusers_library.jpg" width="400"/>
    <br>
<p>

# 🤗 Diffusers: Your Gateway to Cutting-Edge Diffusion Models

**Unleash the power of diffusion models for image, audio, and 3D generation with Hugging Face's 🤗 Diffusers library!** Dive into the world of AI-powered content creation with a library designed for usability, simplicity, and customizability.  Explore a vast ecosystem of pre-trained models and create your own diffusion-based applications.  ([Original Repository](https://github.com/huggingface/diffusers))

<p align="center">
    <a href="https://github.com/huggingface/diffusers/blob/main/LICENSE"><img alt="GitHub" src="https://img.shields.io/github/license/huggingface/datasets.svg?color=blue"></a>
    <a href="https://github.com/huggingface/diffusers/releases"><img alt="GitHub release" src="https://img.shields.io/github/release/huggingface/diffusers.svg"></a>
    <a href="https://pepy.tech/project/diffusers"><img alt="GitHub release" src="https://static.pepy.tech/badge/diffusers/month"></a>
    <a href="CODE_OF_CONDUCT.md"><img alt="Contributor Covenant" src="https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg"></a>
    <a href="https://twitter.com/diffuserslib"><img alt="X account" src="https://img.shields.io/twitter/url/https/twitter.com/diffuserslib.svg?style=social&label=Follow%20%40diffuserslib"></a>
</p>

## Key Features

*   **Pre-trained Diffusion Pipelines:**  Quickly generate images, audio, and more with ready-to-use pipelines.
*   **Flexible Schedulers:** Experiment with various noise schedulers for diverse diffusion speeds and output qualities.
*   **Modular Models:** Utilize pre-trained models as building blocks to construct custom diffusion systems.
*   **Extensive Model Support:** Access a vast library of pre-trained models on the Hugging Face Hub.
*   **Easy Installation:** Simple setup with `pip` or `conda`.
*   **Apple Silicon Support:** Optimized for M1/M2 chips.
*   **Comprehensive Documentation:**  Easy-to-follow tutorials, guides, and examples to get you started quickly.
*   **Active Community:** Engage with a supportive community on Discord and through contributions.

## Installation

Install 🤗 Diffusers in a virtual environment using either PyPI or Conda. Ensure you have the necessary dependencies like PyTorch or Flax installed.

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

### Apple Silicon (M1/M2)

Refer to the [Apple Silicon guide](https://huggingface.co/docs/diffusers/optimization/mps) for optimization instructions.

## Quickstart: Generate Images in Seconds

Generate an image from text using a pre-trained model:

```python
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipeline.to("cuda")
pipeline("An image of a squirrel in Picasso style").images[0]
```

Alternatively, construct your own diffusion system using models and schedulers:

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

For a deeper dive, explore the [Quickstart](https://huggingface.co/docs/diffusers/quicktour) documentation.

## Documentation & Resources

Explore the documentation for comprehensive guides:

| Documentation                                                             | What You Can Learn                                                                                                                                                                                             |
| ------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Tutorial](https://huggingface.co/docs/diffusers/tutorials/tutorial_overview)                            | Master the library's core features, including model and scheduler utilization and training your own diffusion model.                                                                                 |
| [Loading](https://huggingface.co/docs/diffusers/using-diffusers/loading)                                 | Discover how to load and configure pipelines, models, and schedulers, and experiment with various schedulers.                                                                                     |
| [Pipelines for inference](https://huggingface.co/docs/diffusers/using-diffusers/overview_techniques)     | Learn about using pipelines for different inference tasks, including batch generation, controlling outputs and randomness, and contributing new pipelines.                                           |
| [Optimization](https://huggingface.co/docs/diffusers/optimization/fp16)                               | Optimize your diffusion models for enhanced speed and reduced memory usage.                                                                                                                       |
| [Training](https://huggingface.co/docs/diffusers/training/overview) | Develop your skills in training diffusion models for various tasks with different training methodologies.                                                                                                          |

## Contribute

We welcome contributions from the open-source community!  Consult the [Contribution guide](https://github.com/huggingface/diffusers/blob/main/CONTRIBUTING.md) for details.

*   Find [Good first issues](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22).
*   Explore [New model/pipeline](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22New+pipeline%2Fmodel%22) opportunities.
*   Discover [New scheduler](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22New+scheduler%22) projects.

Join the discussion on our public Discord channel:  <a href="https://discord.gg/G7tWnz98XR"><img alt="Join us on Discord" src="https://img.shields.io/discord/823813159592001537?color=5865F2&logo=discord&logoColor=white"></a>.

## Popular Tasks & Pipelines

Explore popular pre-trained models and pipelines for various tasks:

| Task                       | Pipeline                                                                                               | 🤗 Hub                                                                                                 |
| -------------------------- | ------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------- |
| Unconditional Image Generation | [DDPM](https://huggingface.co/docs/diffusers/api/pipelines/ddpm)                                      | [google/ddpm-ema-church-256](https://huggingface.co/google/ddpm-ema-church-256)                        |
| Text-to-Image              | [Stable Diffusion Text-to-Image](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img) | [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) |
| Text-to-Image              | [unCLIP](https://huggingface.co/docs/diffusers/api/pipelines/unclip)                                     | [kakaobrain/karlo-v1-alpha](https://huggingface.co/kakaobrain/karlo-v1-alpha)                          |
| Text-to-Image              | [DeepFloyd IF](https://huggingface.co/docs/diffusers/api/pipelines/deepfloyd_if)                          | [DeepFloyd/IF-I-XL-v1.0](https://huggingface.co/DeepFloyd/IF-I-XL-v1.0)                             |
| Text-to-Image              | [Kandinsky](https://huggingface.co/docs/diffusers/api/pipelines/kandinsky)                                | [kandinsky-community/kandinsky-2-2-decoder](https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder)     |
| Text-guided Image-to-Image | [ControlNet](https://huggingface.co/docs/diffusers/api/pipelines/controlnet)                             | [lllyasviel/sd-controlnet-canny](https://huggingface.co/lllyasviel/sd-controlnet-canny)                     |
| Text-guided Image-to-Image | [InstructPix2Pix](https://huggingface.co/docs/diffusers/api/pipelines/pix2pix)                           | [timbrooks/instruct-pix2pix](https://huggingface.co/timbrooks/instruct-pix2pix)                           |
| Text-guided Image-to-Image | [Stable Diffusion Image-to-Image](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/img2img) | [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) |
| Text-guided Image Inpainting | [Stable Diffusion Inpainting](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/inpaint) | [runwayml/stable-diffusion-inpainting](https://huggingface.co/runwayml/stable-diffusion-inpainting)        |
| Image Variation            | [Stable Diffusion Image Variation](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/image_variation) | [lambdalabs/sd-image-variations-diffusers](https://huggingface.co/lambdalabs/sd-image-variations-diffusers)     |
| Super Resolution           | [Stable Diffusion Upscale](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/upscale)     | [stabilityai/stable-diffusion-x4-upscaler](https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler)      |
| Super Resolution           | [Stable Diffusion Latent Upscale](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/latent_upscale) | [stabilityai/sd-x2-latent-upscaler](https://huggingface.co/stabilityai/sd-x2-latent-upscaler)           |

## Libraries using Diffusers

Explore other amazing projects utilizing 🤗 Diffusers:

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
*   And 14,000+ more!

## Credits

The development of this library builds upon the research and implementations of numerous individuals and organizations. We extend our gratitude to:

*   @CompVis for their latent diffusion models library.
*   @hojonathanho for the original DDPM implementation.
*   @pesser for translating the DDPM implementation into PyTorch.
*   @ermongroup for their DDIM implementation.
*   @yang-song for their Score-VE and Score-VP implementations.
*   @heejkoo for their comprehensive overview of diffusion models.
*   @crowsonkb and @rromb for their valuable discussions.

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