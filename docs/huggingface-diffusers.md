<p align="center">
    <br>
    <img src="https://raw.githubusercontent.com/huggingface/diffusers/main/docs/source/en/imgs/diffusers_library.jpg" width="400"/>
    <br>
</p>

<p align="center">
    <a href="https://github.com/huggingface/diffusers/blob/main/LICENSE"><img alt="GitHub" src="https://img.shields.io/github/license/huggingface/datasets.svg?color=blue"></a>
    <a href="https://github.com/huggingface/diffusers/releases"><img alt="GitHub release" src="https://img.shields.io/github/release/huggingface/diffusers.svg"></a>
    <a href="https://pepy.tech/project/diffusers"><img alt="GitHub release" src="https://static.pepy.tech/badge/diffusers/month"></a>
    <a href="CODE_OF_CONDUCT.md"><img alt="Contributor Covenant" src="https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg"></a>
    <a href="https://twitter.com/diffuserslib"><img alt="X account" src="https://img.shields.io/twitter/url/https/twitter.com/diffuserslib.svg?style=social&label=Follow%20%40diffuserslib"></a>
</p>

# 🤗 Diffusers: Unleash the Power of Diffusion Models

**🤗 Diffusers is the leading Python library for utilizing and experimenting with cutting-edge diffusion models.**  

Explore the original [Diffusers repository](https://github.com/huggingface/diffusers).

## Key Features

*   **State-of-the-Art Diffusion Pipelines:** Quickly generate images, audio, and 3D structures using pre-trained models.
*   **Modular Design:** Build custom diffusion systems with interchangeable noise schedulers and models.
*   **Extensive Model Support:** Access a wide range of pre-trained models for various tasks like text-to-image, image-to-image, and more.
*   **User-Friendly:** Focus on usability, simplicity, and customization for both beginners and advanced users.
*   **Optimized for Performance:**  Includes optimization guides to help you run your diffusion models faster and more efficiently.
*   **Comprehensive Documentation:** Detailed documentation with tutorials, guides, and API references.
*   **Active Community:** Join our Discord to discuss trends, get help, and contribute to the library.

## Installation

Install 🤗 Diffusers in a virtual environment using `pip` or `conda`.  Ensure you have PyTorch and Flax installed.

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

Refer to the [Apple Silicon guide](https://huggingface.co/docs/diffusers/optimization/mps) for optimized performance.

## Quickstart

Generate images from text with just a few lines of code:

```python
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipeline.to("cuda")
pipeline("An image of a squirrel in Picasso style").images[0]
```

Or build a custom diffusion system:

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

Explore the [Quickstart](https://huggingface.co/docs/diffusers/quicktour) for more examples.

## Documentation Overview

| Documentation | What can I learn? |
|---|---|
| [Tutorial](https://huggingface.co/docs/diffusers/tutorials/tutorial_overview) | Use models and schedulers to build and train your own diffusion systems. |
| [Loading](https://huggingface.co/docs/diffusers/using-diffusers/loading) | Load and configure all components, and use different schedulers. |
| [Pipelines for inference](https://huggingface.co/docs/diffusers/using-diffusers/overview_techniques) | Use pipelines, batch generation, and control outputs. |
| [Optimization](https://huggingface.co/docs/diffusers/optimization/fp16) | Optimize your diffusion models for speed and memory usage. |
| [Training](https://huggingface.co/docs/diffusers/training/overview) | Train diffusion models for different tasks. |

## Contribution

We welcome contributions! See the [Contribution guide](https://github.com/huggingface/diffusers/blob/main/CONTRIBUTING.md) and the [issues](https://github.com/huggingface/diffusers/issues).
Join our public Discord:  <a href="https://discord.gg/G7tWnz98XR"><img alt="Join us on Discord" src="https://img.shields.io/discord/823813159592001537?color=5865F2&logo=discord&logoColor=white"></a>.

## Popular Tasks & Pipelines

| Task | Pipeline | 🤗 Hub |
|---|---|---|
| Unconditional Image Generation | DDPM | [google/ddpm-ema-church-256](https://huggingface.co/google/ddpm-ema-church-256) |
| Text-to-Image | Stable Diffusion Text-to-Image | [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) |
| Text-to-Image | unCLIP | [kakaobrain/karlo-v1-alpha](https://huggingface.co/kakaobrain/karlo-v1-alpha) |
| Text-to-Image | DeepFloyd IF | [DeepFloyd/IF-I-XL-v1.0](https://huggingface.co/DeepFloyd/IF-I-XL-v1.0) |
| Text-to-Image | Kandinsky | [kandinsky-community/kandinsky-2-2-decoder](https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder) |
| Text-guided Image-to-Image | ControlNet | [lllyasviel/sd-controlnet-canny](https://huggingface.co/lllyasviel/sd-controlnet-canny) |
| Text-guided Image-to-Image | InstructPix2Pix | [timbrooks/instruct-pix2pix](https://huggingface.co/timbrooks/instruct-pix2pix) |
| Text-guided Image-to-Image | Stable Diffusion Image-to-Image | [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) |
| Text-guided Image Inpainting | Stable Diffusion Inpainting | [runwayml/stable-diffusion-inpainting](https://huggingface.co/runwayml/stable-diffusion-inpainting) |
| Image Variation | Stable Diffusion Image Variation | [lambdalabs/sd-image-variations-diffusers](https://huggingface.co/lambdalabs/sd-image-variations-diffusers) |
| Super Resolution | Stable Diffusion Upscale | [stabilityai/stable-diffusion-x4-upscaler](https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler) |
| Super Resolution | Stable Diffusion Latent Upscale | [stabilityai/sd-x2-latent-upscaler](https://huggingface.co/stabilityai/sd-x2-latent-upscaler) |

## Used by Thousands of Repositories

-   [microsoft/TaskMatrix](https://github.com/microsoft/TaskMatrix)
-   [invoke-ai/InvokeAI](https://github.com/invoke-ai/InvokeAI)
-   [InstantID/InstantID](https://github.com/InstantID/InstantID)
-   [apple/ml-stable-diffusion](https://github.com/apple/ml-stable-diffusion)
-   [Sanster/lama-cleaner](https://github.com/Sanster/lama-cleaner)
-   [IDEA-Research/Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything)
-   [ashawkey/stable-dreamfusion](https://github.com/ashawkey/stable-dreamfusion)
-   [deep-floyd/IF](https://github.com/deep-floyd/IF)
-   [bentoml/BentoML](https://github.com/bentoml/BentoML)
-   [bmaltais/kohya_ss](https://github.com/bmaltais/kohya_ss)
-   +14,000 other amazing GitHub repositories 💪

## Credits

Thank you to the researchers and developers whose work inspired and enabled this library.

-   @CompVis for latent diffusion models.
-   @hojonathanho and @pesser for the original DDPM and PyTorch implementations.
-   @ermongroup for the DDIM implementation.
-   @yang-song for Score-VE and Score-VP implementations.

Also, thanks to @heejkoo for the overview of papers, code, and resources and @crowsonkb and @rromb for useful discussions.

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