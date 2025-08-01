# 🤗 Diffusers: Your Gateway to Cutting-Edge Diffusion Models

Unleash the power of generative AI with 🤗 Diffusers, the leading library for state-of-the-art diffusion models, enabling you to create stunning images, audio, and even 3D molecular structures. [Explore the Diffusers library on GitHub](https://github.com/huggingface/diffusers).

<p align="center">
    <img src="https://raw.githubusercontent.com/huggingface/diffusers/main/docs/source/en/imgs/diffusers_library.jpg" width="400"/>
</p>

<p align="center">
    <a href="https://github.com/huggingface/diffusers/blob/main/LICENSE"><img alt="GitHub" src="https://img.shields.io/github/license/huggingface/datasets.svg?color=blue"></a>
    <a href="https://github.com/huggingface/diffusers/releases"><img alt="GitHub release" src="https://img.shields.io/github/release/huggingface/diffusers.svg"></a>
    <a href="https://pepy.tech/project/diffusers"><img alt="GitHub release" src="https://static.pepy.tech/badge/diffusers/month"></a>
    <a href="CODE_OF_CONDUCT.md"><img alt="Contributor Covenant" src="https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg"></a>
    <a href="https://twitter.com/diffuserslib"><img alt="X account" src="https://img.shields.io/twitter/url/https/twitter.com/diffuserslib.svg?style=social&label=Follow%20%40diffuserslib"></a>
</p>

## Key Features

*   **Diffusion Pipelines:** Ready-to-use pipelines for image, audio, and 3D generation.
*   **Flexible Schedulers:** Experiment with diverse schedulers for optimal performance.
*   **Modular Models:**  Combine pretrained models and schedulers to build custom diffusion systems.
*   **User-Friendly Design:** Emphasizing usability, simplicity, and customizability.
*   **Extensive Hub Integration:** Access thousands of pretrained models on the Hugging Face Hub.
*   **Community Driven:**  Benefit from active community contributions and support.

## Installation

Install 🤗 Diffusers in a virtual environment using pip or conda:

### PyTorch

```bash
pip install --upgrade diffusers[torch]
```

### Flax

```bash
pip install --upgrade diffusers[flax]
```

For installation with conda, and details on Apple Silicon (M1/M2) support, refer to the [original README](https://github.com/huggingface/diffusers) and linked documentation.

## Quickstart

Generate an image from text using a pretrained model:

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

## Documentation

*   **[Tutorial](https://huggingface.co/docs/diffusers/tutorials/tutorial_overview)**: Get started with the basics.
*   **[Loading](https://huggingface.co/docs/diffusers/using-diffusers/loading)**:  Learn how to load and configure components.
*   **[Pipelines for inference](https://huggingface.co/docs/diffusers/using-diffusers/overview_techniques)**: Use pipelines for various tasks.
*   **[Optimization](https://huggingface.co/docs/diffusers/optimization/fp16)**: Improve speed and reduce memory usage.
*   **[Training](https://huggingface.co/docs/diffusers/training/overview)**: Train your own diffusion models.

## Contribution

We welcome contributions!  See the [Contribution guide](https://github.com/huggingface/diffusers/blob/main/CONTRIBUTING.md).

## Popular Tasks & Pipelines

| Task                        | Pipeline                                                                              | 🤗 Hub                                                                                                  |
| :-------------------------- | :------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------ |
| Unconditional Image Generation | DDPM                                                                                 | [google/ddpm-ema-church-256](https://huggingface.co/google/ddpm-ema-church-256)                        |
| Text-to-Image               | Stable Diffusion Text-to-Image                                                         | [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) |
| Text-to-Image               | unCLIP                                                                                 | [kakaobrain/karlo-v1-alpha](https://huggingface.co/kakaobrain/karlo-v1-alpha)                          |
| Text-to-Image               | DeepFloyd IF                                                                           | [DeepFloyd/IF-I-XL-v1.0](https://huggingface.co/DeepFloyd/IF-I-XL-v1.0)                                  |
| Text-to-Image               | Kandinsky                                                                              | [kandinsky-community/kandinsky-2-2-decoder](https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder) |
| Text-guided Image-to-Image  | ControlNet                                                                             | [lllyasviel/sd-controlnet-canny](https://huggingface.co/lllyasviel/sd-controlnet-canny)                  |
| Text-guided Image-to-Image  | InstructPix2Pix                                                                        | [timbrooks/instruct-pix2pix](https://huggingface.co/timbrooks/instruct-pix2pix)                        |
| Text-guided Image-to-Image  | Stable Diffusion Image-to-Image                                                        | [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) |
| Text-guided Image Inpainting| Stable Diffusion Inpainting                                                            | [runwayml/stable-diffusion-inpainting](https://huggingface.co/runwayml/stable-diffusion-inpainting)  |
| Image Variation             | Stable Diffusion Image Variation                                                       | [lambdalabs/sd-image-variations-diffusers](https://huggingface.co/lambdalabs/sd-image-variations-diffusers) |
| Super Resolution            | Stable Diffusion Upscale                                                               | [stabilityai/stable-diffusion-x4-upscaler](https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler) |
| Super Resolution            | Stable Diffusion Latent Upscale                                                        | [stabilityai/sd-x2-latent-upscaler](https://huggingface.co/stabilityai/sd-x2-latent-upscaler)          |

## Libraries using Diffusers

- Microsoft/TaskMatrix
- InvokeAI
- InstantID
- Apple/ml-stable-diffusion
- Sanster/lama-cleaner
- IDEA-Research/Grounded-Segment-Anything
- ashawkey/stable-dreamfusion
- deep-floyd/IF
- bentoml/BentoML
- bmaltais/kohya_ss
- And 14,000+ other amazing GitHub repositories 💪

## Credits

This library builds upon the work of many researchers and developers.  We are grateful for their contributions.

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
```

Key improvements and explanations:

*   **SEO Optimization:**
    *   Keywords: Incorporated relevant keywords like "diffusion models," "image generation," "text-to-image," "pretrained models," "generative AI," and the library name "Diffusers."
    *   Headings:  Used clear, descriptive headings (H1, H2, H3) to structure the information.
    *   Concise language: Avoided overly verbose descriptions.
*   **Summary & Hook:**  The opening sentence is a strong hook to capture attention.
*   **Key Features (Bulleted):** Presented the main benefits in an easy-to-scan bulleted list.
*   **Clear Structure:**  Organized the information logically with distinct sections.
*   **Action-Oriented:**  Included calls to action (e.g., "Explore the Diffusers library").
*   **Links:** Added links to relevant resources (e.g., the GitHub repo, documentation sections).
*   **Conciseness:**  Removed redundant phrases and streamlined the text.
*   **Formatted Code Blocks:** Made code examples more readable with proper formatting.
*   **Table for Tasks & Pipelines:** Improved readability and organization with a table showing popular tasks and pipelines.
*   **Credits Section:** Kept the credits section as is, as it's important for acknowledging the original work.
*   **Citation:** Included the citation in a code block, for easy copy-pasting.
*   **Community Aspect:** Highlighted the contributions of the community.
*   **Removed Licensing Information:** Removed license and copyright information to keep the README concise and focused on the library's use and features. (This information is linked at the top.)