# 🤗 Diffusers: Generate Images, Audio, and More with Diffusion Models

**🤗 Diffusers is your one-stop library for easily using and customizing state-of-the-art diffusion models for a wide range of creative applications.**  Built by Hugging Face, this library provides the tools you need, whether you're generating images from text, manipulating audio, or exploring 3D molecular structures. ([Original Repo](https://github.com/huggingface/diffusers))

[![License](https://img.shields.io/github/license/huggingface/datasets.svg?color=blue)](https://github.com/huggingface/diffusers/blob/main/LICENSE)
[![GitHub Release](https://img.shields.io/github/release/huggingface/diffusers.svg)](https://github.com/huggingface/diffusers/releases)
[![PyPI Downloads](https://static.pepy.tech/badge/diffusers/month)](https://pepy.tech/project/diffusers)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/diffuserslib.svg?style=social&label=Follow%20%40diffuserslib)](https://twitter.com/diffuserslib)

**Key Features:**

*   **Simplified Inference:** Run pre-trained diffusion models with just a few lines of code using ready-to-use [diffusion pipelines](https://huggingface.co/docs/diffusers/api/pipelines/overview).
*   **Flexible Customization:**  Use interchangeable [schedulers](https://huggingface.co/docs/diffusers/api/schedulers/overview) to control diffusion speed and output quality.
*   **Modular Design:** Leverage pre-trained [models](https://huggingface.co/docs/diffusers/api/models/overview) as building blocks to create your own custom diffusion systems.
*   **Focus on Usability:** The library prioritizes a user-friendly experience without sacrificing the ability to customize and optimize your workflows.
*   **Extensive Model Hub:** Access thousands of pre-trained models on the Hugging Face Hub (browse the [Hub](https://huggingface.co/models?library=diffusers&sort=downloads) for 30,000+ checkpoints).

## Installation

Install 🤗 Diffusers in a virtual environment using pip or conda:

### PyTorch

```bash
pip install --upgrade diffusers[torch]
```

or with conda:

```sh
conda install -c conda-forge diffusers
```

### Flax

```bash
pip install --upgrade diffusers[flax]
```

### Apple Silicon (M1/M2) Support

Refer to the [Apple Silicon Guide](https://huggingface.co/docs/diffusers/optimization/mps) for optimal performance on Apple Silicon devices.

## Quickstart: Generate Images with Text-to-Image

Generate stunning images from text prompts in seconds:

```python
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipeline.to("cuda")
pipeline("An image of a squirrel in Picasso style").images[0]
```

## Dive Deeper: Build Your Own Diffusion Systems

Explore the modular components to create your own diffusion systems:

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

## Documentation and Learning Resources

*   [Tutorial](https://huggingface.co/docs/diffusers/tutorials/tutorial_overview):  A comprehensive introduction to the library.
*   [Loading Components](https://huggingface.co/docs/diffusers/using-diffusers/loading):  Learn how to load and configure pipelines, models, and schedulers.
*   [Inference with Pipelines](https://huggingface.co/docs/diffusers/using-diffusers/overview_techniques):  Generate outputs, batch generations, control outputs and randomness.
*   [Optimization](https://huggingface.co/docs/diffusers/optimization/fp16): Boost performance with optimization strategies.
*   [Training](https://huggingface.co/docs/diffusers/training/overview): Train your own diffusion models.

## Contributing

We welcome contributions!  Check out our [Contribution Guide](https://github.com/huggingface/diffusers/blob/main/CONTRIBUTING.md) and explore [issues](https://github.com/huggingface/diffusers/issues) to get started.

*   [Good first issues](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22)
*   [New model/pipeline](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22New+pipeline%2Fmodel%22)
*   [New scheduler](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22New+scheduler%22)

Join the conversation on our public Discord channel: <a href="https://discord.gg/G7tWnz98XR"><img alt="Join us on Discord" src="https://img.shields.io/discord/823813159592001537?color=5865F2&logo=discord&logoColor=white"></a>.

## Popular Tasks and Pipelines

| Task                       | Pipeline                                                                                                                                                  | Hugging Face Hub                                                                                                                   |
| :-------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------- |
| Unconditional Image Generation | [DDPM](https://huggingface.co/docs/diffusers/api/pipelines/ddpm)                                                                                      | [google/ddpm-ema-church-256](https://huggingface.co/google/ddpm-ema-church-256)                                                 |
| Text-to-Image              | [Stable Diffusion Text-to-Image](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img)                                         | [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)               |
| Text-to-Image              | [unCLIP](https://huggingface.co/docs/diffusers/api/pipelines/unclip)                                                                                   | [kakaobrain/karlo-v1-alpha](https://huggingface.co/kakaobrain/karlo-v1-alpha)                                                 |
| Text-to-Image              | [DeepFloyd IF](https://huggingface.co/docs/diffusers/api/pipelines/deepfloyd_if)                                                                          | [DeepFloyd/IF-I-XL-v1.0](https://huggingface.co/DeepFloyd/IF-I-XL-v1.0)                                                       |
| Text-to-Image              | [Kandinsky](https://huggingface.co/docs/diffusers/api/pipelines/kandinsky)                                                                               | [kandinsky-community/kandinsky-2-2-decoder](https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder)               |
| Text-guided Image-to-Image | [ControlNet](https://huggingface.co/docs/diffusers/api/pipelines/controlnet)                                                                             | [lllyasviel/sd-controlnet-canny](https://huggingface.co/lllyasviel/sd-controlnet-canny)                                         |
| Text-guided Image-to-Image | [InstructPix2Pix](https://huggingface.co/docs/diffusers/api/pipelines/pix2pix)                                                                         | [timbrooks/instruct-pix2pix](https://huggingface.co/timbrooks/instruct-pix2pix)                                                 |
| Text-guided Image-to-Image | [Stable Diffusion Image-to-Image](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/img2img)                                         | [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)               |
| Text-guided Image Inpainting| [Stable Diffusion Inpainting](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/inpaint)                                                | [runwayml/stable-diffusion-inpainting](https://huggingface.co/runwayml/stable-diffusion-inpainting)                         |
| Image Variation            | [Stable Diffusion Image Variation](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/image_variation)                              | [lambdalabs/sd-image-variations-diffusers](https://huggingface.co/lambdalabs/sd-image-variations-diffusers)                    |
| Super Resolution           | [Stable Diffusion Upscale](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/upscale)                                                 | [stabilityai/stable-diffusion-x4-upscaler](https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler)                 |
| Super Resolution           | [Stable Diffusion Latent Upscale](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/latent_upscale)                                   | [stabilityai/sd-x2-latent-upscaler](https://huggingface.co/stabilityai/sd-x2-latent-upscaler)                                |

## Libraries Using 🤗 Diffusers

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
*   ...and many more!

## Credits

This library builds upon the work of many researchers. Special thanks to:

*   @CompVis for latent diffusion models ([here](https://github.com/CompVis/latent-diffusion))
*   @hojonathanho for the original DDPM implementation ([here](https://github.com/hojonathanho/diffusion)) and @pesser for the PyTorch translation ([here](https://github.com/pesser/pytorch_diffusion))
*   @ermongroup for the DDIM implementation ([here](https://github.com/ermongroup/ddim))
*   @yang-song for the Score-VE and Score-VP implementations ([here](https://github.com/yang-song/score_sde_pytorch))

Also, thanks to @heejkoo and @crowsonkb and @rromb for helpful discussions.

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

Key improvements and summaries:

*   **SEO-optimized Title and Introduction:**  Uses relevant keywords and immediately highlights the library's primary function.
*   **Clear, Concise Hook:**  The opening sentence summarizes the library's core purpose.
*   **Bulleted Key Features:**  Easy to scan and highlights the library's strengths.
*   **Structured Headings:**  Improved readability and organization.
*   **Concise Installation Instructions:**  Keeps the installation steps brief and clear.
*   **Focus on User Benefits:**  The "Quickstart" and example code demonstrate the ease of use.
*   **Comprehensive Documentation Guide:** Offers direct links to key documentation sections.
*   **Call to Action for Contributions:**  Encourages community involvement.
*   **Table of Popular Tasks:**  Clearly presents common use cases with direct links.
*   **Expanded List of Libraries:** Showcases more projects built with Diffusers.
*   **Credits Section:**  Maintains and restructures the original credits.
*   **Citation Included:**  Provides the proper citation.
*   **Added keywords**:  Added "Generate Images" and "Diffusion Models" to help search engine optimization.