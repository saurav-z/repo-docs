# 🤗 Diffusers: Generate Images, Audio, and More with State-of-the-Art Diffusion Models

**Harness the power of diffusion models to create stunning visuals and more with the 🤗 Diffusers library!**

[![GitHub License](https://img.shields.io/github/license/huggingface/datasets.svg?color=blue)](https://github.com/huggingface/diffusers/blob/main/LICENSE)
[![GitHub Release](https://img.shields.io/github/release/huggingface/diffusers.svg)](https://github.com/huggingface/diffusers/releases)
[![PyPI Monthly Downloads](https://static.pepy.tech/badge/diffusers/month)](https://pepy.tech/project/diffusers)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)
[![Twitter Follow](https://img.shields.io/twitter/url/https/twitter.com/diffuserslib.svg?style=social&label=Follow%20%40diffuserslib)](https://twitter.com/diffuserslib)

🤗 Diffusers is the leading library for utilizing and building upon pretrained diffusion models, enabling you to generate images, audio, and even 3D structures of molecules. This modular toolbox caters to both simple inference tasks and complex model training scenarios.  With a focus on [usability](https://huggingface.co/docs/diffusers/conceptual/philosophy#usability-over-performance), [simplicity](https://huggingface.co/docs/diffusers/conceptual/philosophy#simple-over-easy), and [customization](https://huggingface.co/docs/diffusers/conceptual/philosophy#tweakable-contributorfriendly-over-abstraction), 🤗 Diffusers empowers you to explore the exciting world of generative AI.

**Key Features:**

*   **Diffusion Pipelines:** Utilize ready-to-use pipelines for rapid inference and generation.
*   **Flexible Schedulers:** Experiment with diverse noise schedulers to fine-tune diffusion speeds and output quality.
*   **Modular Models:** Build custom diffusion systems by combining pretrained models and schedulers.

## Installation

Install 🤗 Diffusers in a virtual environment using either `pip` or `conda`.  Ensure you have [PyTorch](https://pytorch.org/get-started/locally/) and optionally [Flax](https://flax.readthedocs.io/en/latest/#installation) installed.

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

See the [How to use Stable Diffusion in Apple Silicon](https://huggingface.co/docs/diffusers/optimization/mps) guide for optimal performance on Apple Silicon devices.

## Quickstart

Generate images in minutes with a few lines of code. Load a pretrained diffusion model (explore the [Hub](https://huggingface.co/models?library=diffusers&sort=downloads) for 30,000+ checkpoints):

```python
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipeline.to("cuda")
pipeline("An image of a squirrel in Picasso style").images[0]
```

Or, construct your own diffusion system using the toolbox:

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

For further guidance, consult the [Quickstart](https://huggingface.co/docs/diffusers/quicktour) and get started today!

## Documentation Overview

| **Documentation**                                                   | **Description**                                                                                                                                                                           |
|---------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Tutorial](https://huggingface.co/docs/diffusers/tutorials/tutorial_overview)                                                            | Learn core concepts like building your own diffusion systems, and training your own diffusion model.  |
| [Loading](https://huggingface.co/docs/diffusers/using-diffusers/loading)                                                             | Learn to load and configure the library's components (pipelines, models, and schedulers), and use different schedulers.                                         |
| [Pipelines for inference](https://huggingface.co/docs/diffusers/using-diffusers/overview_techniques)                                             | Learn how to use pipelines, batch generation, control outputs, and contribute your own pipelines.               |
| [Optimization](https://huggingface.co/docs/diffusers/optimization/fp16)                                                        | Optimize your diffusion model to run faster and use less memory.                                                                                                          |
| [Training](https://huggingface.co/docs/diffusers/training/overview) | Train a diffusion model for various tasks with diverse training techniques.                                                                                               |

## Contribution

We welcome contributions from the open-source community!  Consult our [Contribution guide](https://github.com/huggingface/diffusers/blob/main/CONTRIBUTING.md) for details.  Find opportunities to contribute by reviewing [issues](https://github.com/huggingface/diffusers/issues):

*   [Good first issues](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22)
*   [New model/pipeline](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22New+pipeline%2Fmodel%22)
*   [New scheduler](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22New+scheduler%22)

Join the discussion on our public Discord channel: <a href="https://discord.gg/G7tWnz98XR"><img alt="Join us on Discord" src="https://img.shields.io/discord/823813159592001537?color=5865F2&logo=discord&logoColor=white"></a>

## Popular Tasks & Pipelines

| Task                       | Pipeline                                                                                  | 🤗 Hub                                                                                                     |
| -------------------------- | ----------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| Unconditional Image Generation | [DDPM](https://huggingface.co/docs/diffusers/api/pipelines/ddpm)                                   | [google/ddpm-ema-church-256](https://huggingface.co/google/ddpm-ema-church-256)                            |
| Text-to-Image              | [Stable Diffusion Text-to-Image](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img) | [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) |
| Text-to-Image              | [unCLIP](https://huggingface.co/docs/diffusers/api/pipelines/unclip)                                   | [kakaobrain/karlo-v1-alpha](https://huggingface.co/kakaobrain/karlo-v1-alpha)                            |
| Text-to-Image              | [DeepFloyd IF](https://huggingface.co/docs/diffusers/api/pipelines/deepfloyd_if)                       | [DeepFloyd/IF-I-XL-v1.0](https://huggingface.co/DeepFloyd/IF-I-XL-v1.0)                                  |
| Text-to-Image              | [Kandinsky](https://huggingface.co/docs/diffusers/api/pipelines/kandinsky)                             | [kandinsky-community/kandinsky-2-2-decoder](https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder) |
| Text-guided Image-to-Image | [ControlNet](https://huggingface.co/docs/diffusers/api/pipelines/controlnet)                         | [lllyasviel/sd-controlnet-canny](https://huggingface.co/lllyasviel/sd-controlnet-canny)                   |
| Text-guided Image-to-Image | [InstructPix2Pix](https://huggingface.co/docs/diffusers/api/pipelines/pix2pix)                      | [timbrooks/instruct-pix2pix](https://huggingface.co/timbrooks/instruct-pix2pix)                           |
| Text-guided Image-to-Image | [Stable Diffusion Image-to-Image](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/img2img) | [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) |
| Text-guided Image Inpainting | [Stable Diffusion Inpainting](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/inpaint)     | [runwayml/stable-diffusion-inpainting](https://huggingface.co/runwayml/stable-diffusion-inpainting)     |
| Image Variation            | [Stable Diffusion Image Variation](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/image_variation)    | [lambdalabs/sd-image-variations-diffusers](https://huggingface.co/lambdalabs/sd-image-variations-diffusers) |
| Super Resolution           | [Stable Diffusion Upscale](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/upscale)         | [stabilityai/stable-diffusion-x4-upscaler](https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler) |
| Super Resolution           | [Stable Diffusion Latent Upscale](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/latent_upscale)       | [stabilityai/sd-x2-latent-upscaler](https://huggingface.co/stabilityai/sd-x2-latent-upscaler)         |

## Libraries using 🤗 Diffusers

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
*   and +14,000 other repositories!

Thank you for using 🤗 Diffusers!

## Credits

This library builds upon the work of many researchers and developers. We'd like to thank the following implementations for their contributions:

*   [@CompVis](https://github.com/CompVis)' latent diffusion models library ([link](https://github.com/CompVis/latent-diffusion))
*   [@hojonathanho](https://github.com/hojonathanho) original DDPM implementation ([link](https://github.com/hojonathanho/diffusion)) and the PyTorch translation by @pesser ([link](https://github.com/pesser/pytorch_diffusion))
*   [@ermongroup](https://github.com/ermongroup)'s DDIM implementation ([link](https://github.com/ermongroup/ddim))
*   [@yang-song](https://github.com/yang-song)'s Score-VE and Score-VP implementations ([link](https://github.com/yang-song/score_sde_pytorch))

We also appreciate @heejkoo for their diffusion models resources ([link](https://github.com/heejkoo/Awesome-Diffusion-Models)) and @crowsonkb and @rromb for their insights.

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

[Back to Top](#)
```

Key improvements and optimizations:

*   **SEO-Friendly Title & Introduction:** The title and first sentence now target relevant keywords ("diffusion models," "generate images," "state-of-the-art").
*   **Clear Headings & Structure:**  Uses headings to organize information logically, improving readability and SEO.
*   **Bulleted Key Features:** Highlights the core functionality concisely.
*   **Targeted Keywords:** Includes relevant terms throughout the document, such as "image generation," "audio generation," "pretrained models," "pipelines," "schedulers," and names of specific models.
*   **Internal Linking:** Uses links within the document (e.g., to documentation sections) to improve user experience and SEO.  Added a "Back to Top" link for ease of navigation.
*   **Concise Summarization:** Reduces the length while retaining all critical information.
*   **Calls to Action:** Encourages users to explore the Quickstart and contribute.
*   **Expanded Documentation Overview:**  More descriptive titles and descriptions
*   **Popular Library Showcase:**  Includes a broader listing of libraries to boost SEO