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

**🤗 Diffusers is your comprehensive toolkit for cutting-edge diffusion models, simplifying generation of images, audio, and more!** (See original repo: [https://github.com/huggingface/diffusers](https://github.com/huggingface/diffusers))

Built for usability, simplicity, and customizability, 🤗 Diffusers empowers you to explore and create with state-of-the-art diffusion models. Whether you're looking for ready-to-use solutions or building your own models, this library provides the building blocks you need.

## Key Features

*   **Intuitive Pipelines:**  Generate images, audio, and 3D structures with just a few lines of code using pre-built pipelines.
*   **Flexible Schedulers:** Experiment with a variety of noise schedulers to control diffusion speed and output quality.
*   **Modular Models:** Utilize pre-trained models as building blocks for creating custom diffusion systems.
*   **Extensive Model Hub:** Access a vast library of pre-trained models for various tasks.
*   **Optimization Guides:**  Optimize your models for faster inference and reduced memory consumption.
*   **Training Support:** Comprehensive guides for training diffusion models for a variety of tasks.

## Installation

Install 🤗 Diffusers in a virtual environment from PyPI or Conda.

### PyTorch

```bash
pip install --upgrade diffusers[torch]
```

### Flax

```bash
pip install --upgrade diffusers[flax]
```

### Apple Silicon (M1/M2) support

Refer to the [How to use Stable Diffusion in Apple Silicon](https://huggingface.co/docs/diffusers/optimization/mps) guide.

## Quickstart

Generate stunning outputs effortlessly with 🤗 Diffusers. Load a pretrained diffusion model and create images from text:

```python
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipeline.to("cuda")
pipeline("An image of a squirrel in Picasso style").images[0]
```

## Documentation and Tutorials

Discover the full potential of 🤗 Diffusers with our comprehensive documentation:

*   [Tutorial](https://huggingface.co/docs/diffusers/tutorials/tutorial_overview): Learn the library's core features, including building and training diffusion models.
*   [Loading](https://huggingface.co/docs/diffusers/using-diffusers/loading): Learn how to load and configure the components like pipelines, models, and schedulers.
*   [Pipelines for inference](https://huggingface.co/docs/diffusers/using-diffusers/overview_techniques): Master pipelines for diverse inference tasks, including batch generation and output control.
*   [Optimization](https://huggingface.co/docs/diffusers/optimization/fp16):  Optimize your models for performance.
*   [Training](https://huggingface.co/docs/diffusers/training/overview): Train your own diffusion models.

## Contribution

We welcome contributions from the open-source community!  Consult the [Contribution guide](https://github.com/huggingface/diffusers/blob/main/CONTRIBUTING.md) and explore [issues](https://github.com/huggingface/diffusers/issues) to get involved.

*   [Good first issues](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22)
*   [New model/pipeline](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22New+pipeline%2Fmodel%22)
*   [New scheduler](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22New+scheduler%22)

Join the community on Discord: <a href="https://discord.gg/G7tWnz98XR"><img alt="Join us on Discord" src="https://img.shields.io/discord/823813159592001537?color=5865F2&logo=discord&logoColor=white"></a>.

## Popular Tasks & Pipelines

| **Task**                    | **Pipeline**                                                                 | **🤗 Hub**                                                                     |
| :--------------------------- | :--------------------------------------------------------------------------- | :----------------------------------------------------------------------------- |
| Unconditional Image Generation | [DDPM](https://huggingface.co/docs/diffusers/api/pipelines/ddpm)             | [google/ddpm-ema-church-256](https://huggingface.co/google/ddpm-ema-church-256) |
| Text-to-Image                | [Stable Diffusion Text-to-Image](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img) | [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)       |
| Text-to-Image                | [unCLIP](https://huggingface.co/docs/diffusers/api/pipelines/unclip)              | [kakaobrain/karlo-v1-alpha](https://huggingface.co/kakaobrain/karlo-v1-alpha)            |
| Text-to-Image                | [DeepFloyd IF](https://huggingface.co/docs/diffusers/api/pipelines/deepfloyd_if)          | [DeepFloyd/IF-I-XL-v1.0](https://huggingface.co/DeepFloyd/IF-I-XL-v1.0)           |
| Text-to-Image                | [Kandinsky](https://huggingface.co/docs/diffusers/api/pipelines/kandinsky)          | [kandinsky-community/kandinsky-2-2-decoder](https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder)          |
| Text-guided Image-to-Image   | [ControlNet](https://huggingface.co/docs/diffusers/api/pipelines/controlnet)   | [lllyasviel/sd-controlnet-canny](https://huggingface.co/lllyasviel/sd-controlnet-canny)     |
| Text-guided Image-to-Image   | [InstructPix2Pix](https://huggingface.co/docs/diffusers/api/pipelines/pix2pix) | [timbrooks/instruct-pix2pix](https://huggingface.co/timbrooks/instruct-pix2pix)    |
| Text-guided Image-to-Image   | [Stable Diffusion Image-to-Image](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/img2img) | [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)       |
| Text-guided Image Inpainting  | [Stable Diffusion Inpainting](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/inpaint) | [runwayml/stable-diffusion-inpainting](https://huggingface.co/runwayml/stable-diffusion-inpainting) |
| Image Variation               | [Stable Diffusion Image Variation](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/image_variation) | [lambdalabs/sd-image-variations-diffusers](https://huggingface.co/lambdalabs/sd-image-variations-diffusers) |
| Super Resolution              | [Stable Diffusion Upscale](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/upscale)   | [stabilityai/stable-diffusion-x4-upscaler](https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler)    |
| Super Resolution              | [Stable Diffusion Latent Upscale](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/latent_upscale)  | [stabilityai/sd-x2-latent-upscaler](https://huggingface.co/stabilityai/sd-x2-latent-upscaler)   |

## Popular libraries using 🧨 Diffusers

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
*   +14,000 other amazing GitHub repositories 💪

Thank you for using us ❤️.

## Credits

This library builds upon the work of many researchers and developers. We extend our gratitude to the following implementations:

*   [@CompVis](https://github.com/CompVis)'s latent diffusion models library
*   [@hojonathanho](https://github.com/hojonathanho/diffusion) for the original DDPM implementation
*   [@pesser](https://github.com/pesser/pytorch_diffusion) for the PyTorch translation of DDPM
*   [@ermongroup](https://github.com/ermongroup/ddim) for the DDIM implementation
*   [@yang-song](https://github.com/yang-song/score_sde_pytorch) for the Score-VE and Score-VP implementations
*   @heejkoo for the overview of diffusion models
*   @crowsonkb and @rromb for discussions and insights.

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