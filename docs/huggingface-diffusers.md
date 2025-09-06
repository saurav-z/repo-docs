# 🤗 Diffusers: Generate Images, Audio, and More with State-of-the-Art Diffusion Models

**Unleash the power of AI to create stunning visuals and beyond with 🤗 Diffusers, the leading open-source library for diffusion models.** ([Back to Original Repo](https://github.com/huggingface/diffusers))

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

🤗 Diffusers provides a modular and user-friendly toolkit for working with diffusion models, enabling both inference and training.  Built with a focus on ease of use, customizability, and a collaborative approach,  🤗 Diffusers empowers you to create cutting-edge AI applications.

**Key Features:**

*   **Ready-to-Use Pipelines:**  Quickly generate images, audio, and more with pre-trained diffusion pipelines, requiring only a few lines of code.
*   **Flexible Schedulers:** Experiment with various noise schedulers to fine-tune the speed and quality of your diffusion model outputs.
*   **Modular Models:**  Utilize pre-trained models as building blocks, combined with schedulers, to create your own custom diffusion systems.
*   **Extensive Hub Integration:**  Access a vast collection of pre-trained models on the Hugging Face Hub.
*   **Optimization:**  Tools and guides for optimizing your models to run faster and use less memory.
*   **Training Support:**  Comprehensive guides for training your own diffusion models.

## Installation

Get started quickly by installing 🤗 Diffusers in a virtual environment using pip or conda:

### PyTorch

```bash
pip install --upgrade diffusers[torch]
```

### Conda

```sh
conda install -c conda-forge diffusers
```

### Apple Silicon (M1/M2) Support

See the [How to use Stable Diffusion in Apple Silicon](https://huggingface.co/docs/diffusers/optimization/mps) guide for setup instructions.

## Quickstart

Generate images from text with a pre-trained model:

```python
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipeline.to("cuda")
pipeline("An image of a squirrel in Picasso style").images[0]
```

Build your own custom diffusion system using models and schedulers:

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

For a step-by-step introduction, check out the [Quickstart](https://huggingface.co/docs/diffusers/quicktour).

## Documentation & Resources

Explore the comprehensive documentation to master 🤗 Diffusers:

| Documentation                                                    | What Can You Learn?                                                                                                                                                                                            |
| ---------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Tutorial](https://huggingface.co/docs/diffusers/tutorials/tutorial_overview)                                                             | Learn the basics of using the library's key features, including building and training your own diffusion models.                                                                                                        |
| [Loading](https://huggingface.co/docs/diffusers/using-diffusers/loading)                                                              | Master loading and configuring all the components of the library: pipelines, models, and schedulers, and learn to use different schedulers.                                                                            |
| [Pipelines for inference](https://huggingface.co/docs/diffusers/using-diffusers/overview_techniques)                                              | Explore how to use pipelines for various inference tasks, batch generation, controlling output, and contributing a pipeline.                                                                                            |
| [Optimization](https://huggingface.co/docs/diffusers/optimization/fp16)                                                         | Optimize your diffusion models for faster performance and reduced memory consumption.                                                                                                                                      |
| [Training](https://huggingface.co/docs/diffusers/training/overview) | Guides for training a diffusion model for different tasks using various training techniques.                                                                                                                 |

## Contributing

We welcome contributions from the open-source community!

*   Consult the [Contribution guide](https://github.com/huggingface/diffusers/blob/main/CONTRIBUTING.md).
*   Find issues to work on:
    *   [Good first issues](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22)
    *   [New model/pipeline](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22New+pipeline%2Fmodel%22)
    *   [New scheduler](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22New+scheduler%22)

Join the conversation on our public Discord channel: <a href="https://discord.gg/G7tWnz98XR"><img alt="Join us on Discord" src="https://img.shields.io/discord/823813159592001537?color=5865F2&logo=discord&logoColor=white"></a>.

## Popular Tasks & Pipelines

| Task                           | Pipeline                                                                                                                                     | 🤗 Hub                                                                                                 |
| :----------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------- |
| Unconditional Image Generation | [DDPM](https://huggingface.co/docs/diffusers/api/pipelines/ddpm)                                                                                | [google/ddpm-ema-church-256](https://huggingface.co/google/ddpm-ema-church-256)                      |
| Text-to-Image                  | [Stable Diffusion Text-to-Image](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img)                               | [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) |
| Text-to-Image                  | [unCLIP](https://huggingface.co/docs/diffusers/api/pipelines/unclip)                                                                          | [kakaobrain/karlo-v1-alpha](https://huggingface.co/kakaobrain/karlo-v1-alpha)                         |
| Text-to-Image                  | [DeepFloyd IF](https://huggingface.co/docs/diffusers/api/pipelines/deepfloyd_if)                                                                | [DeepFloyd/IF-I-XL-v1.0](https://huggingface.co/DeepFloyd/IF-I-XL-v1.0)                              |
| Text-to-Image                  | [Kandinsky](https://huggingface.co/docs/diffusers/api/pipelines/kandinsky)                                                                    | [kandinsky-community/kandinsky-2-2-decoder](https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder) |
| Text-guided Image-to-Image     | [ControlNet](https://huggingface.co/docs/diffusers/api/pipelines/controlnet)                                                                  | [lllyasviel/sd-controlnet-canny](https://huggingface.co/lllyasviel/sd-controlnet-canny)                    |
| Text-guided Image-to-Image     | [InstructPix2Pix](https://huggingface.co/docs/diffusers/api/pipelines/pix2pix)                                                              | [timbrooks/instruct-pix2pix](https://huggingface.co/timbrooks/instruct-pix2pix)                         |
| Text-guided Image-to-Image     | [Stable Diffusion Image-to-Image](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/img2img)                             | [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) |
| Text-guided Image Inpainting   | [Stable Diffusion Inpainting](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/inpaint)                                | [runwayml/stable-diffusion-inpainting](https://huggingface.co/runwayml/stable-diffusion-inpainting)       |
| Image Variation                | [Stable Diffusion Image Variation](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/image_variation)                       | [lambdalabs/sd-image-variations-diffusers](https://huggingface.co/lambdalabs/sd-image-variations-diffusers)   |
| Super Resolution               | [Stable Diffusion Upscale](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/upscale)                                     | [stabilityai/stable-diffusion-x4-upscaler](https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler)   |
| Super Resolution               | [Stable Diffusion Latent Upscale](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/latent_upscale)                         | [stabilityai/sd-x2-latent-upscaler](https://huggingface.co/stabilityai/sd-x2-latent-upscaler)              |

## Popular Libraries Using 🧨 Diffusers

-   [Microsoft/TaskMatrix](https://github.com/microsoft/TaskMatrix)
-   [InvokeAI/InvokeAI](https://github.com/invoke-ai/InvokeAI)
-   [InstantID/InstantID](https://github.com/InstantID/InstantID)
-   [apple/ml-stable-diffusion](https://github.com/apple/ml-stable-diffusion)
-   [Sanster/lama-cleaner](https://github.com/Sanster/lama-cleaner)
-   [IDEA-Research/Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything)
-   [ashawkey/stable-dreamfusion](https://github.com/ashawkey/stable-dreamfusion)
-   [deep-floyd/IF](https://github.com/deep-floyd/IF)
-   [bentoml/BentoML](https://github.com/bentoml/BentoML)
-   [bmaltais/kohya_ss](https://github.com/bmaltais/kohya_ss)
-   ... and +14,000 other amazing GitHub repositories 💪

Thank you for using 🤗 Diffusers!

## Credits

This library is built upon the work of many researchers and developers. We would like to thank the following:

*   @CompVis' latent diffusion models ([here](https://github.com/CompVis/latent-diffusion))
*   @hojonathanho's original DDPM implementation ([here](https://github.com/hojonathanho/diffusion)) and @pesser's PyTorch translation ([here](https://github.com/pesser/pytorch_diffusion))
*   @ermongroup's DDIM implementation ([here](https://github.com/ermongroup/ddim))
*   @yang-song's Score-VE and Score-VP implementations ([here](https://github.com/yang-song/score_sde_pytorch))

We also want to thank @heejkoo for the helpful overview of papers, code and resources on diffusion models ([here](https://github.com/heejkoo/Awesome-Diffusion-Models)) as well as @crowsonkb and @rromb for useful discussions and insights.

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