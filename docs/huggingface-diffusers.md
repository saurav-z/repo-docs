<p align="center">
    <br>
    <img src="https://raw.githubusercontent.com/huggingface/diffusers/main/docs/source/en/imgs/diffusers_library.jpg" width="400"/>
    <br>
<p>

# 🤗 Diffusers: Generate Amazing Images, Audio, and More with Diffusion Models

**Unleash the power of diffusion models with 🤗 Diffusers, the leading library for state-of-the-art generative AI.**

[See the original repo](https://github.com/huggingface/diffusers)

<p align="center">
    <a href="https://github.com/huggingface/diffusers/blob/main/LICENSE"><img alt="GitHub" src="https://img.shields.io/github/license/huggingface/datasets.svg?color=blue"></a>
    <a href="https://github.com/huggingface/diffusers/releases"><img alt="GitHub release" src="https://img.shields.io/github/release/huggingface/diffusers.svg"></a>
    <a href="https://pepy.tech/project/diffusers"><img alt="GitHub release" src="https://static.pepy.tech/badge/diffusers/month"></a>
    <a href="CODE_OF_CONDUCT.md"><img alt="Contributor Covenant" src="https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg"></a>
    <a href="https://twitter.com/diffuserslib"><img alt="X account" src="https://img.shields.io/twitter/url/https/twitter.com/diffuserslib.svg?style=social&label=Follow%20%40diffuserslib"></a>
</p>

🤗 Diffusers is your one-stop shop for working with cutting-edge diffusion models.  Whether you're a beginner or an expert, the library provides a modular and user-friendly approach to generating images, audio, 3D structures, and more.  Built with a focus on usability, simplicity, and customizability, it empowers you to quickly create and experiment with diffusion models.

**Key Features:**

*   **Easy Inference:**  Run pre-trained diffusion models with just a few lines of code.
*   **Flexible Components:**  Utilize interchangeable noise schedulers for diverse diffusion speeds and output quality.
*   **Modular Building Blocks:**  Combine pre-trained models with schedulers to construct your own custom diffusion systems.
*   **Large Model Hub:**  Access a vast collection of pre-trained models on the Hugging Face Hub.
*   **Comprehensive Documentation:**  Detailed guides, tutorials, and examples to get you started.

## Installation

Get started with 🤗 Diffusers by installing it in a virtual environment using pip or conda.

### PyTorch

**With pip:**

```bash
pip install --upgrade diffusers[torch]
```

**With conda:**

```sh
conda install -c conda-forge diffusers
```

### Flax

**With pip:**

```bash
pip install --upgrade diffusers[flax]
```

### Apple Silicon (M1/M2) Support

Refer to the [How to use Stable Diffusion in Apple Silicon](https://huggingface.co/docs/diffusers/optimization/mps) guide for optimized performance on Apple Silicon devices.

## Quickstart: Generate an Image from Text

Effortlessly generate images using the `from_pretrained` method to load a pre-trained diffusion model.

```python
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipeline.to("cuda")
pipeline("An image of a squirrel in Picasso style").images[0]
```

## Quickstart: Build Your Own Diffusion System

Explore the building blocks (models and schedulers) to create custom diffusion systems.

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

Check out the [Quickstart](https://huggingface.co/docs/diffusers/quicktour) for a deeper dive.

## Documentation and Guides

| **Documentation**                                                   | **What can I learn?**                                                                                                                                                                           |
|---------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Tutorial](https://huggingface.co/docs/diffusers/tutorials/tutorial_overview)                                                            | Learn the basics of using models and schedulers to create your own diffusion system, and training your own diffusion model.  |
| [Loading](https://huggingface.co/docs/diffusers/using-diffusers/loading)                                                             | Guides for how to load and configure all the components (pipelines, models, and schedulers) of the library, as well as how to use different schedulers.                                         |
| [Pipelines for inference](https://huggingface.co/docs/diffusers/using-diffusers/overview_techniques)                                             | Guides for using pipelines for different inference tasks, batched generation, controlling generated outputs and randomness, and contributing a pipeline to the library.               |
| [Optimization](https://huggingface.co/docs/diffusers/optimization/fp16)                                                        | Guides for optimizing your diffusion model to run faster and consume less memory.                                                                                                          |
| [Training](https://huggingface.co/docs/diffusers/training/overview) | Guides for training a diffusion model for different tasks with different training techniques.                                                                                               |

## Contribute

We warmly welcome contributions from the open-source community!

*   Check out our [Contribution guide](https://github.com/huggingface/diffusers/blob/main/CONTRIBUTING.md).
*   Find [issues](https://github.com/huggingface/diffusers/issues) to work on.
    *   See [Good first issues](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22).
    *   See [New model/pipeline](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22New+pipeline%2Fmodel%22).
    *   See [New scheduler](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22New+scheduler%22).

Join our public Discord channel to discuss diffusion models, get help, and collaborate:  <a href="https://discord.gg/G7tWnz98XR"><img alt="Join us on Discord" src="https://img.shields.io/discord/823813159592001537?color=5865F2&logo=discord&logoColor=white"></a>.

## Popular Tasks & Pipelines

| Task                     | Pipeline                                                                                          | 🤗 Hub                                                                                                 |
| ------------------------ | ------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| Unconditional Image Generation | <a href="https://huggingface.co/docs/diffusers/api/pipelines/ddpm"> DDPM </a>                                                                            | <a href="https://huggingface.co/google/ddpm-ema-church-256"> google/ddpm-ema-church-256 </a>                                           |
| Text-to-Image            | <a href="https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img">Stable Diffusion Text-to-Image</a>                                  | <a href="https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5"> stable-diffusion-v1-5/stable-diffusion-v1-5 </a>   |
| Text-to-Image            | <a href="https://huggingface.co/docs/diffusers/api/pipelines/unclip">unCLIP</a>                                                                            | <a href="https://huggingface.co/kakaobrain/karlo-v1-alpha"> kakaobrain/karlo-v1-alpha </a>                                           |
| Text-to-Image            | <a href="https://huggingface.co/docs/diffusers/api/pipelines/deepfloyd_if">DeepFloyd IF</a>                                                                      | <a href="https://huggingface.co/DeepFloyd/IF-I-XL-v1.0"> DeepFloyd/IF-I-XL-v1.0 </a>                                        |
| Text-to-Image            | <a href="https://huggingface.co/docs/diffusers/api/pipelines/kandinsky">Kandinsky</a>                                                                       | <a href="https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder"> kandinsky-community/kandinsky-2-2-decoder </a>                                      |
| Text-guided Image-to-Image | <a href="https://huggingface.co/docs/diffusers/api/pipelines/controlnet">ControlNet</a>                                                                      | <a href="https://huggingface.co/lllyasviel/sd-controlnet-canny"> lllyasviel/sd-controlnet-canny </a>                                      |
| Text-guided Image-to-Image | <a href="https://huggingface.co/docs/diffusers/api/pipelines/pix2pix">InstructPix2Pix</a>                                                                    | <a href="https://huggingface.co/timbrooks/instruct-pix2pix"> timbrooks/instruct-pix2pix </a>                                    |
| Text-guided Image-to-Image | <a href="https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/img2img">Stable Diffusion Image-to-Image</a>                              | <a href="https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5"> stable-diffusion-v1-5/stable-diffusion-v1-5 </a>   |
| Text-guided Image Inpainting | <a href="https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/inpaint">Stable Diffusion Inpainting</a>                                    | <a href="https://huggingface.co/runwayml/stable-diffusion-inpainting"> runwayml/stable-diffusion-inpainting </a>                                  |
| Image Variation          | <a href="https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/image_variation">Stable Diffusion Image Variation</a>                         | <a href="https://huggingface.co/lambdalabs/sd-image-variations-diffusers"> lambdalabs/sd-image-variations-diffusers </a>                                 |
| Super Resolution         | <a href="https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/upscale">Stable Diffusion Upscale</a>                                        | <a href="https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler"> stabilityai/stable-diffusion-x4-upscaler </a>                               |
| Super Resolution         | <a href="https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/latent_upscale">Stable Diffusion Latent Upscale</a>                         | <a href="https://huggingface.co/stabilityai/sd-x2-latent-upscaler"> stabilityai/sd-x2-latent-upscaler </a>                               |

## Used by Top Libraries

- [TaskMatrix](https://github.com/microsoft/TaskMatrix)
- [InvokeAI](https://github.com/invoke-ai/InvokeAI)
- [InstantID](https://github.com/InstantID/InstantID)
- [ml-stable-diffusion](https://github.com/apple/ml-stable-diffusion)
- [lama-cleaner](https://github.com/Sanster/lama-cleaner)
- [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything)
- [stable-dreamfusion](https://github.com/ashawkey/stable-dreamfusion)
- [IF](https://github.com/deep-floyd/IF)
- [BentoML](https://github.com/bentoml/BentoML)
- [kohya_ss](https://github.com/bmaltais/kohya_ss)
- ... and 14,000+ other amazing GitHub repositories!

## Credits

This library builds upon the work of many researchers and developers; we thank:

*   @CompVis for their latent diffusion models library ([here](https://github.com/CompVis/latent-diffusion)).
*   @hojonathanho for the original DDPM implementation ([here](https://github.com/hojonathanho/diffusion)) and @pesser for the PyTorch translation ([here](https://github.com/pesser/pytorch_diffusion)).
*   @ermongroup for the DDIM implementation ([here](https://github.com/ermongroup/ddim)).
*   @yang-song for the Score-VE and Score-VP implementations ([here](https://github.com/yang-song/score_sde_pytorch)).

We also thank @heejkoo for the helpful overview of diffusion models ([here](https://github.com/heejkoo/Awesome-Diffusion-Models)) and @crowsonkb and @rromb for their insights.

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