# 🤗 Diffusers: Your Gateway to Cutting-Edge Diffusion Models

**Generate stunning images, audio, and 3D structures with the leading diffusion model library!**

[Link to Original Repo: Diffusers on GitHub](https://github.com/huggingface/diffusers)

🤗 Diffusers is the premier open-source library for working with state-of-the-art pretrained diffusion models. Whether you're an enthusiast or a researcher, explore the power of diffusion models with our intuitive, customizable, and user-friendly toolkit.

**Key Features:**

*   **Versatile Pipelines:** Leverage pre-built diffusion pipelines for a wide range of tasks, including text-to-image generation, image-to-image manipulation, and more.
*   **Modular Design:** Build custom diffusion systems using interchangeable noise schedulers and pretrained models as building blocks.
*   **Extensive Model Support:** Access a vast collection of pretrained models for image generation, audio synthesis, and even 3D molecular structure creation.
*   **Easy to Use:**  Get started quickly with simple inference solutions and a focus on usability.
*   **Customization Friendly:**  Easily tweak and adapt models to your specific needs, fostering experimentation and research.
*   **Optimized for Performance:** Access guides on how to optimize your diffusion models to run faster and consume less memory.
*   **Community Driven:** Benefit from a vibrant community, with contributions and support encouraged.

## Installation

Install 🤗 Diffusers in a virtual environment using either PyPI or Conda.

**PyTorch:**

```bash
pip install --upgrade diffusers[torch]
```

**Flax:**

```bash
pip install --upgrade diffusers[flax]
```

## Quickstart

Create beautiful outputs in a few lines of code. Load a pretrained diffusion model directly from the Hugging Face Hub (browse over 30,000 models):

```python
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipeline.to("cuda")
pipeline("An image of a squirrel in Picasso style").images[0]
```

Or, build your own diffusion system:

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

## Documentation & Resources

Explore the documentation for in-depth information:

| Documentation                                                   | What you can learn                                                                                                                                                                                            |
| :------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Tutorial](https://huggingface.co/docs/diffusers/tutorials/tutorial_overview)                                                            | Learn the library's core features, including using models and schedulers to build your own diffusion system and training your own diffusion model.  |
| [Loading](https://huggingface.co/docs/diffusers/using-diffusers/loading)                                                             | Discover how to load and configure all library components (pipelines, models, and schedulers) and use different schedulers.                                        |
| [Pipelines for inference](https://huggingface.co/docs/diffusers/using-diffusers/overview_techniques)                                             | Learn how to use pipelines for inference, batch generation, output control, randomness management, and contributing a pipeline.               |
| [Optimization](https://huggingface.co/docs/diffusers/optimization/fp16)                                                        | Get guides on optimizing your diffusion model for speed and memory efficiency.                                                                                                          |
| [Training](https://huggingface.co/docs/diffusers/training/overview) | Learn how to train a diffusion model for different tasks with various training techniques.                                                                                               |

## Contribute

Join the growing community!  We welcome contributions.

*   Explore [Good first issues](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22).
*   Contribute exciting new diffusion models / diffusion pipelines from  [New model/pipeline](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22New+pipeline%2Fmodel%22).
*   Add new schedulers by reviewing [New scheduler](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22New+scheduler%22).

## Popular Tasks & Pipelines

| Task                       | Pipeline                                                                                                                                 | 🤗 Hub                                                                                               |
| :------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------- |
| Unconditional Image Generation | [DDPM](https://huggingface.co/docs/diffusers/api/pipelines/ddpm)                                                                       | [google/ddpm-ema-church-256](https://huggingface.co/google/ddpm-ema-church-256)                       |
| Text-to-Image              | [Stable Diffusion Text-to-Image](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img)                         | [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) |
| Text-to-Image              | [unCLIP](https://huggingface.co/docs/diffusers/api/pipelines/unclip)                                                                     | [kakaobrain/karlo-v1-alpha](https://huggingface.co/kakaobrain/karlo-v1-alpha)                         |
| Text-to-Image              | [DeepFloyd IF](https://huggingface.co/docs/diffusers/api/pipelines/deepfloyd_if)                                                        | [DeepFloyd/IF-I-XL-v1.0](https://huggingface.co/DeepFloyd/IF-I-XL-v1.0)                                |
| Text-to-Image              | [Kandinsky](https://huggingface.co/docs/diffusers/api/pipelines/kandinsky)                                                               | [kandinsky-community/kandinsky-2-2-decoder](https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder) |
| Text-guided Image-to-Image | [ControlNet](https://huggingface.co/docs/diffusers/api/pipelines/controlnet)                                                            | [lllyasviel/sd-controlnet-canny](https://huggingface.co/lllyasviel/sd-controlnet-canny)               |
| Text-guided Image-to-Image | [InstructPix2Pix](https://huggingface.co/docs/diffusers/api/pipelines/pix2pix)                                                            | [timbrooks/instruct-pix2pix](https://huggingface.co/timbrooks/instruct-pix2pix)                     |
| Text-guided Image-to-Image | [Stable Diffusion Image-to-Image](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/img2img)                         | [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) |
| Text-guided Image Inpainting | [Stable Diffusion Inpainting](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/inpaint)                           | [runwayml/stable-diffusion-inpainting](https://huggingface.co/runwayml/stable-diffusion-inpainting)     |
| Image Variation            | [Stable Diffusion Image Variation](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/image_variation)               | [lambdalabs/sd-image-variations-diffusers](https://huggingface.co/lambdalabs/sd-image-variations-diffusers) |
| Super Resolution           | [Stable Diffusion Upscale](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/upscale)                               | [stabilityai/stable-diffusion-x4-upscaler](https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler) |
| Super Resolution           | [Stable Diffusion Latent Upscale](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/latent_upscale)                     | [stabilityai/sd-x2-latent-upscaler](https://huggingface.co/stabilityai/sd-x2-latent-upscaler)           |

##  Join Our Community

Connect with fellow enthusiasts on our Discord server: [![Join us on Discord](https://img.shields.io/discord/823813159592001537?color=5865F2&logo=discord&logoColor=white)](https://discord.gg/G7tWnz98XR) for discussions on diffusion models, contributions, and more!

## Libraries Powered by 🤗 Diffusers

Diffusers is the backbone for amazing projects:

*   [TaskMatrix](https://github.com/microsoft/TaskMatrix)
*   [InvokeAI](https://github.com/invoke-ai/InvokeAI)
*   [InstantID](https://github.com/InstantID/InstantID)
*   [ml-stable-diffusion](https://github.com/apple/ml-stable-diffusion)
*   and 14,000+ more!

## Credits

Special thanks to the researchers and developers whose work made this library possible:

*   @CompVis, @hojonathanho, @pesser, @ermongroup, @yang-song, @heejkoo, @crowsonkb, and @rromb.

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