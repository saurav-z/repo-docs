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

# 🤗 Diffusers: Generate Images, Audio, and More with Diffusion Models

**🤗 Diffusers is your go-to Python library for creating state-of-the-art content with diffusion models, enabling stunning outputs with just a few lines of code.** ([See the original repo](https://github.com/huggingface/diffusers) for more details).

**Key Features:**

*   **User-Friendly Pipelines:** Run pre-trained diffusion models for image, audio, and 3D structure generation with ease.
*   **Flexible Schedulers:** Experiment with various noise schedulers to control diffusion speed and output quality.
*   **Modular Models:** Build your own diffusion systems by combining pretrained models and schedulers.
*   **Extensive Model Hub Integration:** Access a vast library of pre-trained models on the Hugging Face Hub.
*   **Optimized for Customization:**  Prioritizes usability, simplicity, and customizability.

## Installation

Install 🤗 Diffusers in a virtual environment using either `pip` or `conda`.  Ensure you have PyTorch and/or Flax installed, following their official documentation for setup.

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

Refer to the [How to use Stable Diffusion in Apple Silicon](https://huggingface.co/docs/diffusers/optimization/mps) guide for optimized usage.

## Quickstart

Generate outputs effortlessly with 🤗 Diffusers.  Load a pre-trained model and generate an image from text:

```python
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipeline.to("cuda")
pipeline("An image of a squirrel in Picasso style").images[0]
```

Or, build your own diffusion system with the model and scheduler toolboxes:

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

Explore the [Quickstart](https://huggingface.co/docs/diffusers/quicktour) to begin your diffusion journey.

## Documentation Navigation

| Documentation                                                   | What can I learn?                                                                                                                                                                                                                                               |
|---------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Tutorial](https://huggingface.co/docs/diffusers/tutorials/tutorial_overview)                                                            | Learn to use the library's core features, like building your own diffusion systems using models and schedulers, and training your own diffusion model.  |
| [Loading](https://huggingface.co/docs/diffusers/using-diffusers/loading)                                                             | Guides on loading and configuring library components (pipelines, models, and schedulers), and how to utilize different schedulers.                                         |
| [Pipelines for inference](https://huggingface.co/docs/diffusers/using-diffusers/overview_techniques)                                             | Learn how to utilize pipelines for different inference tasks, including batch generation, output control, randomness, and pipeline contributions.               |
| [Optimization](https://huggingface.co/docs/diffusers/optimization/fp16)                                                        | Guides on optimizing your diffusion model to improve speed and reduce memory consumption.                                                                                                         |
| [Training](https://huggingface.co/docs/diffusers/training/overview) | Guides on training a diffusion model for various tasks with different techniques.                                                                                                                                                             |

## Contribution

The library welcomes contributions from the open-source community!  See the [Contribution guide](https://github.com/huggingface/diffusers/blob/main/CONTRIBUTING.md) for details.  Check the [issues](https://github.com/huggingface/diffusers/issues) for opportunities.

*   [Good first issues](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22) - for general contribution opportunities.
*   [New model/pipeline](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22New+pipeline%2Fmodel%22) - for contributing new diffusion models and pipelines.
*   [New scheduler](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22New+scheduler%22)

Join the discussion in our public Discord channel: <a href="https://discord.gg/G7tWnz98XR"><img alt="Join us on Discord" src="https://img.shields.io/discord/823813159592001537?color=5865F2&logo=discord&logoColor=white"></a>.

## Popular Tasks & Pipelines

<table>
  <tr>
    <th>Task</th>
    <th>Pipeline</th>
    <th>🤗 Hub</th>
  </tr>
  <tr style="border-top: 2px solid black">
    <td>Unconditional Image Generation</td>
    <td><a href="https://huggingface.co/docs/diffusers/api/pipelines/ddpm"> DDPM </a></td>
    <td><a href="https://huggingface.co/google/ddpm-ema-church-256"> google/ddpm-ema-church-256 </a></td>
  </tr>
  <tr style="border-top: 2px solid black">
    <td>Text-to-Image</td>
    <td><a href="https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img">Stable Diffusion Text-to-Image</a></td>
      <td><a href="https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5"> stable-diffusion-v1-5/stable-diffusion-v1-5 </a></td>
  </tr>
  <tr>
    <td>Text-to-Image</td>
    <td><a href="https://huggingface.co/docs/diffusers/api/pipelines/unclip">unCLIP</a></td>
      <td><a href="https://huggingface.co/kakaobrain/karlo-v1-alpha"> kakaobrain/karlo-v1-alpha </a></td>
  </tr>
  <tr>
    <td>Text-to-Image</td>
    <td><a href="https://huggingface.co/docs/diffusers/api/pipelines/deepfloyd_if">DeepFloyd IF</a></td>
      <td><a href="https://huggingface.co/DeepFloyd/IF-I-XL-v1.0"> DeepFloyd/IF-I-XL-v1.0 </a></td>
  </tr>
  <tr>
    <td>Text-to-Image</td>
    <td><a href="https://huggingface.co/docs/diffusers/api/pipelines/kandinsky">Kandinsky</a></td>
      <td><a href="https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder"> kandinsky-community/kandinsky-2-2-decoder </a></td>
  </tr>
  <tr style="border-top: 2px solid black">
    <td>Text-guided Image-to-Image</td>
    <td><a href="https://huggingface.co/docs/diffusers/api/pipelines/controlnet">ControlNet</a></td>
      <td><a href="https://huggingface.co/lllyasviel/sd-controlnet-canny"> lllyasviel/sd-controlnet-canny </a></td>
  </tr>
  <tr>
    <td>Text-guided Image-to-Image</td>
    <td><a href="https://huggingface.co/docs/diffusers/api/pipelines/pix2pix">InstructPix2Pix</a></td>
      <td><a href="https://huggingface.co/timbrooks/instruct-pix2pix"> timbrooks/instruct-pix2pix </a></td>
  </tr>
  <tr>
    <td>Text-guided Image-to-Image</td>
    <td><a href="https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/img2img">Stable Diffusion Image-to-Image</a></td>
      <td><a href="https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5"> stable-diffusion-v1-5/stable-diffusion-v1-5 </a></td>
  </tr>
  <tr style="border-top: 2px solid black">
    <td>Text-guided Image Inpainting</td>
    <td><a href="https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/inpaint">Stable Diffusion Inpainting</a></td>
      <td><a href="https://huggingface.co/runwayml/stable-diffusion-inpainting"> runwayml/stable-diffusion-inpainting </a></td>
  </tr>
  <tr style="border-top: 2px solid black">
    <td>Image Variation</td>
    <td><a href="https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/image_variation">Stable Diffusion Image Variation</a></td>
      <td><a href="https://huggingface.co/lambdalabs/sd-image-variations-diffusers"> lambdalabs/sd-image-variations-diffusers </a></td>
  </tr>
  <tr style="border-top: 2px solid black">
    <td>Super Resolution</td>
    <td><a href="https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/upscale">Stable Diffusion Upscale</a></td>
      <td><a href="https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler"> stabilityai/stable-diffusion-x4-upscaler </a></td>
  </tr>
  <tr>
    <td>Super Resolution</td>
    <td><a href="https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/latent_upscale">Stable Diffusion Latent Upscale</a></td>
      <td><a href="https://huggingface.co/stabilityai/sd-x2-latent-upscaler"> stabilityai/sd-x2-latent-upscaler </a></td>
  </tr>
</table>

## Popular Libraries Using 🧨 Diffusers

*   https://github.com/microsoft/TaskMatrix
*   https://github.com/invoke-ai/InvokeAI
*   https://github.com/InstantID/InstantID
*   https://github.com/apple/ml-stable-diffusion
*   https://github.com/Sanster/lama-cleaner
*   https://github.com/IDEA-Research/Grounded-Segment-Anything
*   https://github.com/ashawkey/stable-dreamfusion
*   https://github.com/deep-floyd/IF
*   https://github.com/bentoml/BentoML
*   https://github.com/bmaltais/kohya_ss
*   +14,000 other amazing GitHub repositories 💪

Thank you for using us ❤️.

## Credits

(Same as original)

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

Key improvements:

*   **SEO-friendly title and headings:** Uses relevant keywords like "diffusion models", "image generation," etc.
*   **One-sentence hook:** Clearly defines what the library does.
*   **Bulleted key features:** Highlights the core benefits in an easy-to-read format.
*   **Clearer language:** Improved readability and flow.
*   **Concise summaries:**  Shorter descriptions where appropriate.
*   **Emphasis on benefits:** Focuses on what users can *do* with the library.
*   **Links to original repo:**  Added a direct link at the beginning.
*   **Removed extraneous comments:** Cleans up the code.
*   **Formatted for readability:** Consistent formatting.
*   **Keywords:** Incorporated relevant search terms to improve discoverability.