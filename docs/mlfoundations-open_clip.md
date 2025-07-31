# OpenCLIP: Open Source Implementation of CLIP for Image-Text Understanding

**OpenCLIP provides open-source, high-performing models trained using Contrastive Language-Image Pre-training (CLIP), enabling powerful image-text understanding capabilities.** [Original Repo](https://github.com/mlfoundations/open_clip)

*   [Paper](https://arxiv.org/abs/2212.07143)
*   [Citations](#citing)
*   [CLIP Colab](https://colab.research.google.com/github/mlfoundations/open_clip/blob/master/docs/Interacting_with_open_clip.ipynb)
*   [CoCa Colab](https://colab.research.google.com/github/mlfoundations/open_clip/blob/master/docs/Interacting_with_open_coca.ipynb)
*   [![pypi](https://img.shields.io/pypi/v/open_clip_torch.svg)](https://pypi.python.org/pypi/open_clip_torch)

OpenCLIP offers a robust and versatile framework for training and utilizing CLIP models, allowing you to explore the intersection of images and text. This repository offers:

*   **Pre-trained Models:** Access a diverse collection of pre-trained models, including those trained on datasets like LAION-400M, LAION-2B, and DataComp-1B, covering a wide range of model architectures and sizes.
*   **Reproducible Scaling Laws:** Leverage models and findings detailed in the paper [reproducible scaling laws for contrastive language-image learning](https://arxiv.org/abs/2212.07143).
*   **Ease of Use:** A simple API for instantiating and utilizing pre-trained models, enabling quick integration into your projects.
*   **Fine-tuning Support:** Instructions for fine-tuning pre-trained models on downstream classification tasks.
*   **Training Tools:** Comprehensive training scripts and utilities for training your own CLIP models, including support for multi-GPU and distributed training.
*   **Model Performance:** Access high-performing models with impressive zero-shot ImageNet-1k accuracy, including ConvNext, ViT, and SigLIP models.

## Key Features

*   **State-of-the-Art Models:** Utilize models with impressive zero-shot performance, including ViT and ConvNext architectures, pre-trained on large-scale datasets.
*   **Flexible Training:** Fine-tune pre-trained models or train new models from scratch with extensive training options, including multi-GPU and distributed training support, with options to integrate with SLURM clusters.
*   **Efficient Training:** Optimized training strategies, including gradient accumulation and int8 support, for faster and more efficient model training.
*   **Easy Integration:** Simple API for model loading and usage, including built-in tokenizers and preprocessors, for easy integration into projects.
*   **Model Distillation:** Distill models from pre-trained models.
*   **CoCa Support:** Support for CoCa models that enable the generation of text from images.

## Quick Start

Install the library:

```bash
pip install open_clip_torch
```

Get started with a basic example:

```python
import torch
from PIL import Image
import open_clip

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
model.eval()
tokenizer = open_clip.get_tokenizer('ViT-B-32')

image = preprocess(Image.open("docs/CLIP.png")).unsqueeze(0)
text = tokenizer(["a diagram", "a dog", "a cat"])

with torch.no_grad(), torch.autocast("cuda"):
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)
```

## Pretrained Models

Discover available pretrained models:

```python
>>> import open_clip
>>> open_clip.list_pretrained()
```

More details about available models can be found [here](docs/PRETRAINED.md) and in the [Hugging Face Hub](https://huggingface.co/models?library=open_clip).

## Training and Usage

Detailed instructions for training, fine-tuning, and using OpenCLIP can be found in the sections below, including how to set up your training environment, run training scripts, and evaluate model performance, and how to fine tune models.

## Example Model Performance

| Model              | Training data | Resolution | # of samples seen | ImageNet zero-shot acc. |
| ------------------ | ------------- | ---------- | ----------------- | ----------------------- |
| ConvNext-Base      | LAION-2B      | 256px      | 13B               | 71.5%                   |
| ConvNext-Large     | LAION-2B      | 320px      | 29B               | 76.9%                   |
| ConvNext-XXLarge   | LAION-2B      | 256px      | 34B               | 79.5%                   |
| ViT-B-32-256       | DataComp-1B   | 256px      | 34B               | 72.8%                   |
| ViT-B-16           | DataComp-1B   | 224px      | 13B               | 73.5%                   |
| ViT-L-14           | LAION-2B      | 224px      | 32B               | 75.3%                   |
| ViT-H-14           | LAION-2B      | 224px      | 32B               | 78.0%                   |
| ViT-L-14           | DataComp-1B   | 224px      | 13B               | 79.2%                   |
| ViT-bigG-14        | LAION-2B      | 224px      | 34B               | 80.1%                   |

*See original README for additional model comparisons.*

## Acknowledgments

*See original README for acknowledgements.*

## Citing

If you found this repository useful, please consider citing:

```bibtex
@software{ilharco_gabriel_2021_5143773,
  author       = {Ilharco, Gabriel and
                  Wortsman, Mitchell and
                  Wightman, Ross and
                  Gordon, Cade and
                  Carlini, Nicholas and
                  Taori, Rohan and
                  Dave, Achal and
                  Shankar, Vaishaal and
                  Namkoong, Hongseok and
                  Miller, John and
                  Hajishirzi, Hannaneh and
                  Farhadi, Ali and
                  Schmidt, Ludwig},
  title        = {OpenCLIP},
  month        = jul,
  year         = 2021,
  note         = {If you use this software, please cite it as below.},
  publisher    = {Zenodo},
  version      = {0.1},
  doi          = {10.5281/zenodo.5143773},
  url          = {https://doi.org/10.5281/zenodo.5143773}
}
```

*See original README for additional citations.*

[![DOI](https://zenodo.org/badge/390536799.svg)](https://zenodo.org/badge/latestdoi/390536799)