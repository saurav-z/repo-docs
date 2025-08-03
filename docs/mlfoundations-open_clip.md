# OpenCLIP: Open Source Implementation of CLIP for Image-Text Understanding

**OpenCLIP provides an open-source implementation of OpenAI's CLIP (Contrastive Language-Image Pre-training), enabling powerful image-text understanding and retrieval.  Explore state-of-the-art models and reproduce research results with ease!** [View the Original Repository](https://github.com/mlfoundations/open_clip)

Key Features:

*   **Pre-trained Models:** Access a wide range of pre-trained CLIP models trained on diverse datasets, including LAION-400M, LAION-2B, and DataComp-1B, with detailed performance metrics like zero-shot ImageNet accuracy.
*   **Reproducible Research:** Leverage our codebase to reproduce state-of-the-art results and explore the scaling properties of CLIP models, as detailed in our research paper.
*   **Flexible Usage:** Easily integrate OpenCLIP into your projects with straightforward installation and usage examples, including clear code snippets for image and text encoding.
*   **Fine-tuning Support:** Fine-tune pre-trained models on downstream classification tasks or explore our separate repository, WiSE-FT, for robust fine-tuning techniques.
*   **Comprehensive Data Support:** Benefit from support for various datasets, including webdataset format, and tools for downloading datasets like Conceptual Captions (CC3M) and YFCC.
*   **Training & Evaluation Utilities:** Utilize our comprehensive training scripts, including support for multi-GPU and distributed training setups, model distillation, and evaluation tools.
*   **Model Distillation:** Easily distill from a pre-trained model.
*   **Int8 Support:** Beta support for int8 training and inference.

## Quick Start: Installation and Basic Usage

```bash
pip install open_clip_torch
```

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

print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]
```

## Pre-trained Models

OpenCLIP offers a convenient interface to load pre-trained models.  Use the following code snippet to list available models:

```python
import open_clip
open_clip.list_pretrained()
```
More details about our pretrained models are available [here](docs/PRETRAINED.md).  Model cards with additional model specific details can be found on the Hugging Face Hub under the OpenCLIP library tag: https://huggingface.co/models?library=open_clip.

## Training CLIP

Follow the instructions in the original repo for setting up your training environment.

## Evaluation and Zero-Shot Prediction

For systematic evaluation, consider using [CLIP_benchmark](https://github.com/LAION-AI/CLIP_benchmark).

## Acknowledgments and Citation

This project builds upon the foundational work of OpenAI and benefits from the contributions of many researchers.  Please cite the appropriate papers if you use this repository.

*   [Original Paper](https://arxiv.org/abs/2212.07143)
*   [Cite OpenCLIP](https://github.com/mlfoundations/open_clip#citing)
*   Other citations available in original README.

[![DOI](https://zenodo.org/badge/390536799.svg)](https://zenodo.org/badge/latestdoi/390536799)