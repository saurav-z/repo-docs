# OpenCLIP: Open Source CLIP Implementation for Image-Text Understanding

**OpenCLIP empowers researchers and developers with a powerful open-source implementation of OpenAI's CLIP (Contrastive Language-Image Pre-training) model, enabling advanced image-text understanding capabilities.**  Explore the original repo [here](https://github.com/mlfoundations/open_clip).

*   **Reproducible Scaling Laws:** Train models on diverse datasets, from small-scale experiments to massive datasets like LAION-2B and DataComp-1B.  Our paper details reproducible scaling laws for contrastive language-image learning.
*   **Pre-trained Models:** Access a wide array of pre-trained models, including ConvNext, ViT, and SigLIP, with detailed zero-shot ImageNet-1k accuracy results.  Browse a comprehensive list of pretrained models [here](docs/PRETRAINED.md).
*   **Flexible Usage:**  Easily integrate OpenCLIP into your projects using the provided Python package and example code.  Leverage our [interactive Colab notebooks](https://colab.research.google.com/github/mlfoundations/open_clip/blob/master/docs/Interacting_with_open_clip.ipynb) and [CoCa Colab](https://colab.research.google.com/github/mlfoundations/open_clip/blob/master/docs/Interacting_with_open_coca.ipynb) for hands-on experience.
*   **Training & Fine-tuning:** Comprehensive training scripts and guidance for fine-tuning on downstream classification tasks.  For fine-tuning code, see [WiSE-FT](https://github.com/mlfoundations/wise-ft) for robust fine-tuning techniques.
*   **CoCa Model Support:** Includes training and generation examples for CoCa models (Contrastive Captions), allowing for image captioning and text generation tasks.
*   **Efficient Data Handling:** Supports webdataset format and integration with img2dataset for large-scale datasets.

## Key Features

*   **State-of-the-art Models:** Utilize models trained on massive datasets for high-performance image-text understanding.
*   **Ease of Use:** Simple installation and a user-friendly API make it easy to get started.
*   **Customization:** Train your own models, fine-tune existing models, and experiment with different architectures.
*   **Scalability:** Supports multi-GPU and distributed training for efficient large-scale model training.
*   **Model Distillation:**  Distill models for improved performance on smaller models
*   **Int8 Support:** Beta support for int8 training and inference for faster training.

## Quick Start

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

print("Label probs:", text_probs)
```

## Pre-trained Models
Explore available pre-trained models using:

```python
>>> import open_clip
>>> open_clip.list_pretrained()
```
More details about the models we support (e.g. number of parameters, FLOPs) are available in [this table](docs/model_profile.csv).

## Training and Fine-tuning

Comprehensive instructions and examples are provided for training CLIP models from scratch.  Utilize multi-GPU training and options for optimization.  Detailed guides are also available for fine-tuning on downstream tasks such as image classification.
See the "Training CLIP" section in the original README for more details, including:

*   Installation instructions for training.
*   Single-process running code examples.
*   Multi-GPU and SLURM training setup.
*   Resuming from checkpoints.
*   CoCa Training, Generating, and Fine-tuning.
*   Model Distillation.
*   Gradient accumulation.
*   Int8 support.

## Evaluation

Evaluate your models using the provided scripts and the [CLIP_benchmark](https://github.com/LAION-AI/CLIP_benchmark) framework.

## Acknowledgments & Citation

This project is a collaborative effort.  We thank the Gauss Centre for Supercomputing e.V. and the John von Neumann Institute for Computing (NIC) for providing computing resources.

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
and the other citations available in the original README.