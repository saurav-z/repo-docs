# OpenCLIP: State-of-the-Art Image-Text Understanding

**OpenCLIP provides open-source implementations of CLIP models, enabling cutting-edge image-text understanding through contrastive learning.**

[[Paper]](https://arxiv.org/abs/2212.07143) | [[Citations]](#citing) | [[Colab: CLIP]](https://colab.research.google.com/github/mlfoundations/open_clip/blob/master/docs/Interacting_with_open_clip.ipynb) | [[Colab: CoCa]](https://colab.research.google.com/github/mlfoundations/open_clip/blob/master/docs/Interacting_with_open_coca.ipynb) | [![PyPI](https://img.shields.io/pypi/v/open_clip_torch.svg)](https://pypi.python.org/pypi/open_clip_torch) | [Original Repo](https://github.com/mlfoundations/open_clip)

OpenCLIP offers a powerful toolkit for researchers and developers interested in image-text models, providing:

*   **Pre-trained Models:** Access a wide range of models trained on diverse datasets like LAION-400M, LAION-2B, and DataComp-1B, including state-of-the-art architectures.
*   **Reproducibility:**  Train your own CLIP models, and reproduce the models of the paper "[Reproducible scaling laws for contrastive language-image learning](https://arxiv.org/abs/2212.07143)".
*   **Flexible Usage:** Easily integrate OpenCLIP models into your projects with a simple Python API for encoding images and text.
*   **Efficient Training:** Optimize training with multi-GPU support, gradient accumulation, and other advanced features.
*   **Fine-tuning:** Code for fine-tuning the zero-shot models is available in [WiSE-FT](https://github.com/mlfoundations/wise-ft).
*   **CoCa Support:** Supports training and interacting with CoCa models.

## Key Features

*   **Wide Variety of Models:** Includes ConvNext, ViT (Vision Transformer), and SigLIP models, with different sizes and training data.
*   **Zero-Shot Performance:** Explore impressive zero-shot ImageNet-1k accuracy and performance on 38 datasets.
*   **Easy Installation:** Install with `pip install open_clip_torch`.
*   **Model Loading:** Easily load pre-trained and untrained models with `open_clip.create_model_and_transforms`.
*   **Training Scripts:** Offers comprehensive training scripts for single and multi-GPU environments, including SLURM support.
*   **Data Handling:** Supports webdataset and CSV datasets.
*   **Int8 Support:** Beta support for int8 training and inference for faster training and inference.
*   **Hugging Face Integration:** Push models to the Hugging Face Hub.

## Quick Start

### Installation

```bash
pip install open_clip_torch
```

### Usage

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

See [here](docs/PRETRAINED.md) for more details about the pre-trained models. You can find available models using:

```python
import open_clip
open_clip.list_pretrained()
```

## Training CLIP

Detailed instructions for training CLIP models are available in the original README.

## Evaluation and Zero-Shot

Evaluate your models using the provided scripts or integrate them with external evaluation frameworks.  We recommend [CLIP_benchmark](https://github.com/LAION-AI/CLIP_benchmark#how-to-use).

## CoCa

OpenCLIP also supports CoCa models. See details in the original README.

## Acknowledgments

We gratefully acknowledge the Gauss Centre for Supercomputing e.V. (www.gauss-centre.eu) for funding this part of work by providing computing time through the John von Neumann Institute for Computing (NIC) on the GCS Supercomputer JUWELS Booster at Jülich Supercomputing Centre (JSC).

## The Team

Current development of this repository is led by [Ross Wightman](https://rwightman.com/), [Romain Beaumont](https://github.com/rom1504), [Cade Gordon](http://cadegordon.io/), and [Vaishaal Shankar](http://vaishaal.com/).

The original version of this repository is from a group of researchers at UW, Google, Stanford, Amazon, Columbia, and Berkeley.

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

```bibtex
@inproceedings{cherti2023reproducible,
  title={Reproducible scaling laws for contrastive language-image learning},
  author={Cherti, Mehdi and Beaumont, Romain and Wightman, Ross and Wortsman, Mitchell and Ilharco, Gabriel and Gordon, Cade and Schuhmann, Christoph and Schmidt, Ludwig and Jitsev, Jenia},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={2818--2829},
  year={2023}
}
```

```bibtex
@inproceedings{Radford2021LearningTV,
  title={Learning Transferable Visual Models From Natural Language Supervision},
  author={Alec Radford and Jong Wook Kim and Chris Hallacy and A. Ramesh and Gabriel Goh and Sandhini Agarwal and Girish Sastry and Amanda Askell and Pamela Mishkin and Jack Clark and Gretchen Krueger and Ilya Sutskever},
  booktitle={ICML},
  year={2021}
}
```

```bibtex
@inproceedings{schuhmann2022laionb,
  title={{LAION}-5B: An open large-scale dataset for training next generation image-text models},
  author={Christoph Schuhmann and
          Romain Beaumont and
          Richard Vencu and
          Cade W Gordon and
          Ross Wightman and
          Mehdi Cherti and
          Theo Coombes and
          Aarush Katta and
          Clayton Mullis and
          Mitchell Wortsman and
          Patrick Schramowski and
          Srivatsa R Kundurthy and
          Katherine Crowson and
          Ludwig Schmidt and
          Robert Kaczmarczyk and
          Jenia Jitsev},
  booktitle={Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2022},
  url={https://openreview.net/forum?id=M3Y74vmsMcY}
}
```

[![DOI](https://zenodo.org/badge/390536799.svg)](https://zenodo.org/badge/latestdoi/390536799)