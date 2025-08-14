# OpenCLIP: Open Source Implementation of CLIP for Image-Text Learning

**OpenCLIP** is an open-source initiative to reproduce and advance the state-of-the-art in contrastive language-image pre-training (CLIP), offering a comprehensive toolkit for researchers and developers to explore and build upon this powerful technology. [(Paper)](https://arxiv.org/abs/2212.07143) | [(Citations)](#citing) | [(Clip Colab)](https://colab.research.google.com/github/mlfoundations/open_clip/blob/master/docs/Interacting_with_open_clip.ipynb) | [(Coca Colab)](https://colab.research.google.com/github/mlfoundations/open_clip/blob/master/docs/Interacting_with_open_coca.ipynb)
[![PyPI](https://img.shields.io/pypi/v/open_clip_torch.svg)](https://pypi.python.org/pypi/open_clip_torch)

OpenCLIP provides a robust and flexible framework for training, evaluating, and utilizing CLIP models. It features a wide range of pre-trained models and comprehensive training tools to empower researchers to explore the potential of image-text understanding.  This repository allows you to:

*   **Access a diverse library of pre-trained models:** Including models trained on datasets like LAION-400M, LAION-2B, and DataComp-1B, with varying architectures and performance characteristics.
*   **Train your own CLIP models:**  With flexible training scripts supporting multi-GPU, multi-node, and SLURM environments, enabling you to train CLIP models from scratch or fine-tune pre-trained models.
*   **Easily integrate into your projects:** Use the provided Python package and examples to encode images and text, perform zero-shot classification, and explore other image-text tasks.
*   **Fine-tune and experiment with CoCa models:** Experiment with cutting-edge CoCa models which leverage both contrastive and generative losses.
*   **Leverage Hugging Face Hub:** Easily push and use the open_clip models on the Hugging Face Hub.

## Key Features

*   **Pre-trained Model Zoo:** Extensive collection of pre-trained models with varying architectures and datasets, including state-of-the-art models and OpenAI's ViT-L models.
*   **Flexible Training Framework:** Supports multi-GPU and multi-node training, SLURM integration, and features for data augmentation, mixed-precision training, and distributed training strategies.
*   **Easy-to-Use API:** Simple Python API for loading models, encoding images and text, and performing zero-shot classification.
*   **Comprehensive Documentation:** Detailed documentation and tutorials to guide you through the process of using and contributing to the library.
*   **CoCa Support:** Support for both training and generation with CoCa models, facilitating the exploration of multimodal understanding.
*   **Int8 Support:** Support for int8 training and inference.
*   **Remote Training Support:** Option to resume from remote files (e.g., s3 buckets)

## Quick Start

### Installation

```bash
pip install open_clip_torch
```

### Basic Usage

```python
import torch
from PIL import Image
import open_clip

# Load a pre-trained model
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
model.eval()  # Set the model to evaluation mode

# Get the tokenizer for the model
tokenizer = open_clip.get_tokenizer('ViT-B-32')

# Preprocess the image
image = preprocess(Image.open("docs/CLIP.png")).unsqueeze(0)

# Tokenize the text
text = tokenizer(["a diagram", "a dog", "a cat"])

# Perform inference
with torch.no_grad(), torch.autocast("cuda"):
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)
```

## Pre-trained Models

Explore our full collection of pre-trained models [here](docs/PRETRAINED.md).  You can list available models with:

```python
import open_clip
open_clip.list_pretrained()
```

## Fine-tuning

For fine-tuning on downstream classification tasks, see [WiSE-FT](https://github.com/mlfoundations/wise-ft).

## Training

Detailed instructions for training CLIP models, including multi-GPU/node setup and data preparation, are available in the [Training CLIP](#training-clip) section of the README.

## Evaluation

We recommend using [CLIP_benchmark](https://github.com/LAION-AI/CLIP_benchmark#how-to-use) for systematic evaluation on 40 datasets.

## Acknowledgments

We gratefully acknowledge the Gauss Centre for Supercomputing e.V. (www.gauss-centre.eu) for funding this part of work by providing computing time through the John von Neumann Institute for Computing (NIC) on the GCS Supercomputer JUWELS Booster at Jülich Supercomputing Centre (JSC).

## The Team

Current development of this repository is led by [Ross Wightman](https://rwightman.com/), [Romain Beaumont](https://github.com/rom1504), [Cade Gordon](http://cadegordon.io/), and [Vaishaal Shankar](http://vaishaal.com/).

The original version of this repository is from a group of researchers at UW, Google, Stanford, Amazon, Columbia, and Berkeley.

[Gabriel Ilharco*](http://gabrielilharco.com/), [Mitchell Wortsman*](https://mitchellnw.github.io/), [Nicholas Carlini](https://nicholas.carlini.com/), [Rohan Taori](https://www.rohantaori.com/), [Achal Dave](http://www.achaldave.com/), [Vaishaal Shankar](http://vaishaal.com/), [John Miller](https://people.eecs.berkeley.edu/~miller_john/), [Hongseok Namkoong](https://hsnamkoong.github.io/), [Hannaneh Hajishirzi](https://homes.cs.washington.edu/~hannaneh/), [Ali Farhadi](https://homes.cs.washington.edu/~ali/), [Ludwig Schmidt](https://people.csail.mit.edu/ludwigs/)

Special thanks to [Jong Wook Kim](https://jongwook.kim/) and [Alec Radford](https://github.com/Newmu) for help with reproducing CLIP!

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

---

**[Go to the original repository](https://github.com/mlfoundations/open_clip)**