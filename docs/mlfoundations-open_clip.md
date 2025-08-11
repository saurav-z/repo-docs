# OpenCLIP: Open Source Implementation of CLIP for Image-Text Learning

**Unlock the power of visual understanding with OpenCLIP, an open-source implementation of OpenAI's Contrastive Language-Image Pre-training (CLIP).** Explore a wide range of pre-trained models and training capabilities, enabling you to connect images and text effectively.  Learn more about the research behind OpenCLIP in the [original paper](https://arxiv.org/abs/2212.07143) and see how it compares to other models in the paper's [table of results](docs/openclip_results.csv).

[![PyPI version](https://img.shields.io/pypi/v/open_clip_torch.svg)](https://pypi.python.org/pypi/open_clip_torch)
[![DOI](https://zenodo.org/badge/390536799.svg)](https://zenodo.org/badge/latestdoi/390536799)

[Original Repo](https://github.com/mlfoundations/open_clip) | [Paper](https://arxiv.org/abs/2212.07143) | [Citations](#citing) | [CLIP Colab](https://colab.research.google.com/github/mlfoundations/open_clip/blob/master/docs/Interacting_with_open_clip.ipynb) | [CoCa Colab](https://colab.research.google.com/github/mlfoundations/open_clip/blob/master/docs/Interacting_with_open_coca.ipynb)

## Key Features

*   **Pre-trained Models:** Access a diverse collection of pre-trained models, including those trained on LAION-400M, LAION-2B, and DataComp-1B datasets.  See [here](docs/PRETRAINED.md) for more information.
*   **Reproducible Results:**  Reproduce and explore the scaling properties of contrastive language-image learning, as detailed in the research paper.
*   **Flexible Training:**  Train your own CLIP models with customizable configurations, supporting multi-GPU and distributed training.
*   **Model Distillation:** Leverage model distillation for improved performance or efficiency.
*   **Easy to Use:** Simple model interface to instantiate both pre-trained and untrained models.
*   **Evaluation:** Evaluate your models on various datasets using the provided scripts or through integration with [CLIP benchmark](https://github.com/LAION-AI/CLIP_benchmark#how-to-use).
*   **Int8 Support:** Enable int8 training and inference using `--use-bnb-linear SwitchBackLinearGlobal` or `--use-bnb-linear SwitchBackLinearGlobalMemEfficient`.
*   **CoCa Model Support:** Train and generate text with CoCa models ([CoCa](https://arxiv.org/abs/2205.01917)).

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

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
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

For more detailed examples and usage instructions, see the [[Clip Colab]](https://colab.research.google.com/github/mlfoundations/open_clip/blob/master/docs/Interacting_with_open_clip.ipynb).

## Pre-trained Models

Explore the available pre-trained models with `open_clip.list_pretrained()`.  More model details can be found [here](docs/PRETRAINED.md). Example of zero-shot ImageNet-1k accuracy results of select models:

| Model                | Training data   | Resolution | ImageNet zero-shot acc. |
| -------------------- | --------------- | ---------- | ----------------------- |
| ConvNext-Large       | LAION-2B        | 320px      | 76.9%                  |
| ViT-H-14             | LAION-2B        | 224px      | 78.0%                  |
| ViT-L-14-quickgelu    | WIT             | 224px      | 75.5%                  |
| PE-Core-bigG-14-448 | MetaCLIP-5.4B   | 448px      | 85.4%                  |
| ViT-gopt-16-SigLIP2-384 | WebLI (multi-lang) | 384px      | 85.0%                  |

## Fine-tuning

This repository focuses on training CLIP models. For fine-tuning *trained* zero-shot models on downstream tasks, see [WiSE-FT](https://github.com/mlfoundations/wise-ft).

## Training and Evaluation

### Training

Comprehensive training instructions, including multi-GPU setup, SLURM scripts, and data preparation, are provided in the original README. Training is highly customizable; consult the `--help` flag for the training script for detailed options.

### Evaluation

Use the provided scripts for evaluating checkpoints, or leverage the [CLIP benchmark](https://github.com/LAION-AI/CLIP_benchmark#how-to-use) for comprehensive evaluations.

## Acknowledgments

This project was supported by the Gauss Centre for Supercomputing e.V. (www.gauss-centre.eu) and funded by the John von Neumann Institute for Computing (NIC) on the GCS Supercomputer JUWELS Booster at Jülich Supercomputing Centre (JSC).

## The Team

Current development of this repository is led by [Ross Wightman](https://rwightman.com/), [Romain Beaumont](https://github.com/rom1504), [Cade Gordon](http://cadegordon.io/), and [Vaishaal Shankar](http://vaishaal.com.).

## Citing

If you use this project, please consider citing the relevant papers:

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

```

[![DOI](https://zenodo.org/badge/390536799.svg)](https://zenodo.org/badge/latestdoi/390536799)