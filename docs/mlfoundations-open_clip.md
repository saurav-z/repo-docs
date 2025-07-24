# OpenCLIP: Open Source Implementation of CLIP

**Unlock the power of image-text understanding with OpenCLIP, a versatile library for contrastive language-image pre-training, enabling zero-shot image recognition, and more.** Explore the [paper](https://arxiv.org/abs/2212.07143) and [interactive Colabs](https://colab.research.google.com/github/mlfoundations/open_clip/blob/master/docs/Interacting_with_open_clip.ipynb) to get started!  Find the code on [GitHub](https://github.com/mlfoundations/open_clip).

[![PyPI version](https://img.shields.io/pypi/v/open_clip_torch.svg)](https://pypi.python.org/pypi/open_clip_torch)

## Key Features

*   **Pre-trained Models:** Access a wide range of pre-trained models trained on diverse datasets, including LAION-400M, LAION-2B, and DataComp-1B.  Find details on available models [here](docs/PRETRAINED.md).
*   **Reproducible Results:**  Leverage models studied in the paper ["Reproducible scaling laws for contrastive language-image learning"](https://arxiv.org/abs/2212.07143).
*   **Zero-Shot Capabilities:**  Achieve impressive zero-shot performance on various image recognition tasks.
*   **Flexible Usage:**  Easily integrate OpenCLIP into your projects with a straightforward installation process.
*   **Training Support:** Comprehensive instructions and tools are provided for training your own CLIP models.
*   **CoCa Support:** Support for [CoCa](https://arxiv.org/abs/2205.01917) models, see [tutorial](https://colab.research.google.com/github/mlfoundations/open_clip/blob/master/docs/Interacting_with_open_coca.ipynb).

## Model Performance Highlights

| Model                 | Training Data | Resolution | # of Samples Seen | ImageNet Zero-shot Accuracy |
| :-------------------- | :------------ | :--------- | :---------------- | :-------------------------- |
| ConvNext-Base         | LAION-2B      | 256px      | 13B               | 71.5%                       |
| ConvNext-Large        | LAION-2B      | 320px      | 29B               | 76.9%                       |
| ConvNext-XXLarge      | LAION-2B      | 256px      | 34B               | 79.5%                       |
| ViT-B-32-256          | DataComp-1B   | 256px      | 34B               | 72.8%                       |
| ... and more!         | ...           | ...        | ...               | ...                         |
| PE-Core-bigG-14-448   | MetaCLIP-5.4B | 448px      | 86B               | 85.4%                       |

*See [docs/PRETRAINED.md](docs/PRETRAINED.md) for a complete list and details.*

## Getting Started

### Installation

```bash
pip install open_clip_torch
```

### Usage Example

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

### Pretrained Models

Explore available pretrained models with:

```python
import open_clip
open_clip.list_pretrained()
```

More details on models (parameters, FLOPs, etc.) are available [here](docs/model_profile.csv).

### Loading Models

Load models using `open_clip.create_model_and_transforms`. The `pretrained` argument accepts model names from `open_clip.list_pretrained()` or local paths, and also supports checkpoints from Hugging Face.

## Training and Fine-tuning

This repository focuses on CLIP model training. For fine-tuning on downstream tasks, see [WiSE-FT](https://github.com/mlfoundations/wise-ft).

### Training CLIP

*   **Installation:**  Install with `pip install 'open_clip_torch[training]'` after creating a virtual environment.
*   **Example Training Command:**

```bash
python -m open_clip_train.main \
    --save-frequency 1 \
    --zeroshot-frequency 1 \
    --report-to tensorboard \
    --train-data="/path/to/train_data.csv"  \
    --val-data="/path/to/validation_data.csv"  \
    --csv-img-key filepath \
    --csv-caption-key title \
    --imagenet-val=/path/to/imagenet/root/val/ \
    --warmup 10000 \
    --batch-size=128 \
    --lr=1e-3 \
    --wd=0.1 \
    --epochs=30 \
    --workers=8 \
    --model RN50
```

*   **Multi-GPU and Distributed Training:**  Detailed instructions and examples are provided for multi-GPU training, SLURM clusters, and other distributed training scenarios.

## Evaluation

We recommend using [CLIP_benchmark](https://github.com/LAION-AI/CLIP_benchmark#how-to-use) for systematic evaluation on 40 datasets.

### Evaluating a Local Checkpoint

```bash
python -m open_clip_train.main \
    --val-data="/path/to/validation_data.csv"  \
    --model RN101 \
    --pretrained /path/to/checkpoints/epoch_K.pt
```

### Evaluating a Pretrained Checkpoint

```bash
python -m open_clip_train.main \
    --imagenet-val /path/to/imagenet/validation \
    --model ViT-B-32-quickgelu \
    --pretrained laion400m_e32
```

## Additional Resources

*   **[CLIP Colab](https://colab.research.google.com/github/mlfoundations/open_clip/blob/master/docs/Interacting_with_open_clip.ipynb)**: Interactive Colab notebook to explore the features.
*   **[CoCa Colab](https://colab.research.google.com/github/mlfoundations/open_clip/blob/master/docs/Interacting_with_open_coca.ipynb)**: Interactive Colab notebook to explore the features.
*   **Model Cards:** Find model-specific details on the Hugging Face Hub under the OpenCLIP library tag:  https://huggingface.co/models?library=open_clip.

## Acknowledgments

*   Gauss Centre for Supercomputing e.V. (www.gauss-centre.eu)
*   John von Neumann Institute for Computing (NIC)
*   Original contributors (See README)

## The Team

Led by [Ross Wightman](https://rwightman.com/), [Romain Beaumont](https://github.com/rom1504), [Cade Gordon](http://cadegordon.io/), and [Vaishaal Shankar](http://vaishaal.com/).

## Citation

If you find this repository useful, please cite the original work:

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