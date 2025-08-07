# OpenCLIP: Open Source Implementation of CLIP for Image-Text Learning

**OpenCLIP provides a powerful and flexible open-source implementation of OpenAI's CLIP (Contrastive Language-Image Pre-training) model, enabling cutting-edge image-text understanding and generation.**  [Explore the original repository](https://github.com/mlfoundations/open_clip).

*   **[Paper](https://arxiv.org/abs/2212.07143)**
*   **[Citations](#citing)**
*   **[Clip Colab](https://colab.research.google.com/github/mlfoundations/open_clip/blob/master/docs/Interacting_with_open_clip.ipynb)**
*   **[Coca Colab](https://colab.research.google.com/github/mlfoundations/open_clip/blob/master/docs/Interacting_with_open_coca.ipynb)**
*   [![PyPI](https://img.shields.io/pypi/v/open_clip_torch.svg)](https://pypi.python.org/pypi/open_clip_torch)

OpenCLIP allows you to leverage the power of CLIP to create models that understand the relationship between images and text. It offers pre-trained models, training scripts, and tools to fine-tune and evaluate models on a variety of tasks, along with support for CoCa models and Hugging Face hub integration.

## Key Features

*   **Pre-trained Models:** Access a wide range of pre-trained CLIP models trained on diverse datasets like LAION-400M, LAION-2B, and DataComp-1B.  Find more information about these pre-trained models [here](docs/PRETRAINED.md).
*   **Reproducible Scaling Laws:**  The project's scaling properties are detailed in the paper "[reproducible scaling laws for contrastive language-image learning](https://arxiv.org/abs/2212.07143)."
*   **Flexible Training:** Utilize provided scripts and extensive documentation for training CLIP models, including multi-GPU support, data loading from webdataset, and various optimization techniques.
*   **CoCa Support:** Includes training and inference capabilities for CoCa (Contrastive Captions) models, enabling text generation from images.
*   **Hugging Face Hub Integration:** Easily push and load models from the Hugging Face Hub, simplifying model sharing and deployment.
*   **Fine-tuning Support:** Code provided for fine-tuning on downstream classification tasks is available [here](https://github.com/mlfoundations/wise-ft).

## Quick Start

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

print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]
```

### Pretrained Models

To list available pretrained models:

```python
>>> import open_clip
>>> open_clip.list_pretrained()
```

Find details like parameters and FLOPs [here](docs/model_profile.csv).

### Loading Models

```python
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
```

You can also load models from local paths or Hugging Face Hub:

```python
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='/path/to/open_clip_pytorch_model.bin')
```

### Pre-Trained Model Performance

| Model              | Training Data | Resolution | Samples Seen | ImageNet Zero-Shot Accuracy |
| ------------------ | ------------- | ---------- | ------------ | --------------------------- |
| ConvNext-Base      | LAION-2B      | 256px      | 13B          | 71.5%                       |
| ConvNext-Large     | LAION-2B      | 320px      | 29B          | 76.9%                       |
| ConvNext-XXLarge   | LAION-2B      | 256px      | 34B          | 79.5%                       |
| ViT-B-32-256       | DataComp-1B   | 256px      | 34B          | 72.8%                       |
| ViT-B-16           | DataComp-1B   | 224px      | 13B          | 73.5%                       |
| ViT-L-14           | LAION-2B      | 224px      | 32B          | 75.3%                       |
| ViT-H-14           | LAION-2B      | 224px      | 32B          | 78.0%                       |
| ViT-L-14           | DataComp-1B   | 224px      | 13B          | 79.2%                       |
| ViT-bigG-14        | LAION-2B      | 224px      | 34B          | 80.1%                       |
| ViT-L-14-quickgelu | WIT           | 224px      | 13B          | 75.5%                       |
| ViT-SO400M-14-SigLIP | WebLI       | 224px      | 45B          | 82.0%                       |
| ViT-L-14 (DFN) | DFN-2B       | 224px      | 39B          | 82.2%                       |
| ViT-L-16-256 (SigLIP2) | WebLI (multi-lang)       | 256px      | 40B          | 82.5%                       |
| ViT-SO400M-14-SigLIP-384 | WebLI       | 384px      | 45B          | 83.1%                       |
| ViT-H-14-quickgelu (DFN) | DFN-5B       | 224px      | 39B          | 83.4%                       |
| PE-Core-L-14-336 | MetaCLIP-5.4B       | 336px      | 58B          | 83.5%                       |
| ViT-SO400M-16-SigLIP2-384 | WebLI (multi-lang)      | 384px      | 40B          | 84.1%                       |
| ViT-H-14-378-quickgelu (DFN) | DFN-5B       | 378px      | 44B          | 84.4%                       |
| ViT-gopt-16-SigLIP2-384 | WebLI (multi-lang)      | 384px      | 40B          | 85.0%                       |
| PE-Core-bigG-14-448 | MetaCLIP-5.4B       | 448px      | 86B          | 85.4%                       |

## Training

Detailed instructions are available for setting up your environment, training models, multi-GPU support, and fine-tuning on classification tasks within the [original repository](https://github.com/mlfoundations/open_clip).

## Evaluation

Use [CLIP_benchmark](https://github.com/LAION-AI/CLIP_benchmark#how-to-use) for evaluating trained models.

## Acknowledgments

This project benefits from the computing resources provided by the Gauss Centre for Supercomputing e.V. (www.gauss-centre.eu) through the John von Neumann Institute for Computing (NIC) on the GCS Supercomputer JUWELS Booster at Jülich Supercomputing Centre (JSC).

## Team

The core development team includes [Ross Wightman](https://rwightman.com/), [Romain Beaumont](https://github.com/rom1504), [Cade Gordon](http://cadegordon.io/), and [Vaishaal Shankar](http://vaishaal.com/).

## Citing

If you use this repository, please cite the following:

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

and

```bibtex
@inproceedings{cherti2023reproducible,
  title={Reproducible scaling laws for contrastive language-image learning},
  author={Cherti, Mehdi and Beaumont, Romain and Wightman, Ross and Wortsman, Mitchell and Ilharco, Gabriel and Gordon, Cade and Schuhmann, Christoph and Schmidt, Ludwig and Jitsev, Jenia},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={2818--2829},
  year={2023}
}
```

and
```bibtex
@inproceedings{Radford2021LearningTV,
  title={Learning Transferable Visual Models From Natural Language Supervision},
  author={Alec Radford and Jong Wook Kim and Chris Hallacy and A. Ramesh and Gabriel Goh and Sandhini Agarwal and Girish Sastry and Amanda Askell and Pamela Mishkin and Jack Clark and Gretchen Krueger and Ilya Sutskever},
  booktitle={ICML},
  year={2021}
}
```

and
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