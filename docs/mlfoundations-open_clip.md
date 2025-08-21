# OpenCLIP: Open Source Implementation of CLIP (Contrastive Language-Image Pre-training)

**Unleash the power of visual understanding with OpenCLIP, a robust open-source implementation of OpenAI's CLIP, offering state-of-the-art image-text models and training tools. ➡️  [Explore the Original Repo](https://github.com/mlfoundations/open_clip)**

[![PyPI Version](https://img.shields.io/pypi/v/open_clip_torch.svg)](https://pypi.python.org/pypi/open_clip_torch)
[![Paper](https://img.shields.io/badge/paper-arXiv-red.svg)](https://arxiv.org/abs/2212.07143)
[![Citations](https://img.shields.io/badge/citations-see%20below-brightgreen)](/README.md#citing)
[![CLIP Colab](https://img.shields.io/badge/Colab-CLIP-blue.svg)](https://colab.research.google.com/github/mlfoundations/open_clip/blob/master/docs/Interacting_with_open_clip.ipynb)
[![Coca Colab](https://img.shields.io/badge/Colab-CoCa-blue.svg)](https://colab.research.google.com/github/mlfoundations/open_clip/blob/master/docs/Interacting_with_open_coca.ipynb)

OpenCLIP provides a comprehensive and efficient framework for training and utilizing CLIP models, enabling you to:

*   **Leverage Pre-trained Models:** Access a wide range of pre-trained CLIP models, including those trained on LAION-2B, DataComp-1B, and more, with varying architectures and sizes.
*   **Achieve High Accuracy:** Utilize models that achieve impressive zero-shot ImageNet-1k accuracy, surpassing some state-of-the-art alternatives.
*   **Fine-tune for Downstream Tasks:** Fine-tune pre-trained models on custom datasets using the [WiSE-FT repository](https://github.com/mlfoundations/wise-ft).
*   **Customize Training:** Train your own CLIP models from scratch or refine existing ones with flexible training configurations and support for multi-GPU and SLURM clusters.
*   **Experiment with Cutting-Edge Techniques:** Explore features like patch dropout, gradient accumulation, and Int8 support for enhanced training efficiency and performance.

## Key Features

*   **Extensive Pre-trained Models:** A diverse collection of pre-trained models, including ConvNext, ViT, and SigLIP architectures.
*   **Flexible Training Framework:**  Support for various datasets, including webdataset, and multi-node training.
*   **Ease of Use:** Simple API for model loading, feature extraction, and fine-tuning.
*   **Model Distillation:** Distill knowledge from larger models to create smaller, more efficient ones.
*   **Hugging Face Integration:** Seamlessly push your trained models to the Hugging Face Hub for easy sharing and deployment.

## Model Performance Highlights

Here are some of the top-performing models available through OpenCLIP:

| Model                  | Training Data | Resolution | ImageNet Zero-Shot Accuracy |
| ---------------------- | ------------- | ---------- | -------------------------- |
| ConvNext-Base          | LAION-2B      | 256px      | 71.5%                      |
| ConvNext-Large         | LAION-2B      | 320px      | 76.9%                      |
| ConvNext-XXLarge       | LAION-2B      | 256px      | 79.5%                      |
| ViT-B-32-256           | DataComp-1B   | 256px      | 72.8%                      |
| ViT-B-16               | DataComp-1B   | 224px      | 73.5%                      |
| ViT-L-14               | LAION-2B      | 224px      | 75.3%                      |
| ... and many more ...   | ...           | ...        | ...                        |

For a complete list of pre-trained models and their performance, see the [Pretrained Models Documentation](docs/PRETRAINED.md).

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

print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]
```

## Training

Detailed instructions for training CLIP models can be found in the [Training Guide](docs/TRAINING.md).  Key training parameters and best practices are outlined.

## Evaluation

For systematic evaluation on 40 datasets, we recommend using [CLIP_benchmark](https://github.com/LAION-AI/CLIP_benchmark#how-to-use).

### Evaluating Local Checkpoint

```bash
python -m open_clip_train.main \
    --val-data="/path/to/validation_data.csv"  \
    --model RN101 \
    --pretrained /path/to/checkpoints/epoch_K.pt
```

### Evaluating Hosted Pretrained Checkpoint on ImageNet Zero-Shot Prediction

```bash
python -m open_clip_train.main \
    --imagenet-val /path/to/imagenet/validation \
    --model ViT-B-32-quickgelu \
    --pretrained laion400m_e32
```

## Contributions

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Acknowledgements

We are grateful for the support from the Gauss Centre for Supercomputing e.V. (www.gauss-centre.eu) through the John von Neumann Institute for Computing (NIC) at Jülich Supercomputing Centre (JSC).

## The Team

This project is led by [Ross Wightman](https://rwightman.com/), [Romain Beaumont](https://github.com/rom1504), [Cade Gordon](http://cadegordon.io/), and [Vaishaal Shankar](http://vaishaal.com/).

## Citing

If you use this repository, please consider citing:

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
```