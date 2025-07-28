# OpenCLIP: Open Source Implementation of CLIP

**Unlock the power of visual understanding with OpenCLIP, an open-source implementation of OpenAI's CLIP, enabling groundbreaking image-text learning.**

[[Paper]](https://arxiv.org/abs/2212.07143) [[Citations]](#citing) [[Clip Colab]](https://colab.research.google.com/github/mlfoundations/open_clip/blob/master/docs/Interacting_with_open_clip.ipynb) [[Coca Colab]](https://colab.research.google.com/github/mlfoundations/open_clip/blob/master/docs/Interacting_with_open_coca.ipynb)
[![pypi](https://img.shields.io/pypi/v/open_clip_torch.svg)](https://pypi.python.org/pypi/open_clip_torch)

OpenCLIP provides a robust and accessible framework for contrastive language-image pre-training (CLIP).  This repository offers:

*   **Pre-trained Models:** Access a diverse collection of models trained on various datasets, including LAION-400M, LAION-2B, and DataComp-1B.
*   **Reproducible Research:**  Leverage models and scaling properties studied in detail in the paper "Reproducible scaling laws for contrastive language-image learning" ([https://arxiv.org/abs/2212.07143](https://arxiv.org/abs/2212.07143)).
*   **Easy-to-Use API:**  A simple interface for creating and utilizing both pre-trained and untrained models.
*   **Fine-tuning Support:**  Utilize the code for fine-tuning trained zero-shot models on downstream classification tasks.
*   **Multi-GPU Training:**  Efficient distributed training solutions with native support for SLURM clusters.
*   **Model Distillation:**  Distill knowledge from pre-trained models to improve performance and efficiency.

Here's a glimpse of some top-performing models:

| Model             | Training Data | Resolution | # of Samples Seen | ImageNet Zero-shot Acc. |
|-------------------|---------------|------------|--------------------|-------------------------|
| ConvNext-Base     | LAION-2B      | 256px      | 13B                | 71.5%                   |
| ConvNext-Large    | LAION-2B      | 320px      | 29B                | 76.9%                   |
| ConvNext-XXLarge  | LAION-2B      | 256px      | 34B                | 79.5%                   |
| ViT-B-32-256      | DataComp-1B   | 256px      | 34B                | 72.8%                   |
| ViT-B-16          | DataComp-1B   | 224px      | 13B                | 73.5%                   |
| ViT-L-14          | LAION-2B      | 224px      | 32B                | 75.3%                   |
| ViT-H-14          | LAION-2B      | 224px      | 32B                | 78.0%                   |
| ViT-L-14          | DataComp-1B   | 224px      | 13B                | 79.2%                   |
| ViT-bigG-14       | LAION-2B      | 224px      | 34B                | 80.1%                   |
| ViT-L-14-quickgelu (Original CLIP) | WIT | 224px | 13B | 75.5% |
| ViT-SO400M-14-SigLIP [(SigLIP)](https://arxiv.org/abs/2303.15343) | WebLI | 224px | 45B | 82.0% |
| ViT-L-14 [(DFN)](https://arxiv.org/abs/2309.17425) | DFN-2B | 224px | 39B | 82.2% |
| ViT-L-16-256 [(SigLIP2)](https://arxiv.org/abs/2502.14786) |  WebLI (multi-lang) | 256px | 40B | 82.5% |
| ViT-SO400M-14-SigLIP-384 [(SigLIP)](https://arxiv.org/abs/2303.15343) |  WebLI | 384px | 45B | 83.1% |
| ViT-H-14-quickgelu [(DFN)](https://arxiv.org/abs/2309.17425) | DFN-5B | 224px | 39B | 83.4% |
| PE-Core-L-14-336 [(PE)](https://arxiv.org/abs/2504.13181) | MetaCLIP-5.4B | 336px | 58B | 83.5% |
| ViT-SO400M-16-SigLIP2-384 [(SigLIP2)](https://arxiv.org/abs/2502.14786) |  WebLI (multi-lang) | 384px | 40B | 84.1% |
| ViT-H-14-378-quickgelu [(DFN)](https://arxiv.org/abs/2309.17425) | DFN-5B | 378px | 44B | 84.4% |
| ViT-gopt-16-SigLIP2-384 [(SigLIP2)](https://arxiv.org/abs/2502.14786) | WebLI (multi-lang) | 384px | 40B | 85.0% |
| PE-Core-bigG-14-448 [(PE)](https://arxiv.org/abs/2504.13181) | MetaCLIP-5.4B | 448px | 86B | 85.4% |

Find more details about available models on the [Hugging Face Hub](https://huggingface.co/models?library=open_clip).

**Get started today and explore the potential of OpenCLIP!**

## Key Features

*   **Pre-trained Models:** Access a variety of pre-trained CLIP models.
*   **Model Training:** Train your own CLIP models.
*   **Ease of Use:** Simple API for loading and using models.
*   **Multi-GPU and Distributed Training:** Scalable training for large models.
*   **Fine-tuning:** Fine-tune CLIP models on downstream tasks.
*   **CoCa Support:**  Training and inference support for CoCa models.
*   **Int8 Support:** Experimental support for int8 training and inference.

## Quick Start

```bash
pip install open_clip_torch
```

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

For more in-depth instructions, see the [[Clip Colab]](https://colab.research.google.com/github/mlfoundations/open_clip/blob/master/docs/Interacting_with_open_clip.ipynb).

## Usage

*   **Pretrained models:**  Use `open_clip.list_pretrained()` to see available models.  More details on pretrained models [here](docs/PRETRAINED.md).
*   **Loading models:** Use `open_clip.create_model_and_transforms`. The `pretrained` argument also accepts local paths and checkpoints from Hugging Face (e.g., `/path/to/open_clip_pytorch_model.bin`).
*   **Fine-tuning:**  Refer to [WiSE-FT](https://github.com/mlfoundations/wise-ft) for fine-tuning on classification tasks.
*   **CoCa:** Training and inference examples are provided.  See the [[Coca Colab]](https://colab.research.google.com/github/mlfoundations/open_clip/blob/master/docs/Interacting_with_open_coca.ipynb).
*   **Training:**  Detailed training instructions are available, including single-process, multi-GPU, SLURM, and checkpoint resuming.

## Data

*   **Data Preparation:** Recommended use of [img2dataset](https://github.com/rom1504/img2dataset) for dataset creation.
*   **WebDataset Support:** Recommended for large-scale datasets with webdataset.
*   **YFCC:** Information provided for the YFCC dataset.

## Training CLIP

*   **Installation:** Instructions for creating a virtual environment and installing dependencies.
*   **Sample single-process running code:** Example code provided.
*   **Multi-GPU and Beyond:**  Guidance on distributed training, including the use of SLURM.
*   **Training Parameters:**  Details on epochs, patch dropout, multiple data sources, single-node, and multi-node.
*   **Resuming from a checkpoint:** Command provided.
*   **Model distillation:** Option to distill from a pre-trained model.
*   **Gradient accumulation:** Instructions on simulating larger batches.
*   **Int8 Support:**  Information about int8 training and inference.
*   **Remote loading/training:** Instructions for loading and training from remote locations.
*   **Pushing Models to Hugging Face Hub:** Instructions for pushing models to the HF Hub.

## Evaluation / Zero-Shot

Evaluation recommendations and examples. We recommend using [CLIP_benchmark](https://github.com/LAION-AI/CLIP_benchmark#how-to-use) for systematic evaluation on 40 datasets.

## Acknowledgments

Detailed acknowledgments for funding and contributions to the project.

## The Team

A list of the core contributors to the project.

## Citing

If you found this repository useful, please consider citing the project.
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