# OpenCLIP: Open Source Implementation of CLIP for Image-Text Learning

**OpenCLIP** is an open-source implementation of OpenAI's Contrastive Language-Image Pre-training (CLIP), enabling you to train and utilize state-of-the-art models for image-text understanding. [[Paper]](https://arxiv.org/abs/2212.07143) | [[Citations]](#citing) | [[Clip Colab]](https://colab.research.google.com/github/mlfoundations/open_clip/blob/master/docs/Interacting_with_open_clip.ipynb) | [[Coca Colab]](https://colab.research.google.com/github/mlfoundations/open_clip/blob/master/docs/Interacting_with_open_coca.ipynb) [![PyPI version](https://img.shields.io/pypi/v/open_clip_torch.svg)](https://pypi.python.org/pypi/open_clip_torch)

## Key Features

*   **Pre-trained Models:** Access a wide variety of pre-trained models, including those trained on LAION-2B, LAION-400M, DataComp-1B and others, achieving high zero-shot accuracy on ImageNet and other datasets.
*   **Training Flexibility:** Train your own CLIP models with customizable parameters and support for multi-GPU and multi-node training.
*   **Easy to Use:** Simple API for loading models, tokenizing text, and encoding images and text.
*   **Fine-tuning Support:** Fine-tune pre-trained models on downstream classification tasks using our WiSE-FT repository ([https://github.com/mlfoundations/wise-ft](https://github.com/mlfoundations/wise-ft)).
*   **Reproducibility:** Detailed training configurations, model cards and results in the paper [reproducible scaling laws for contrastive language-image learning](https://arxiv.org/abs/2212.07143)

## Pre-trained Models & Performance

OpenCLIP offers a diverse set of pre-trained models, excelling in zero-shot performance across multiple datasets. Below are some examples from the paper showcasing state-of-the-art results. Further details are available [here](docs/PRETRAINED.md).

| Model                | Training Data | Resolution | # of Samples | ImageNet Zero-Shot Acc. |
| -------------------- | ------------- | ---------- | ------------- | ------------------------ |
| ConvNext-Base        | LAION-2B      | 256px      | 13B           | 71.5%                    |
| ConvNext-Large       | LAION-2B      | 320px      | 29B           | 76.9%                    |
| ConvNext-XXLarge     | LAION-2B      | 256px      | 34B           | 79.5%                    |
| ViT-B-32-256         | DataComp-1B   | 256px      | 34B           | 72.8%                    |
| ViT-B-16             | DataComp-1B   | 224px      | 13B           | 73.5%                    |
| ViT-L-14             | LAION-2B      | 224px      | 32B           | 75.3%                    |
| ViT-H-14             | LAION-2B      | 224px      | 32B           | 78.0%                    |
| ViT-L-14             | DataComp-1B   | 224px      | 13B           | 79.2%                    |
| ViT-bigG-14          | LAION-2B      | 224px      | 34B           | 80.1%                    |
| ViT-L-14-quickgelu [(Original CLIP)](https://arxiv.org/abs/2103.00020) | WIT | 224px | 13B | 75.5% |
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

For additional details, explore the models available on the Hugging Face Hub under the OpenCLIP library tag: [https://huggingface.co/models?library=open_clip](https://huggingface.co/models?library=open_clip).

## Quickstart

Install the necessary packages:

```bash
pip install open_clip_torch
```

Then use the code below to implement the CLIP model:

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

## Training and Usage

The repository provides comprehensive documentation, Colab notebooks, and example scripts for training and utilizing CLIP models, including options for multi-GPU and distributed training. More details are in the [Usage](#usage) section.

## Contributing

We welcome contributions!  Feel free to submit issues or contact us with suggestions.

## Acknowledgments

*   We gratefully acknowledge the Gauss Centre for Supercomputing e.V. (www.gauss-centre.eu) for funding this part of work.
*   This code is based on the original OpenAI CLIP implementation.

## The Team

*   Led by [Ross Wightman](https://rwightman.com/), [Romain Beaumont](https://github.com/rom1504), [Cade Gordon](http://cadegordon.io/), and [Vaishaal Shankar](http://vaishaal.com/).

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