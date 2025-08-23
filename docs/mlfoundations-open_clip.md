# OpenCLIP: Open Source Implementation of CLIP

**Unlock the power of image-text understanding with OpenCLIP, an open-source implementation of OpenAI's CLIP, empowering you to train and utilize cutting-edge models for various applications.** ([Original Repo](https://github.com/mlfoundations/open_clip))

[![PyPI version](https://img.shields.io/pypi/v/open_clip_torch.svg)](https://pypi.org/project/open_clip_torch/) | [[Paper]](https://arxiv.org/abs/2212.07143) [[Citations]](#citing) [[Clip Colab]](https://colab.research.google.com/github/mlfoundations/open_clip/blob/master/docs/Interacting_with_open_clip.ipynb) [[Coca Colab]](https://colab.research.google.com/github/mlfoundations/open_clip/blob/master/docs/Interacting_with_open_coca.ipynb)

OpenCLIP provides a robust framework for training and utilizing CLIP models, offering a range of pre-trained models and flexible training options. This project enables researchers and developers to explore and apply state-of-the-art image-text understanding techniques.

**Key Features:**

*   **Pre-trained Models:** Access a wide array of pre-trained CLIP models, including those trained on LAION-2B, LAION-400M, and DataComp-1B datasets, with performance details in the paper [reproducible scaling laws for contrastive language-image learning](https://arxiv.org/abs/2212.07143).
*   **Flexible Training:** Train your own CLIP models with customizable parameters using a highly configurable training pipeline.
*   **Easy-to-Use API:** Utilize a simple and intuitive API for model loading, image and text encoding, and zero-shot classification.
*   **CoCa Support:** Includes support for training and utilizing CoCa models for multimodal understanding.
*   **Hugging Face Integration:** Seamlessly push and pull models from the Hugging Face Hub for easy sharing and collaboration.
*   **Int8 Support:** Beta support for int8 training and inference for faster training with reduced memory requirements.
*   **Multi-GPU and Distributed Training:** Supports multi-GPU and distributed training, including SLURM, for efficient large-scale model training.
*   **Support for Remote Loading/Training:** It is always possible to resume directly from a remote file, e.g., a file in an s3 bucket.

**Performance Highlights:**

| Model            | Training Data | Resolution | ImageNet Zero-Shot Accuracy |
| ---------------- | ------------- | ---------- | --------------------------- |
| ConvNext-Base    | LAION-2B      | 256px      | 71.5%                       |
| ConvNext-Large   | LAION-2B      | 320px      | 76.9%                       |
| ConvNext-XXLarge | LAION-2B      | 256px      | 79.5%                       |
| ViT-B-32-256     | DataComp-1B   | 256px      | 72.8%                       |
| ViT-B-16         | DataComp-1B   | 224px      | 73.5%                       |
| ViT-L-14         | LAION-2B      | 224px      | 75.3%                       |
| ...              | ...           | ...        | ...                         |

More details about our full collection of pretrained models can be found [here](docs/PRETRAINED.md), and zero-shot results for 38 datasets [here](docs/openclip_results.csv).

## Installation

```bash
pip install open_clip_torch
```

## Usage

```python
import torch
from PIL import Image
import open_clip

# Load a pre-trained model
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
model.eval()  # Set model to evaluation mode
tokenizer = open_clip.get_tokenizer('ViT-B-32')

# Prepare input data
image = preprocess(Image.open("docs/CLIP.png")).unsqueeze(0)
text = tokenizer(["a diagram", "a dog", "a cat"])

# Perform inference
with torch.no_grad(), torch.autocast("cuda"):
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)  # Expected output: [[1., 0., 0.]]
```

## Training

For comprehensive training instructions, please refer to the original [README](https://github.com/mlfoundations/open_clip) or the more detailed guidance at the following [links](https://github.com/mlfoundations/open_clip#training-clip).

## Fine-tuning

For fine-tuning on downstream tasks, see [WiSE-FT](https://github.com/mlfoundations/wise-ft).

## Data
To download datasets as webdataset, we recommend [img2dataset](https://github.com/rom1504/img2dataset).
See the original [README](https://github.com/mlfoundations/open_clip#data) for data specifics.

## Model Distillation

You can distill from a pre-trained model by using `--distill-model` and `--distill-pretrained` to specify the model you'd like to distill from.
For instance, to distill from OpenAI ViT-L/14 use `--distill-model ViT-L-14 --distill-pretrained openai`.

## Pushing Models to Hugging Face Hub

The module `open_clip.push_to_hf_hub` includes helpers for pushing models /w weights and config to the HF Hub.

The tool can be run from command line, ex:
`python -m open_clip.push_to_hf_hub --model convnext_large_d_320 --pretrained /train/checkpoints/epoch_12.pt --repo-id laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft`

## Acknowledgments

The team gratefully acknowledges the Gauss Centre for Supercomputing e.V. (www.gauss-centre.eu) for funding this part of work by providing computing time through the John von Neumann Institute for Computing (NIC) on the GCS Supercomputer JUWELS Booster at Jülich Supercomputing Centre (JSC).

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