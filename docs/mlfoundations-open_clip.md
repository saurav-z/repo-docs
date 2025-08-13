# OpenCLIP: Open Source CLIP Implementation for Image-Text Learning

**OpenCLIP** offers an open-source implementation of OpenAI's [CLIP](https://arxiv.org/abs/2103.00020) model, enabling you to train and utilize powerful image-text models.  

[Paper](https://arxiv.org/abs/2212.07143) | [Citations](#citing) | [CLIP Colab](https://colab.research.google.com/github/mlfoundations/open_clip/blob/master/docs/Interacting_with_open_clip.ipynb) | [CoCa Colab](https://colab.research.google.com/github/mlfoundations/open_clip/blob/master/docs/Interacting_with_open_coca.ipynb) | [PyPI](https://pypi.org/project/open_clip_torch/)

OpenCLIP provides a versatile toolkit for researchers and developers interested in:

*   **Reproducing and Extending CLIP:** Fine-tune and experiment with different architectures, training datasets, and hyperparameters.
*   **Zero-Shot Image Classification:** Leverage pre-trained models for immediate image classification tasks without task-specific training.
*   **Image-Text Understanding:**  Explore the connection between images and text using the power of contrastive learning.

## Key Features

*   **Pre-trained Models:** Access a wide range of pre-trained CLIP models trained on diverse datasets like LAION-400M, LAION-2B, and DataComp-1B, including those studied in the paper [reproducible scaling laws for contrastive language-image learning](https://arxiv.org/abs/2212.07143).
*   **Model Versatility**: Includes pre-trained models using both QuickGELU and native torch.nn.GELU activations.
*   **Flexible Training:** Train CLIP models from scratch or fine-tune existing models on your own datasets.
*   **Efficient Training:** Support for multi-GPU training, including solutions for large datasets and distributed training across multiple nodes.
*   **Easy-to-Use API:** Simple interface for loading models, tokenizing text, and encoding images and text.
*   **Integration with CLIP Benchmark:**  Easily evaluate your models using [CLIP_benchmark](https://github.com/LAION-AI/CLIP_benchmark#how-to-use)
*   **CoCa Support:** Includes CoCa models for multimodal understanding, fine-tuning and text generation capabilities.

## Model Performance Highlights

The table below showcases the performance of some of our best models. For a complete list, see [here](docs/PRETRAINED.md), and [here](docs/openclip_results.csv) for zero-shot results for 38 datasets:

| Model             | Training Data | Resolution | # of Samples Seen | ImageNet Zero-Shot Accuracy |
| ----------------- | ------------- | ---------- | ----------------- | --------------------------- |
| ConvNext-Base     | LAION-2B      | 256px      | 13B               | 71.5%                       |
| ConvNext-Large    | LAION-2B      | 320px      | 29B               | 76.9%                       |
| ConvNext-XXLarge  | LAION-2B      | 256px      | 34B               | 79.5%                       |
| ViT-B-32-256      | DataComp-1B   | 256px      | 34B               | 72.8%                       |
| ViT-B-16          | DataComp-1B   | 224px      | 13B               | 73.5%                       |
| ViT-L-14          | LAION-2B      | 224px      | 32B               | 75.3%                       |
| ViT-H-14          | LAION-2B      | 224px      | 32B               | 78.0%                       |
| ViT-L-14          | DataComp-1B   | 224px      | 13B               | 79.2%                       |
| ViT-bigG-14       | LAION-2B      | 224px      | 34B               | 80.1%                       |
| ...               | ...           | ...        | ...               | ...                         |

*For additional models and benchmark results please see the links included at the top.*

Model cards with additional details can be found on the Hugging Face Hub: [https://huggingface.co/models?library=open_clip](https://huggingface.co/models?library=open_clip)

## Installation

Install OpenCLIP with:

```bash
pip install open_clip_torch
```

For training and development:

```bash
pip install 'open_clip_torch[training]'
# or
make install # and install PyTorch as per https://pytorch.org/get-started/locally/
make install-training # for training dependencies
make install-test
make test
```

## Usage

Here's a quick example of how to use OpenCLIP:

```python
import torch
from PIL import Image
import open_clip

# Load model and preprocessor
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
model.eval() # Set model to evaluation mode
tokenizer = open_clip.get_tokenizer('ViT-B-32')

# Prepare input
image = preprocess(Image.open("docs/CLIP.png")).unsqueeze(0)
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

## Pretrained Models

```python
>>> import open_clip
>>> open_clip.list_pretrained()
```
More information on the models can be found [here](docs/PRETRAINED.md), with the number of parameters and FLOPs available in [this table](docs/model_profile.csv).

### Loading Models

Models can be loaded using `open_clip.create_model_and_transforms`, as shown above. The `model_name` and `pretrained` keys are compatible with the output of `open_clip.list_pretrained()`. The `pretrained` argument also accepts local paths (e.g., `/path/to/my/b32.pt`) and Hugging Face Hub checkpoints.

### Fine-tuning

For fine-tuning a zero-shot model on downstream classification tasks, please refer to [WiSE-FT](https://github.com/mlfoundations/wise-ft).

## Training

### Training with your own dataset

To train your own models, follow these steps and see the provided example script.

1.  **Install the library** with `pip install 'open_clip_torch[training]'`.
2.  **Prepare your data** in a suitable format (CSV, webdataset, etc.).
3.  **Run the training script** using the command-line arguments to configure your model, dataset, and training parameters.
4.  **Monitor Training.**

For a detailed guide to training, consult the "Training CLIP" section in the original README.

### Training CoCa

Training [CoCa](https://arxiv.org/abs/2205.01917) models is enabled through specifying a CoCa config using the ```--model``` parameter of the training script.

For a complete guide to the training process, refer to the sections "Training CoCa" and "Fine Tuning CoCa".

## Evaluation/Zero-Shot

For detailed guidance on evaluating your models and conducting zero-shot classification, consult the "Evaluation / Zero-Shot" section in the original README.
We recommend the use of [CLIP_benchmark](https://github.com/LAION-AI/CLIP_benchmark#how-to-use) for systematic evaluation on 40 datasets.

## Acknowledgments

The original authors gratefully acknowledge the Gauss Centre for Supercomputing e.V. (www.gauss-centre.eu) for their support.

## Team

The current development of this repository is led by [Ross Wightman](https://rwightman.com/), [Romain Beaumont](https://github.com/rom1504), [Cade Gordon](http://cadegordon.io/), and [Vaishaal Shankar](http://vaishaal.com/).  The original version of this repository is from a group of researchers at UW, Google, Stanford, Amazon, Columbia, and Berkeley.

## Citing

If you find this repository useful, please consider citing the relevant papers:

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

**[Visit the OpenCLIP repository on GitHub](https://github.com/mlfoundations/open_clip) for the latest updates, code, and contributions.**