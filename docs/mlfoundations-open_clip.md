# OpenCLIP: Open Source Implementation of CLIP for Advanced Image-Text Understanding

**OpenCLIP** empowers you to explore state-of-the-art image-text models by providing an open-source implementation of OpenAI's CLIP (Contrastive Language-Image Pre-training), enabling you to train, evaluate, and deploy powerful vision-language models.  [Visit the Original Repo](https://github.com/mlfoundations/open_clip)

*   **Pre-trained Models:** Access a wide range of pre-trained CLIP models trained on diverse datasets like LAION-400M, LAION-2B, and DataComp-1B.
*   **Reproducible Research:**  Leverage our codebase to reproduce and build upon cutting-edge research in contrastive language-image learning.
*   **Flexible Training:** Easily fine-tune models on downstream tasks using our robust training scripts, supporting multi-GPU and distributed training configurations.
*   **Model Loading & Usage:** Simplified API for loading models, tokenizers, and performing image-text encoding.
*   **CoCa Integration:** Train and utilize CoCa (Contrastive Captions) models for advanced image captioning tasks.
*   **Hugging Face Hub Integration:** Effortlessly push and pull your models and configurations to and from the Hugging Face Hub.

## Key Features

*   **Model Variety:** Supports a wide variety of architectures, including ConvNext, ViT, and more, offering flexibility in model selection.
*   **Large-Scale Training:** Optimized for training on large datasets and with distributed computing resources.
*   **Zero-Shot Performance:** Achieve impressive zero-shot performance on various benchmarks.
*   **Fine-tuning Support:**  Detailed instructions and resources for fine-tuning models on specific tasks (see [WiSE-FT](https://github.com/mlfoundations/wise-ft)).
*   **Easy Installation:**  Simple `pip install` command for quick setup.
*   **Int8 Support:** Experimental support for int8 training and inference, for potentially faster training times.
*   **Remote Training:** Support for loading data from remote filesystems and backing up to S3 during training.

## Quick Start

### Installation
```bash
pip install open_clip_torch
```

### Example Usage
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

print("Label probs:", text_probs) # prints: [[1., 0., 0.]]
```
*See the [CLIP Colab](https://colab.research.google.com/github/mlfoundations/open_clip/blob/master/docs/Interacting_with_open_clip.ipynb) and the [CoCa Colab](https://colab.research.google.com/github/mlfoundations/open_clip/blob/master/docs/Interacting_with_open_coca.ipynb) for interactive examples.*

## Pre-trained Models

Explore a comprehensive collection of pre-trained models to quickly get started with image-text tasks. Detailed information on available models and their performance can be found [here](docs/PRETRAINED.md).

*   Use `open_clip.list_pretrained()` to list available pre-trained models.

## Training

To begin training your own models, please refer to the detailed instructions in the original repository's README.  [Visit the original repository for full training instructions.](https://github.com/mlfoundations/open_clip)

## Fine-tuning

To fine-tune a trained zero-shot model on downstream classification tasks (e.g., ImageNet), please see our separate repository: [WiSE-FT](https://github.com/mlfoundations/wise-ft).

## Acknowledgements
* The Gauss Centre for Supercomputing e.V. (www.gauss-centre.eu) and the John von Neumann Institute for Computing (NIC) for computing resources.

## Citation
Please consider citing the following when using OpenCLIP:

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
... (and other citations from the original readme)