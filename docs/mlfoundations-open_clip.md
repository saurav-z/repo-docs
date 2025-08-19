# OpenCLIP: Open Source Implementation of CLIP for Image-Text Understanding

**OpenCLIP** offers a robust and versatile open-source implementation of OpenAI's CLIP (Contrastive Language-Image Pre-training), enabling state-of-the-art performance in image-text understanding.  Explore a comprehensive suite of pre-trained models, training resources, and fine-tuning capabilities to unlock new possibilities in visual AI.  [Explore the original repository](https://github.com/mlfoundations/open_clip).

[![PyPI](https://img.shields.io/pypi/v/open_clip_torch.svg)](https://pypi.python.org/pypi/open_clip_torch)
[![Paper](https://img.shields.io/badge/paper-arXiv-blue.svg)](https://arxiv.org/abs/2212.07143)  [![Citations](https://img.shields.io/badge/citations-see%20below-brightgreen.svg)](#citing)  [![Colab](https://img.shields.io/badge/Colab-Tutorials-blue.svg)](https://colab.research.google.com/github/mlfoundations/open_clip/blob/master/docs/Interacting_with_open_clip.ipynb)

## Key Features:

*   **Pre-trained Models:** Access a wide variety of pre-trained models trained on massive datasets like LAION-2B, LAION-400M, and DataComp-1B, achieving impressive zero-shot ImageNet accuracy.
*   **Training Code:**  Train your own CLIP models with flexible training scripts that support multi-GPU, SLURM clusters, and various data sources.
*   **Fine-tuning Capabilities:** Leverage the repository to fine-tune your models on downstream classification tasks, and find information for robustness on the [WiSE-FT repository](https://github.com/mlfoundations/wise-ft).
*   **Comprehensive Documentation:**  Benefit from detailed documentation, including usage examples, model loading guides, and training instructions, along with Colab tutorials.
*   **Efficient Implementations:**  Utilize features like gradient accumulation, Int8 support, and remote loading/training for optimized performance.
*   **Model Distillation Support:** Fine-tune your models while improving the accuracy by using the Model Distillation capability.

## Quickstart

**Installation:**

```bash
pip install open_clip_torch
```

**Usage Example:**

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

print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]
```

## Pretrained Models & Performance

OpenCLIP provides a range of pre-trained models, offering diverse architectures and training datasets, and are accessible via `open_clip.create_model_and_transforms`. Details about available models are available [here](docs/PRETRAINED.md).

| Model             | Training Data | Resolution | ImageNet Zero-Shot Acc. |
| :---------------- | :------------ | :--------- | :---------------------- |
| ConvNext-Base     | LAION-2B      | 256px      | 71.5%                   |
| ConvNext-Large    | LAION-2B      | 320px      | 76.9%                   |
| ConvNext-XXLarge  | LAION-2B      | 256px      | 79.5%                   |
| ViT-B-32-256      | DataComp-1B   | 256px      | 72.8%                   |
| ViT-B-16          | DataComp-1B   | 224px      | 73.5%                   |
| ViT-L-14          | LAION-2B      | 224px      | 75.3%                   |
| ViT-H-14          | LAION-2B      | 224px      | 78.0%                   |
| ViT-L-14          | DataComp-1B   | 224px      | 79.2%                   |
| ViT-bigG-14       | LAION-2B      | 224px      | 80.1%                   |

*(See the original README for more detailed model comparisons, which is continually updated.)*

## Training and Fine-tuning

OpenCLIP provides the tools to train new CLIP models, offers guidance for fine-tuning on classification tasks, and more. Check the original repository for detailed information on data preparation, training scripts, and multi-GPU/SLURM setups, and fine-tuning methods.

## Model Cards

Model cards with additional model specific details can be found on the Hugging Face Hub under the OpenCLIP library tag: https://huggingface.co/models?library=open_clip.

## Acknowledgments

(As per the original README)

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

(And the other citations from the original README)

[![DOI](https://zenodo.org/badge/390536799.svg)](https://zenodo.org/badge/latestdoi/390536799)