# OpenCLIP: Open Source Implementation of CLIP for Image-Text Learning

**OpenCLIP** is a powerful open-source implementation of OpenAI's CLIP (Contrastive Language-Image Pre-training), enabling you to train and utilize models that excel at bridging the gap between images and text.  [Explore the original repository](https://github.com/mlfoundations/open_clip).

*   **Key Features:**
    *   **Pre-trained Models:** Access a wide range of pre-trained models, including those trained on massive datasets like LAION-2B and DataComp-1B, for immediate use.
    *   **Reproducible Research:**  Based on the paper "Reproducible scaling laws for contrastive language-image learning," OpenCLIP provides a robust framework for studying and scaling CLIP models.
    *   **Flexible Training:**  Supports training on diverse datasets and offers various training configurations, including multi-GPU and SLURM support.
    *   **Easy Integration:** Simple Python API for model loading, image and text encoding, and downstream tasks.
    *   **Fine-tuning Capabilities:**  Fine-tune pre-trained models on classification tasks using our companion repository [WiSE-FT](https://github.com/mlfoundations/wise-ft).
    *   **CoCa Support:** Train and utilize CoCa (Contrastive Captioned) models.
    *   **Int8 Support:** Beta support for int8 training and inference.
    *   **Remote Training & Loading:** Supports training and loading from remote filesystems.

*   **Key Benefits:**
    *   **State-of-the-Art Performance:** Leverage models that achieve cutting-edge zero-shot accuracy on various image recognition tasks.
    *   **Community-Driven:** Benefit from an active community and contribute to open-source research in image-text learning.
    *   **Scalability:** Train models on a wide range of datasets using the optimized training pipeline.

## Quick Start

1.  **Installation:**

```bash
pip install open_clip_torch
```

2.  **Basic Usage:**

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

## Pre-trained Models

OpenCLIP offers a convenient interface for accessing a variety of pre-trained models. Explore the available models and their performance characteristics in [docs/PRETRAINED.md](docs/PRETRAINED.md).

*   **Example:** List available pre-trained models:

```python
import open_clip
open_clip.list_pretrained()
```

## Training

Detailed instructions for training your own CLIP models are available in the [Training](#training) section of the original README.

## Evaluation / Zero-Shot

For systematic evaluation, we recommend the [CLIP_benchmark](https://github.com/LAION-AI/CLIP_benchmark#how-to-use) for assessing model performance across various datasets.

## Acknowledgments & Citation

This project is a collaborative effort, supported by various organizations.  If you use OpenCLIP, please cite the relevant papers, detailed in the [Citing](#citing) section of the original README.

---

*For more detailed information, including training procedures, evaluation, and model details, please refer to the original [OpenCLIP GitHub repository](https://github.com/mlfoundations/open_clip).*