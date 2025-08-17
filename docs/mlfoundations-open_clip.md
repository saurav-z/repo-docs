# OpenCLIP: Open Source Implementation of CLIP for Image-Text Understanding

**OpenCLIP empowers researchers and developers to explore and build upon the groundbreaking capabilities of OpenAI's CLIP (Contrastive Language-Image Pre-training) models.**  Explore the original repository on [GitHub](https://github.com/mlfoundations/open_clip) for source code and further details.

*   **Pre-trained Models:** Access a wide range of pre-trained models trained on diverse datasets, including LAION-400M, LAION-2B, and DataComp-1B, offering varying performance and resource requirements.
*   **Reproducible Scaling Laws:**  Leverage models and findings detailed in the paper, "Reproducible scaling laws for contrastive language-image learning," for insights into model scaling and performance.
*   **Comprehensive Training & Evaluation:** Train and evaluate CLIP models, including support for distributed training, SLURM clusters, and advanced features like gradient accumulation and Int8 support.
*   **Integration with CLIP-Benchmark:** Seamlessly evaluate models across diverse datasets using integration with the CLIP-benchmark framework.
*   **Model Distillation Support:** Efficiently distill knowledge from pre-trained models for improved performance and faster training.

## Key Features

*   **State-of-the-Art Models:** Includes models that rival or exceed the performance of the original CLIP models and other open-source alternatives, such as ViT-bigG-14 and SigLIP variants.
*   **Flexible Usage:** Easily load and utilize pre-trained models, with clear code examples for image and text encoding.  Supports both PyTorch's native GELU and QuickGELU (for OpenAI compatibility).
*   **Fine-tuning Support:**  Offers tools and guidance for fine-tuning models on downstream tasks.
*   **Multi-GPU and Distributed Training:**  Highly optimized for distributed training across multiple GPUs, supporting single-node, multi-node, and SLURM environments.
*   **Data Management:** Supports webdataset for large-scale datasets and provides guidance on data preparation.
*   **Model Distillation and Gradient Accumulation:** Optimize training with model distillation and gradient accumulation techniques.
*   **Int8 Support:** Utilize Int8 training and inference for potential speedups.
*   **Hugging Face Hub Integration:**  Push models and configurations directly to the Hugging Face Hub for easy sharing and deployment.
*   **CoCa support:** support for training and inference using [CoCa](https://arxiv.org/abs/2205.01917) models.
*   **Text Encoder Flexibility:** Supports various text encoder architectures, including those from Hugging Face Transformers.

## Quick Start

1.  **Installation:**

    ```bash
    pip install open_clip_torch
    ```

2.  **Usage Example:**

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

## Pretrained Models

A wide array of pre-trained models is available. Use the following code snippet to list available models. More details about the pretrained models can be found [here](docs/PRETRAINED.md).

```python
import open_clip
open_clip.list_pretrained()
```

## Fine-tuning

For fine-tuning a *trained* zero-shot model on a downstream classification task such as ImageNet, please see [WiSE-FT](https://github.com/mlfoundations/wise-ft).

## Training

Detailed instructions for training, including data preparation, multi-GPU setups, and SLURM configuration, are available in the original README.

## Evaluation

For evaluation, use [CLIP_benchmark](https://github.com/LAION-AI/CLIP_benchmark#how-to-use) for comprehensive evaluation on 40 datasets.

## Acknowledgements

The project acknowledges the Gauss Centre for Supercomputing e.V. (www.gauss-centre.eu) and the John von Neumann Institute for Computing (NIC).  The original repository was developed by researchers from UW, Google, Stanford, Amazon, Columbia, and Berkeley.  Special thanks to Jong Wook Kim and Alec Radford.

## Citing

If you find this repository useful, please cite the relevant publications. Citation information is available in the original README.

## The Team

Current development is led by Ross Wightman, Romain Beaumont, Cade Gordon, and Vaishaal Shankar.