# OpenCLIP: Open Source Implementation of CLIP

**OpenCLIP enables you to train, evaluate, and utilize powerful Contrastive Language-Image Pre-training (CLIP) models, fostering advancements in image understanding and multimodal AI.**

[Paper](https://arxiv.org/abs/2212.07143) | [Citations](#citing) | [CLIP Colab](https://colab.research.google.com/github/mlfoundations/open_clip/blob/master/docs/Interacting_with_open_clip.ipynb) | [CoCa Colab](https://colab.research.google.com/github/mlfoundations/open_clip/blob/master/docs/Interacting_with_open_coca.ipynb) | [PyPI](https://pypi.python.org/pypi/open_clip_torch) | [Hugging Face Models](https://huggingface.co/models?library=open_clip)

This repository provides a robust and open-source implementation of OpenAI's CLIP (Contrastive Language-Image Pre-training) framework. We offer a wide range of pre-trained models and comprehensive tools for training, evaluation, and fine-tuning, empowering researchers and developers to explore and build upon the latest advancements in image-text understanding.

**Key Features:**

*   **Pre-trained Models:** Access a diverse collection of pre-trained models, including those trained on LAION-400M, LAION-2B, and DataComp-1B datasets, with detailed zero-shot performance metrics.
*   **Reproducible Research:** Benefit from a codebase that facilitates reproducible research, as showcased in our paper on [reproducible scaling laws for contrastive language-image learning](https://arxiv.org/abs/2212.07143).
*   **Flexible Training:** Train your own CLIP models with customizable configurations, multi-GPU support, and integration with popular tools like SLURM and TensorBoard.
*   **Model Distillation:** Distill from a pre-trained model using `--distill-model` and `--distill-pretrained`
*   **Fine-tuning Capabilities:** Leverage the repository for zero-shot models with our complementary [WiSE-FT repository](https://github.com/mlfoundations/wise-ft) to preserve robustness under distribution shift.
*   **CoCa Support:** Train and utilize models using CoCa architectures.
*   **Int8 Support:** Experiment with int8 training and inference.
*   **Pushing Models to Hugging Face Hub:** Push models /w weights and config to the HF Hub.

**Model Highlights:**

| Model          | Training Data  | Resolution | ImageNet Zero-Shot Accuracy |
| -------------- | -------------- | ---------- | --------------------------- |
| ConvNext-Base  | LAION-2B       | 256px      | 71.5%                       |
| ConvNext-Large | LAION-2B       | 320px      | 76.9%                       |
| ViT-L-14       | DataComp-1B    | 224px      | 79.2%                       |
| ViT-bigG-14    | LAION-2B       | 224px      | 80.1%                       |

**Get Started:**

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

    For more detailed examples and advanced usage, see the [CLIP Colab](https://colab.research.google.com/github/mlfoundations/open_clip/blob/master/docs/Interacting_with_open_clip.ipynb).

3.  **Pretrained Models:**

    ```python
    import open_clip
    open_clip.list_pretrained()
    ```
    Find details about the models we support (e.g. number of parameters, FLOPs) in [this table](docs/model_profile.csv).

    *   To load models, run:

        ```python
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        ```
4.  **Fine-tuning:**

    For fine-tuning on downstream classification tasks, refer to the [WiSE-FT repository](https://github.com/mlfoundations/wise-ft).

**See the [original repo](https://github.com/mlfoundations/open_clip) for additional information.**

## Training

See the README for information about the following topics:
*   Installation
*   Sample single-process running code:
*   Multi-GPU and Beyond
*   Resuming from a checkpoint
*   Training CoCa
*   Generating text with CoCa
*   Fine Tuning CoCa
*   Training with pre-trained language models as text encoder
*   Loss Curves
*   Logging

## Evaluation / Zero-Shot

We recommend https://github.com/LAION-AI/CLIP_benchmark#how-to-use for systematic evaluation on 40 datasets.

See the README for information about the following topics:
*   Evaluating local checkpoint
*   Evaluating hosted pretrained checkpoint on ImageNet zero-shot prediction
*   Model distillation
*   Gradient accumulation
*   Int8 Support
*   Support for remote loading/training
*   Pushing Models to Hugging Face Hub

## Acknowledgments

We are grateful to the contributors and the community for their support. See the README for the full list of the team and acknowledgments.

## Citing

If you find this project useful, please cite the following:

```bibtex
# Add citation bibtex entries here
```

[![DOI](https://zenodo.org/badge/390536799.svg)](https://zenodo.org/badge/latestdoi/390536799)