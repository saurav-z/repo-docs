# OpenCLIP: Open Source Implementation of CLIP for Image-Text Understanding

**OpenCLIP is an open-source implementation of OpenAI's CLIP, enabling powerful image-text understanding through contrastive learning.** [[Paper](https://arxiv.org/abs/2212.07143)] [[Citations](#citing)] [[Clip Colab](https://colab.research.google.com/github/mlfoundations/open_clip/blob/master/docs/Interacting_with_open_clip.ipynb)] [[Coca Colab](https://colab.research.google.com/github/mlfoundations/open_clip/blob/master/docs/Interacting_with_open_coca.ipynb)]

[![PyPI version](https://img.shields.io/pypi/v/open_clip_torch.svg)](https://pypi.python.org/pypi/open_clip_torch)

This repository provides a flexible and scalable framework for training and utilizing CLIP models, including those trained on massive datasets like LAION-2B and DataComp-1B. We offer a wide range of pre-trained models and comprehensive tools for various applications, including image retrieval, zero-shot classification, and fine-tuning.

**Key Features:**

*   **Pre-trained Models:** Access a diverse collection of pre-trained models trained on various datasets and compute budgets, including state-of-the-art models and those trained using the [LAION-400M](https://arxiv.org/abs/2111.02114), [LAION-2B](https://arxiv.org/abs/2210.08402), and [DataComp-1B](https://arxiv.org/abs/2304.14108) datasets.

*   **Training Framework:**  Easily train your own CLIP models with our comprehensive training scripts supporting distributed training, SLURM integration, gradient accumulation, and more.

*   **Flexible Usage:** Utilize the provided code to quickly encode images and text, calculate image/text similarities, and leverage pre-trained models.

*   **Fine-tuning Support:**  Use the provided models as a starting point for further fine-tuning tasks.

*   **CoCa Model Support:** Train and use CoCa (Contrastive Captioners) models.

*   **Extensive Documentation:** Explore our detailed documentation, including Colab notebooks, training guides, and model details.

**Model Performance Highlights:**

| Model              | Training Data | Resolution | ImageNet Zero-Shot Accuracy |
| ------------------ | ------------- | ---------- | --------------------------- |
| ConvNext-Base      | LAION-2B      | 256px      | 71.5%                       |
| ConvNext-Large     | LAION-2B      | 320px      | 76.9%                       |
| ViT-B-32-256       | DataComp-1B   | 256px      | 72.8%                       |
| ViT-L-14           | LAION-2B      | 224px      | 75.3%                       |
| ViT-H-14           | LAION-2B      | 224px      | 78.0%                       |
| ViT-bigG-14        | LAION-2B      | 224px      | 80.1%                       |

For more information, see our full collection of pretrained models [here](docs/PRETRAINED.md).

**Get Started:**

1.  **Installation:**
    ```bash
    pip install open_clip_torch
    ```

2.  **Quickstart Example:**
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

3.  **Explore Pretrained Models:**
    ```python
    import open_clip
    open_clip.list_pretrained()
    ```

**Further Resources:**

*   [[Clip Colab]](https://colab.research.google.com/github/mlfoundations/open_clip/blob/master/docs/Interacting_with_open_clip.ipynb)
*   [[Coca Colab]](https://colab.research.google.com/github/mlfoundations/open_clip/blob/master/docs/Interacting_with_open_coca.ipynb)
*   [Model cards](https://huggingface.co/models?library=open_clip) on the Hugging Face Hub.
*   Find out about the models we support (e.g. number of parameters, FLOPs) in [this table](docs/model_profile.csv).

**Contributing:**  We welcome contributions!  Please submit issues or email us with your requests or suggestions.

**Original repository**: [https://github.com/mlfoundations/open_clip](https://github.com/mlfoundations/open_clip)

**Acknowledgments:** [See original README for full acknowledgements.]

**Citations:**
```bibtex
[See original README for citations.]