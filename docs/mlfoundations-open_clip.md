# OpenCLIP: Open Source Implementation of CLIP for Image-Text Learning

OpenCLIP is a powerful open-source implementation of OpenAI's CLIP (Contrastive Language-Image Pre-training), offering a flexible framework for training and utilizing state-of-the-art image-text models.  Explore the code and contribute on [GitHub](https://github.com/mlfoundations/open_clip).

**Key Features:**

*   **Pre-trained Models:**  Access a wide range of pre-trained models, including those trained on datasets like LAION-400M, LAION-2B, and DataComp-1B.  See the [Hugging Face Hub](https://huggingface.co/models?library=open_clip) for available model cards.
*   **Reproducible Research:** Built upon the foundation of the original CLIP paper and extended with additional models and training runs detailed in the paper on [reproducible scaling laws for contrastive language-image learning](https://arxiv.org/abs/2212.07143).
*   **Zero-Shot Capabilities:**  Leverage models that achieve impressive zero-shot performance on tasks like ImageNet classification.
*   **Flexible Training:**  Train your own CLIP models with a customizable training pipeline supporting webdataset and multi-GPU training.
*   **Model Distillation:** Experiment with model distillation for improved performance.
*   **CoCa Support:** Includes support for CoCa models and training.
*   **Int8 Support:** Enables int8 training and inference for significant performance improvements.

**Key Advantages:**
*   **State-of-the-Art Performance:** OpenCLIP models excel at zero-shot image classification and other tasks.
*   **Extensive Pre-trained Models:** Ready-to-use models covering a wide variety of architectures and training datasets are readily available.
*   **Modular and Customizable:** Allows you to fine-tune and build your models on existing model parameters.
*   **Active Development:** The repository is actively maintained and updated with new features, models, and performance improvements.

**Getting Started:**

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

    For more detailed examples, including a working [CLIP Colab](https://colab.research.google.com/github/mlfoundations/open_clip/blob/master/docs/Interacting_with_open_clip.ipynb), explore the interactive notebooks in the `docs` directory.

3.  **Model Loading:**
    ```python
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    ```
    Explore available pre-trained models with:
    ```python
    >>> import open_clip
    >>> open_clip.list_pretrained()
    ```

4.  **Fine-tuning and Training:**
    This repository is focused on training CLIP models. To fine-tune a *trained* zero-shot model on a downstream classification task such as ImageNet, please see [our other repository: WiSE-FT](https://github.com/mlfoundations/wise-ft).

    For training details, refer to the [Training CLIP](#training-clip) section.

**[Original GitHub Repo](https://github.com/mlfoundations/open_clip)**