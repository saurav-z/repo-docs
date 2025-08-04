# OpenCLIP: Open Source Implementation of CLIP

**OpenCLIP provides an open-source implementation of CLIP, offering state-of-the-art performance in image-text understanding.** This repository allows you to leverage the power of Contrastive Language-Image Pre-training (CLIP) with a focus on reproducibility, scalability, and access to a wide range of pre-trained models. Check out the original repository on [GitHub](https://github.com/mlfoundations/open_clip)

**Key Features:**

*   **Pre-trained Models:** Access a diverse collection of pre-trained CLIP models, trained on datasets like LAION-400M, LAION-2B, and DataComp-1B, along with results for 38 datasets.
*   **Model Variety:** Supports various architectures, including ConvNext, ViT, and SigLIP models, with a range of sizes for different performance needs.
*   **Reproducibility:**  Explore the paper [Reproducible scaling laws for contrastive language-image learning](https://arxiv.org/abs/2212.07143) for in-depth analysis and insights into model scaling.
*   **Ease of Use:** Simple model instantiation and preprocessing steps for quick integration.
*   **Fine-tuning:**  Tools and guidance for fine-tuning zero-shot models on downstream classification tasks are available in the [WiSE-FT](https://github.com/mlfoundations/wise-ft) repository.
*   **Training Support:** Comprehensive instructions for training CLIP models with various data formats, distributed training configurations (including SLURM), and features like gradient accumulation and int8 support.
*   **Model Distillation**  Distill from a pre-trained model, `--distill-model` and `--distill-pretrained` to specify the model you'd like to distill from.
*   **Remote loading and Training:**  Support for remote filesystems (e.g. S3).

**Zero-Shot ImageNet-1k Performance (Example)**

| Model             | Training Data | ImageNet Zero-Shot Accuracy |
| ----------------- | ------------- | --------------------------- |
| ConvNext-Base     | LAION-2B      | 71.5%                       |
| ViT-H-14          | LAION-2B      | 78.0%                       |
| ViT-bigG-14       | LAION-2B      | 80.1%                       |
| ViT-gopt-16-SigLIP2-384 | WebLI (multi-lang) | 85.0%                       |

**Get Started:**

1.  **Installation:** `pip install open_clip_torch`
2.  **Example Usage:**

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
3.  **Explore Pretrained Models:** `open_clip.list_pretrained()`
4.  **Fine-tuning:** Refer to [WiSE-FT](https://github.com/mlfoundations/wise-ft) repository for fine-tuning details.

**For detailed information on usage, model details, and training procedures, please refer to the [official documentation](docs/PRETRAINED.md, and other docs)**

**Acknowledgments:**

This project is supported by the GCS and John von Neumann Institute for Computing (NIC) on the GCS Supercomputer JUWELS Booster at Jülich Supercomputing Centre (JSC).

**Citations:**

Please cite the relevant papers if you use this repository. Details can be found in the "Citing" section of the original README.