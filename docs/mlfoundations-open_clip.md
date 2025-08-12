# OpenCLIP: Open-Source Contrastive Language-Image Pre-training

**Unlock the power of image-text understanding with OpenCLIP, an open-source implementation of OpenAI's CLIP, offering a comprehensive suite of pre-trained models and training tools.**  Access the original repository [here](https://github.com/mlfoundations/open_clip).

*   **Reproducible CLIP Models:** Train and utilize a wide range of CLIP models, from small-scale experiments to large-scale models trained on datasets like LAION-400M, LAION-2B, and DataComp-1B.
*   **Pre-trained Model Variety:** Choose from a diverse selection of pre-trained models, including ConvNext, ViT, and SigLIP architectures, with detailed performance metrics and zero-shot results across 38 datasets.
*   **Flexible Usage:** Easily load and utilize pre-trained models with a simple Python interface for image and text encoding, supporting various architectures and datasets.
*   **Comprehensive Training Tools:** Access robust training scripts and configurations for multi-GPU, multi-node, and SLURM environments, enabling efficient training on large-scale datasets.
*   **Fine-tuning and Integration:** Supports fine-tuning on downstream tasks and integration with tools like clip-retrieval for efficient embedding computation.

**Key Features:**

*   **Extensive Pre-trained Models:**  Access a wide range of CLIP models, with results on ImageNet and other benchmarks, see [here](docs/PRETRAINED.md).
*   **Flexible Model Loading:**  Easily load models using `open_clip.create_model_and_transforms`.  Supports local paths and Hugging Face Hub integration.
*   **Detailed Training Instructions:** Includes detailed instructions and examples for training CLIP models, including multi-GPU and SLURM configurations.
*   **Data Handling:** Supports various data formats like CSV and webdataset for flexible data loading and processing.
*   **CoCa Support:** Includes code for CoCa models and text generation (see [here](https://colab.research.google.com/github/mlfoundations/open_clip/blob/master/docs/Interacting_with_open_coca.ipynb)).
*   **Int8 Support:** Includes support for training and inference with INT8.

**Key Metrics & Performance (Partial list - More details in the README):**

| Model              | Training Data | Resolution | ImageNet Zero-Shot Acc. |
| ------------------ | ------------- | ---------- | ----------------------- |
| ConvNext-Base      | LAION-2B      | 256px      | 71.5%                  |
| ConvNext-Large     | LAION-2B      | 320px      | 76.9%                  |
| ViT-H-14           | LAION-2B      | 224px      | 78.0%                  |
| ViT-bigG-14        | LAION-2B      | 224px      | 80.1%                  |
| ViT-gopt-16-SigLIP2-384 | WebLI (multi-lang) | 384px | 85.0% |
| PE-Core-bigG-14-448  | MetaCLIP-5.4B | 448px | 85.4% |

**Getting Started**

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

    print("Label probs:", text_probs)
    ```
    **Further Resources:**
    *   [[Clip Colab]](https://colab.research.google.com/github/mlfoundations/open_clip/blob/master/docs/Interacting_with_open_clip.ipynb)
    *   [[Coca Colab]](https://colab.research.google.com/github/mlfoundations/open_clip/blob/master/docs/Interacting_with_open_coca.ipynb)

**Training and Evaluation**

Refer to the detailed instructions within the [original repository](https://github.com/mlfoundations/open_clip) for training CLIP models, including setup, data preparation, and multi-GPU configurations. Evaluation on ImageNet and other datasets is also covered.

**Acknowledgments & Citation**

This project is led by Ross Wightman, Romain Beaumont, Cade Gordon, and Vaishaal Shankar, with contributions from a broader team of researchers.  If you use OpenCLIP, please consider citing the relevant papers (see the original README for BibTeX).  We also acknowledge the Gauss Centre for Supercomputing for their support.