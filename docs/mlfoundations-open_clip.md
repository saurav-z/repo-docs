# OpenCLIP: Open Source Implementation of CLIP for Image-Text Understanding

**OpenCLIP provides a versatile and powerful open-source implementation of OpenAI's CLIP (Contrastive Language-Image Pre-training), enabling cutting-edge image-text understanding and generation.  [Explore the original repository](https://github.com/mlfoundations/open_clip) for full details.**

*   **Key Features:**
    *   **Pre-trained Models:** Access a wide range of pre-trained models trained on diverse datasets, including LAION-400M, LAION-2B, and DataComp-1B.
    *   **Reproducible Scaling Laws:** Explore models and their scaling properties as studied in detail in the paper [reproducible scaling laws for contrastive language-image learning](https://arxiv.org/abs/2212.07143).
    *   **Zero-Shot Performance:** Achieve impressive zero-shot performance on ImageNet-1k and other datasets.
    *   **Flexible Training:** Train CLIP models with support for multi-GPU, SLURM clusters, and various data sources.
    *   **Model Distillation:**  Distill from pre-trained models to create a smaller, more efficient models.
    *   **CoCa Support:** Train CoCa models (Contrastive Captions) for image captioning and generation.
    *   **Int8 Support:** Accelerate training and inference with int8 quantization.
    *   **Easy Integration:**  Seamlessly integrate with tools like [clip-retrieval](https://github.com/rom1504/clip-retrieval) for efficient embedding computations and the Hugging Face Hub for model sharing.

*   **Model Highlights:**
    *   Pretrained models are available on the Hugging Face Hub under the OpenCLIP library tag: https://huggingface.co/models?library=open_clip.

    | Model                | Training Data | Resolution | ImageNet Zero-Shot Accuracy |
    | -------------------- | ------------- | ---------- | --------------------------- |
    | ConvNext-Base        | LAION-2B      | 256px      | 71.5%                       |
    | ConvNext-Large       | LAION-2B      | 320px      | 76.9%                       |
    | ConvNext-XXLarge     | LAION-2B      | 256px      | 79.5%                       |
    | ViT-B-32-256         | DataComp-1B   | 256px      | 72.8%                       |
    | ViT-L-14             | DataComp-1B   | 224px      | 79.2%                       |
    | ViT-bigG-14          | LAION-2B      | 224px      | 80.1%                       |
    | ... (and many more!) | ...           | ...        | ...                         |

*   **Installation:**
    ```bash
    pip install open_clip_torch
    ```

*   **Usage Example:**
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

*   **Fine-tuning:**
    For fine-tuning zero-shot models on downstream tasks, explore the [WiSE-FT repository](https://github.com/mlfoundations/wise-ft).

*   **Training CLI Example:**
    ```bash
    python -m open_clip_train.main \
        --train-data="/path/to/train_data.csv"  \
        --val-data="/path/to/validation_data.csv"  \
        --csv-img-key filepath \
        --csv-caption-key title \
        --imagenet-val=/path/to/imagenet/root/val/ \
        --warmup 10000 \
        --batch-size=128 \
        --lr=1e-3 \
        --wd=0.1 \
        --epochs=30 \
        --workers=8 \
        --model RN50
    ```

*   **Acknowledgments and Citations:**  The development team gratefully acknowledges the Gauss Centre for Supercomputing e.V. (www.gauss-centre.eu) for funding this part of work by providing computing time through the John von Neumann Institute for Computing (NIC) on the GCS Supercomputer JUWELS Booster at Jülich Supercomputing Centre (JSC).  Please cite the provided bibtex entries if you use this library.

*   **Team:**
    Current development is led by [Ross Wightman](https://rwightman.com/), [Romain Beaumont](https://github.com/rom1504), [Cade Gordon](http://cadegordon.io/), and [Vaishaal Shankar](http://vaishaal.com/).