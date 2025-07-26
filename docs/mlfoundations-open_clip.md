# OpenCLIP: Open Source CLIP Implementation for Image-Text Understanding

OpenCLIP offers an open-source implementation of OpenAI's [CLIP](https://arxiv.org/abs/2103.00020), enabling you to train and utilize powerful image-text models.

*   **Pre-trained Models:** Access a wide variety of pre-trained models, including those trained on datasets like LAION-400M, LAION-2B, and DataComp-1B. ([More Details](docs/PRETRAINED.md))
*   **Reproducible Research:**  This repository supports the research in [Reproducible scaling laws for contrastive language-image learning](https://arxiv.org/abs/2212.07143), as well as a wide range of other recent research.
*   **Flexible Training:**  Train your own CLIP models with support for multi-GPU training, SLURM clusters, and various data formats, including webdataset.
*   **Zero-Shot Capabilities:** Leverage the zero-shot image classification performance of our models, evaluated on 38 datasets. ([Results](docs/openclip_results.csv))
*   **Fine-tuning Support:** Utilize the trained zero-shot model on a downstream classification task.  For fine-tuning, see our other repository: [WiSE-FT](https://github.com/mlfoundations/wise-ft).
*   **CoCa Model Training Support:** Train CoCa models to perform image captioning and other multimodal tasks.
*   **Int8 Support:**  Experimental support for int8 training and inference.
*   **Model Distillation:** You can distill from a pre-trained by specifying `--distill-model` and `--distill-pretrained` to specify the model you'd like to distill from.
*   **Hugging Face Hub Integration:** Easily push models and configurations to the Hugging Face Hub using `open_clip.push_to_hf_hub`.

**Get started with OpenCLIP today to explore the potential of image-text understanding!**

[Visit the Original Repository on GitHub](https://github.com/mlfoundations/open_clip)

## Key Features

*   **Model Variety:** Offers a wide selection of pre-trained models with diverse architectures, including ConvNext, ViT, and SigLIP models.
*   **Data Compatibility:** Supports various datasets and formats, including CSV, webdataset, and integration with img2dataset for data download.
*   **Training Flexibility:** Provides comprehensive training scripts, including multi-GPU support, gradient accumulation, and integration with logging tools like TensorBoard and WandB.
*   **CoCa Training:** Support for training CoCa models.
*   **Efficient Training:** Includes features like patch dropout and model distillation to optimize training.
*   **Easy to use inference**: Provides simple example usage to load the models and tokenize inputs.

## Quick Start

1.  **Installation:**

    ```bash
    pip install open_clip_torch
    ```

2.  **Inference Example:**

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

## Pretrained Models

A comprehensive list of pre-trained models is available, with details on their performance and training data. [More Details](docs/PRETRAINED.md).

## Training

Detailed instructions and scripts are provided for training CLIP models, including multi-GPU and SLURM configurations.

*   **Single-Process Example:**

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

## Evaluation

Evaluation scripts and guidance are provided for assessing model performance on various benchmarks.

*   **Evaluating a Local Checkpoint:**

    ```bash
    python -m open_clip_train.main \
        --val-data="/path/to/validation_data.csv"  \
        --model RN101 \
        --pretrained /path/to/checkpoints/epoch_K.pt
    ```

## Acknowledgments

The OpenCLIP project is supported by the Gauss Centre for Supercomputing e.V. (GCS) and the John von Neumann Institute for Computing (NIC).  We thank the team members and contributors listed in the original README for their valuable contributions to this project.