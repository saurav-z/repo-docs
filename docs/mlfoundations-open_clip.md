# OpenCLIP: Open Source Implementation of CLIP (Contrastive Language-Image Pre-training)

**Unlock the power of visual understanding with OpenCLIP, offering state-of-the-art open-source models for image-text understanding and generation.**

[Paper](https://arxiv.org/abs/2212.07143) | [Citations](#citing) | [CLIP Colab](https://colab.research.google.com/github/mlfoundations/open_clip/blob/master/docs/Interacting_with_open_clip.ipynb) | [CoCa Colab](https://colab.research.google.com/github/mlfoundations/open_clip/blob/master/docs/Interacting_with_open_coca.ipynb) | [PyPI](https://pypi.python.org/pypi/open_clip_torch)

OpenCLIP provides an open-source implementation of OpenAI's CLIP (Contrastive Language-Image Pre-training) for researchers and developers to experiment with and build upon. We offer a wide range of pre-trained models, detailed performance results, and flexible training options.

## Key Features

*   **Pre-trained Models:** Access a diverse collection of pre-trained models trained on datasets like LAION-400M, LAION-2B, and DataComp-1B.  Explore a variety of architectures, including ConvNeXt and ViT models, for optimal performance.
*   **Reproducible Results:** Our models and their scaling properties are thoroughly studied and documented in our research paper, "Reproducible scaling laws for contrastive language-image learning".
*   **Zero-Shot Performance:** Evaluate models on 38 different datasets with zero-shot results, showcasing their ability to generalize to new tasks without fine-tuning.  Explore ImageNet zero-shot accuracy from 71.5% up to 85.4% for the best models.
*   **Flexible Training:** Train your own CLIP models with extensive training configurations and multi-GPU/multi-node support using `torchrun` and SLURM.  Easily integrate with datasets like Conceptual Captions and YFCC using webdataset format.
*   **CoCa Support:** Experiment with CoCa models (Contrastive Captioners) by using pre-configured CoCa models and instructions for fine-tuning and text generation.
*   **Int8 Support:** Optimize for int8 training and inference, leading to training speedups with no accuracy loss.
*   **Hugging Face Hub Integration:** Easily push your models to the Hugging Face Hub for sharing and collaboration.

## Model Performance Highlights

The following table highlights some of the top-performing models available in OpenCLIP, along with their ImageNet zero-shot accuracy:

| Model                    | Training Data     | Resolution | # of Samples | ImageNet Zero-Shot Accuracy |
| ------------------------ | ----------------- | ---------- | ------------- | --------------------------- |
| ConvNext-Base            | LAION-2B          | 256px      | 13B           | 71.5%                       |
| ConvNext-Large           | LAION-2B          | 320px      | 29B           | 76.9%                       |
| ConvNext-XXLarge         | LAION-2B          | 256px      | 34B           | 79.5%                       |
| ViT-bigG-14              | LAION-2B          | 224px      | 34B           | 80.1%                       |
| ViT-L-14-quickgelu       | WIT               | 224px      | 13B           | 75.5%                       |
| ViT-SO400M-14-SigLIP     | WebLI             | 224px      | 45B           | 82.0%                       |
| ViT-H-14-378-quickgelu   | DFN-5B            | 378px      | 44B           | 84.4%                       |
| PE-Core-bigG-14-448      | MetaCLIP-5.4B     | 448px      | 86B           | 85.4%                       |
| &nbsp;                   | &nbsp;            | &nbsp;     | &nbsp;        | &nbsp;                      |

*For the full model list and zero-shot results, see the [PRETRAINED](docs/PRETRAINED.md) and [openclip_results.csv](docs/openclip_results.csv) documents.*

## Quick Start: Installation and Usage

1.  **Installation:**
    ```bash
    pip install open_clip_torch
    ```

2.  **Basic Usage Example:**
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

    *For more detailed examples and explanations, see the [CLIP Colab](https://colab.research.google.com/github/mlfoundations/open_clip/blob/master/docs/Interacting_with_open_clip.ipynb).*

## Training with OpenCLIP

OpenCLIP offers comprehensive support for training your own CLIP models.  Follow these key steps to get started:

1.  **Install Training Dependencies:**
    ```bash
    pip install 'open_clip_torch[training]'
    ```

2.  **Prepare Your Data:**  Format your image-text data into a compatible format such as CSV or webdataset.  See the [Data](#data) section for details.

3.  **Run the Training Script:** Use the provided training script (`open_clip_train.main`) with various command-line arguments to configure your training run.  See the example command below:
    ```bash
    python -m open_clip_train.main \
        --save-frequency 1 \
        --zeroshot-frequency 1 \
        --report-to tensorboard \
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

    *Refer to the [Training CLIP](#training-clip) section of the original README for detailed instructions and options.*

4. **Multi-GPU and Beyond** Comprehensive support for distributed training, SLURM cluster integration, and gradient accumulation are provided to efficiently scale to your compute needs.  See [Multi-GPU and Beyond](#multi-gpu-and-beyond) section for detailed instructions and options.

## Additional Resources

*   **Model Cards:** Find additional model-specific details on the Hugging Face Hub: [https://huggingface.co/models?library=open_clip](https://huggingface.co/models?library=open_clip)
*   **WiSE-FT:** For fine-tuning on downstream tasks, see our [WiSE-FT repository](https://github.com/mlfoundations/wise-ft).
*   **Data:** Learn about recommended datasets and data preparation in the [Data](#data) section of the original README.

## Acknowledgements

We extend our gratitude to the Gauss Centre for Supercomputing e.V. for funding and computing resources. We also thank the OpenCLIP Team and the original authors of CLIP.

## Cite Us

If you use this repository, please cite the following publications:

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

```bibtex
@inproceedings{cherti2023reproducible,
  title={Reproducible scaling laws for contrastive language-image learning},
  author={Cherti, Mehdi and Beaumont, Romain and Wightman, Ross and Wortsman, Mitchell and Ilharco, Gabriel and Gordon, Cade and Schuhmann, Christoph and Schmidt, Ludwig and Jitsev, Jenia},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={2818--2829},
  year={2023}
}
```
```bibtex
@inproceedings{Radford2021LearningTV,
  title={Learning Transferable Visual Models From Natural Language Supervision},
  author={Alec Radford and Jong Wook Kim and Chris Hallacy and A. Ramesh and Gabriel Goh and Sandhini Agarwal and Girish Sastry and Amanda Askell and Pamela Mishkin and Jack Clark and Gretchen Krueger and Ilya Sutskever},
  booktitle={ICML},
  year={2021}
}
```
```bibtex
@inproceedings{schuhmann2022laionb,
  title={{LAION}-5B: An open large-scale dataset for training next generation image-text models},
  author={Christoph Schuhmann and
          Romain Beaumont and
          Richard Vencu and
          Cade W Gordon and
          Ross Wightman and
          Mehdi Cherti and
          Theo Coombes and
          Aarush Katta and
          Clayton Mullis and
          Mitchell Wortsman and
          Patrick Schramowski and
          Srivatsa R Kundurthy and
          Katherine Crowson and
          Ludwig Schmidt and
          Robert Kaczmarczyk and
          Jenia Jitsev},
  booktitle={Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2022},
  url={https://openreview.net/forum?id=M3Y74vmsMcY}
}
```

[![DOI](https://zenodo.org/badge/390536799.svg)](https://zenodo.org/badge/latestdoi/390536799)

**Contribute and collaborate on OpenCLIP:  [https://github.com/mlfoundations/open_clip](https://github.com/mlfoundations/open_clip)**