# OpenCLIP: State-of-the-Art Image-Text Models for Computer Vision

OpenCLIP is an open-source implementation of OpenAI's CLIP (Contrastive Language-Image Pre-training), offering a powerful suite of models trained on massive datasets to bridge the gap between images and text. Access the original repo [here](https://github.com/mlfoundations/open_clip).

**Key Features:**

*   **Pre-trained Models:** Explore a diverse selection of pre-trained models, including ConvNext, ViT, and others, trained on datasets like LAION-2B and DataComp-1B.
*   **Reproducible Results:** Train and evaluate your own models with a codebase focused on reproducible research.
*   **Flexible Usage:** Utilize pre-trained models for zero-shot image classification, image retrieval, and various downstream tasks.
*   **Fine-tuning Capabilities:**  Fine-tune pre-trained models on your specific classification datasets using the WiSE-FT repository.
*   **Easy Integration:**  Seamlessly integrate OpenCLIP into your projects with a straightforward `pip install` command and easy-to-use API.
*   **Multi-GPU and Distributed Training:** Efficiently train models on multiple GPUs, with support for SLURM clusters and gradient accumulation.
*   **Model Distillation**: Distill from pre-trained models to create smaller, faster models.
*   **Int8 Support**: Beta support for int8 training and inference.

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

**Pretrained Models:** Explore the available pre-trained models and their performance in the [PRETRAINED.md](docs/PRETRAINED.md) documentation.

**Training Guide:**  Detailed instructions for training your own OpenCLIP models can be found in the sections on Training CLIP, with examples for single-process and multi-GPU training, data preparation, and logging.

**Key Performance Highlights:**

| Model               | Training Data | Resolution | ImageNet Zero-Shot Accuracy |
| ------------------- | ------------- | ---------- | --------------------------- |
| ConvNext-Base       | LAION-2B      | 256px      | 71.5%                       |
| ConvNext-Large      | LAION-2B      | 320px      | 76.9%                       |
| ViT-H-14            | LAION-2B      | 224px      | 78.0%                       |
| ViT-L-14 (DataComp) | DataComp-1B   | 224px      | 79.2%                       |
| ViT-bigG-14         | LAION-2B      | 224px      | 80.1%                       |

**Citing**

If you found this repository useful, please consider citing the appropriate papers listed in the original README.
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