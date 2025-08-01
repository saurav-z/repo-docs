# OpenCLIP: Open Source CLIP Models for Image-Text Understanding

**OpenCLIP provides open-source implementations of CLIP (Contrastive Language-Image Pre-training), enabling powerful image-text understanding across various applications. [Explore the original repository](https://github.com/mlfoundations/open_clip) for in-depth details.**

**Key Features:**

*   **Pre-trained Models:** Access a diverse collection of pre-trained models trained on various datasets like LAION-400M, LAION-2B, and DataComp-1B.
*   **Reproducibility:**  Built upon the paper "Reproducible scaling laws for contrastive language-image learning" (see citations below).
*   **Flexible Architecture:** Supports models using different image encoders and tokenizers, including ConvNext, ViT, and more.
*   **Easy Integration:** Simple model interface to instantiate pre-trained and untrained models, and readily available PyPi package (`open_clip_torch`).
*   **Fine-tuning Support:**  Includes support for fine-tuning on downstream classification tasks.
*   **Training Tools:** Comprehensive training scripts and utilities for data loading, distributed training, and evaluation.
*   **CoCa Support**: Integrated with CoCa models
*   **Hugging Face Integration:**  Push your model to the Hugging Face Hub.
*   **Int8 Support:**  Support for int8 training and inference for training speedup

**Key Results:**

OpenCLIP models have achieved impressive zero-shot ImageNet-1k accuracy. See below for a sample of the best-performing models (updated from the original, and with additional context from the original README):

| Model                     | Training data      | Resolution | ImageNet zero-shot acc. |
| ------------------------- | ------------------ | ---------- | ------------------------ |
| ConvNext-Base             | LAION-2B           | 256px      | 71.5%                    |
| ConvNext-Large            | LAION-2B           | 320px      | 76.9%                    |
| ConvNext-XXLarge          | LAION-2B           | 256px      | 79.5%                    |
| ViT-B-32-256              | DataComp-1B        | 256px      | 72.8%                    |
| ViT-B-16                  | DataComp-1B        | 224px      | 73.5%                    |
| ViT-L-14                  | LAION-2B           | 224px      | 75.3%                    |
| ViT-H-14                  | LAION-2B           | 224px      | 78.0%                    |
| ViT-L-14                  | DataComp-1B        | 224px      | 79.2%                    |
| ViT-bigG-14               | LAION-2B           | 224px      | 80.1%                    |
| ViT-L-14-quickgelu        | WIT                | 224px      | 75.5%                    |
| ViT-SO400M-14-SigLIP     | WebLI              | 224px      | 82.0%                    |
| ViT-L-14                  | DFN-2B             | 224px      | 82.2%                    |
| ViT-L-16-256              | WebLI (multi-lang) | 256px      | 82.5%                    |
| ViT-SO400M-14-SigLIP-384 | WebLI              | 384px      | 83.1%                    |
| ViT-H-14-quickgelu        | DFN-5B             | 224px      | 83.4%                    |
| PE-Core-L-14-336          | MetaCLIP-5.4B      | 336px      | 83.5%                    |
| ViT-SO400M-16-SigLIP2-384 | WebLI (multi-lang) | 384px      | 84.1%                    |
| ViT-H-14-378-quickgelu    | DFN-5B             | 378px      | 84.4%                    |
| ViT-gopt-16-SigLIP2-384   | WebLI (multi-lang) | 384px      | 85.0%                    |
| PE-Core-bigG-14-448       | MetaCLIP-5.4B      | 448px      | 85.4%                    |

**Getting Started:**

1.  **Installation:** `pip install open_clip_torch`
2.  **Quick Usage:**

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

**Pretrained Models:**

List available pretrained models:

```python
>>> import open_clip
>>> open_clip.list_pretrained()
```

Find details like model parameters in the [model_profile.csv](docs/model_profile.csv).

**Citations:**

If you found this repository useful, please consider citing the following works:

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