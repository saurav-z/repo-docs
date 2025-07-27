# OpenCLIP: Open Source Implementation of CLIP for Image-Text Understanding

**OpenCLIP is a powerful, open-source framework for training and utilizing Contrastive Language-Image Pre-training (CLIP) models, enabling a wide range of image-text applications.**  [Explore the original repository on GitHub](https://github.com/mlfoundations/open_clip).

*   **Key Features:**
    *   **Reproducible Scaling Laws:** Provides models and research on the scaling properties of CLIP models.
    *   **Pre-trained Models:** Access to a diverse collection of pre-trained models trained on various datasets, including LAION-400M, LAION-2B, and DataComp-1B.
    *   **Model Flexibility:** Support for various model architectures and configurations, including ConvNext, ViT, and more.
    *   **Zero-Shot Capabilities:** Enables zero-shot image classification and other tasks.
    *   **Easy to Use:** Simple model interface for instantiating and utilizing pre-trained models.
    *   **Fine-tuning Support:**  Provides guidance and resources for fine-tuning models on downstream tasks.
    *   **Training Framework:** Comprehensive training scripts for CLIP models, with support for multi-GPU training, distributed training, and various data sources.
    *   **CoCa Support:** Enables training and generation for CoCa models
    *   **Int8 Support:** Offers int8 training and inference for certain architectures.

*   **Pretrained Model Highlights:**

    | Model          | Training Data | Resolution | ImageNet Zero-Shot Accuracy |
    | -------------- | ------------- | ---------- | ---------------------------- |
    | ConvNext-Base  | LAION-2B      | 256px      | 71.5%                       |
    | ConvNext-Large | LAION-2B      | 320px      | 76.9%                       |
    | ViT-B-32       | DataComp-1B   | 256px      | 72.8%                       |
    | ViT-L-14       | LAION-2B      | 224px      | 75.3%                       |
    | ViT-H-14       | LAION-2B      | 224px      | 78.0%                       |
    | ViT-bigG-14    | LAION-2B      | 224px      | 80.1%                       |
    | PE-Core-bigG-14-448 | MetaCLIP-5.4B | 448px      | 85.4%                       |
   ... many other top performing models available.

*   **Installation:**

```bash
pip install open_clip_torch
```

*   **Quick Usage Example:**

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

*   **Model Cards:** Model cards with details are found on the Hugging Face Hub under the OpenCLIP library tag: [https://huggingface.co/models?library=open_clip](https://huggingface.co/models?library=open_clip)

*   **Training:**  Detailed instructions for training CLIP models are available, including multi-GPU and SLURM setups.

*   **Fine-tuning:** To fine-tune on classification tasks, refer to the [WiSE-FT repository](https://github.com/mlfoundations/wise-ft).

*   **Citing:**
    Please consider citing the following papers if you use OpenCLIP:

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

*   **Acknowledgments:**
    This project acknowledges funding from the Gauss Centre for Supercomputing e.V. (www.gauss-centre.eu) through the John von Neumann Institute for Computing (NIC) and the contributions of the team.