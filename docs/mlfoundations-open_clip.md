# OpenCLIP: Open Source Implementation of CLIP for Image-Text Pre-training

**OpenCLIP provides an open-source implementation of OpenAI's CLIP, offering a robust framework for training and utilizing state-of-the-art image-text models.** This repository allows researchers and developers to leverage powerful contrastive language-image pre-training models for various applications, including image retrieval, classification, and generation.  [Explore the Original Repo](https://github.com/mlfoundations/open_clip)

**Key Features:**

*   **Pre-trained Models:** Access a wide array of pre-trained models trained on diverse datasets (LAION-400M, LAION-2B, DataComp-1B) for immediate use.  Find detailed model information and zero-shot results [here](docs/PRETRAINED.md).
*   **Reproducible Results:**  Train and evaluate CLIP models, with many models and their scaling properties studied in detail in the paper [reproducible scaling laws for contrastive language-image learning](https://arxiv.org/abs/2212.07143).
*   **Flexible Training:**  Supports training CLIP models with various architectures (ConvNext, ViT, etc.) and datasets.  Detailed instructions are available to easily customize your training process.
*   **Easy Integration:**  Provides a simple model interface, including tokenizers and preprocessing, for seamless integration into your projects.
*   **Multi-GPU and Distributed Training:** Optimized for efficient training on multiple GPUs and distributed setups, scaling to 1024+ GPUs with ease.
*   **Fine-tuning Support:**  Fine-tune trained zero-shot models on downstream classification tasks using [WiSE-FT](https://github.com/mlfoundations/wise-ft).
*   **CoCa Model Support:** Training and Generating capabilities for CoCa models [CoCa Paper](https://arxiv.org/abs/2205.01917).

**Key Models and Zero-Shot ImageNet-1k Accuracy:**

| Model              | Training Data | Resolution | ImageNet Zero-Shot Accuracy |
| ------------------ | ------------- | ---------- | --------------------------- |
| ConvNext-Base      | LAION-2B      | 256px      | 71.5%                       |
| ConvNext-Large     | LAION-2B      | 320px      | 76.9%                       |
| ConvNext-XXLarge   | LAION-2B      | 256px      | 79.5%                       |
| ViT-B-32-256       | DataComp-1B   | 256px      | 72.8%                       |
| ViT-B-16           | DataComp-1B   | 224px      | 73.5%                       |
| ViT-L-14           | LAION-2B      | 224px      | 75.3%                       |
| ViT-H-14           | LAION-2B      | 224px      | 78.0%                       |
| ViT-L-14           | DataComp-1B   | 224px      | 79.2%                       |
| ViT-bigG-14        | LAION-2B      | 224px      | 80.1%                       |
| ViT-L-14-quickgelu (Original CLIP)  | WIT | 224px | 75.5% |
| ViT-SO400M-14-SigLIP [(SigLIP)](https://arxiv.org/abs/2303.15343) | WebLI | 224px | 82.0% |
| ViT-L-14 [(DFN)](https://arxiv.org/abs/2309.17425) | DFN-2B | 224px | 82.2% |
| ViT-L-16-256 [(SigLIP2)](https://arxiv.org/abs/2502.14786) |  WebLI (multi-lang) | 256px | 82.5% |
| ViT-SO400M-14-SigLIP-384 [(SigLIP)](https://arxiv.org/abs/2303.15343) |  WebLI | 384px | 83.1% |
| ViT-H-14-quickgelu [(DFN)](https://arxiv.org/abs/2309.17425) | DFN-5B | 224px | 83.4% |
| PE-Core-L-14-336 [(PE)](https://arxiv.org/abs/2504.13181) | MetaCLIP-5.4B | 336px | 83.5% |
| ViT-SO400M-16-SigLIP2-384 [(SigLIP2)](https://arxiv.org/abs/2502.14786) |  WebLI (multi-lang) | 384px | 84.1% |
| ViT-H-14-378-quickgelu [(DFN)](https://arxiv.org/abs/2309.17425) | DFN-5B | 378px | 84.4% |
| ViT-gopt-16-SigLIP2-384 [(SigLIP2)](https://arxiv.org/abs/2502.14786) | WebLI (multi-lang) | 384px | 85.0% |
| PE-Core-bigG-14-448 [(PE)](https://arxiv.org/abs/2504.13181) | MetaCLIP-5.4B | 448px | 86B | 85.4% |

**Usage Example:**

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

**Installation:**

```bash
pip install open_clip_torch
```

**Resources:**

*   [Paper](https://arxiv.org/abs/2212.07143)
*   [CLIP Colab](https://colab.research.google.com/github/mlfoundations/open_clip/blob/master/docs/Interacting_with_open_clip.ipynb)
*   [CoCa Colab](https://colab.research.google.com/github/mlfoundations/open_clip/blob/master/docs/Interacting_with_open_coca.ipynb)
*   [Hugging Face Hub](https://huggingface.co/models?library=open_clip)
*   [OpenAI's CLIP repository](https://github.com/openai/CLIP)

**Get Started Today**

OpenCLIP provides a powerful toolkit for researchers and developers to access the power of image-text models. Dive into the documentation and examples to start exploring the possibilities!