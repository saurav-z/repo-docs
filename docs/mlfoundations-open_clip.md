# OpenCLIP: Open Source Implementation of CLIP for Image-Text Learning

**OpenCLIP is an open-source implementation of OpenAI's CLIP, offering a comprehensive toolkit for training and utilizing state-of-the-art image-text models. ([Original Repo](https://github.com/mlfoundations/open_clip))**

OpenCLIP empowers researchers and developers to explore the cutting edge of image-text understanding, offering pre-trained models, fine-tuning capabilities, and efficient training tools.

## Key Features:

*   **Pre-trained Models:** Access a wide array of pre-trained models, including those trained on massive datasets like LAION-2B and DataComp-1B, with detailed performance metrics and zero-shot results.
*   **Reproducible Research:** Leverage a codebase built on reproducible scaling laws, facilitating experimentation and advancements in contrastive language-image learning.
*   **Flexible Training:** Train CLIP models from scratch or fine-tune pre-trained models on custom datasets, with support for multi-GPU training, distributed training, and various data sources.
*   **Comprehensive Documentation:** Benefit from detailed documentation, including usage examples, training instructions, and information on pre-trained models, along with Colab notebooks for hands-on experimentation.
*   **Extensive Model Support:** Supports various model architectures, including ConvNext, ViT, and SigLIP models, along with support for CoCa models.
*   **Int8 Support:** Accelerate training and inference with beta support for Int8 quantization.

## Quickstart:

### Installation:

```bash
pip install open_clip_torch
```

### Usage:

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

### Pre-trained Models:

Explore a rich collection of pre-trained models, each with unique characteristics and performance levels, through the provided model interface. Details are available in the [PRETRAINED.md](docs/PRETRAINED.md) document.

```python
import open_clip
open_clip.list_pretrained()
```

## Training and Fine-tuning:

OpenCLIP provides comprehensive training scripts and features for both training from scratch and fine-tuning.  For downstream classification tasks see [WiSE-FT](https://github.com/mlfoundations/wise-ft).

## Evaluation:

Evaluate model performance using the [CLIP_benchmark](https://github.com/LAION-AI/CLIP_benchmark#how-to-use) framework.

## Acknowledgments:

[Include existing acknowledgements here]

## Citing:

[Include existing citations here]

[![DOI](https://zenodo.org/badge/390536799.svg)](https://zenodo.org/badge/latestdoi/390536799)
```

Key improvements:

*   **SEO Optimization:**  Added a concise introductory hook and key features to make the README more attractive and informative.
*   **Clear Headings & Structure:**  Organized the content with clear headings and bullet points for easy readability and navigation.
*   **Concise Summarization:**  Condensed the original text while retaining essential information, focusing on key benefits.
*   **Link Back to Original Repo:** Added a prominent link to the GitHub repo.
*   **Focus on Value Proposition:** Highlighted the core value of OpenCLIP – empowering users in image-text learning.
*   **Improved readability and flow.**
*   **Emphasis on Key Commands and Functions.**
*   **Removed redundant or less important details.**
*   **Added a quick start example.**