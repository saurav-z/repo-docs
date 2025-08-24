# OpenCLIP: Open Source Implementation of CLIP for Image-Text Learning

**OpenCLIP provides a robust, open-source implementation of CLIP (Contrastive Language-Image Pre-training) models, enabling researchers and developers to explore and utilize state-of-the-art image-text learning techniques.** ([Original Repo](https://github.com/mlfoundations/open_clip))

*   **Key Features:**

    *   **Pre-trained Models:** Access a wide range of pre-trained models trained on diverse datasets, including LAION-400M, LAION-2B, and DataComp-1B.
    *   **Reproducible Results:** Replicate and build upon research detailed in the paper "[Reproducible scaling laws for contrastive language-image learning](https://arxiv.org/abs/2212.07143)".
    *   **Model Versatility:** Load models such as ConvNext, ViT, and SigLIP architectures, supporting various training data resolutions and dataset sizes.
    *   **Ease of Use:** Simple Python interface for loading models, tokenizers, and performing image-text encoding.
    *   **Fine-tuning Support:** Leverage existing CLIP models and fine-tune them on downstream tasks.
    *   **Training Flexibility:** Comprehensive training scripts for multi-GPU and multi-node training, with support for SLURM clusters, data loading, and distributed training.
    *   **CoCa Models**: Implementation of CoCa Models, with fine tuning support.
    *   **Int8 Support**: Beta support for Int8 training and inference.
    *   **Remote Loading & Training**: Support for remote loading of checkpoints, and training while continuously backing up to remote file systems.

*   **Quickstart:**

    *   **Installation:** `pip install open_clip_torch`
    *   **Example Usage:**

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

*   **Pretrained Models:** See the [OpenCLIP Model Cards](https://huggingface.co/models?library=open_clip) for more model specific details.
*   **Fine-tuning:** Learn how to fine-tune the models in the [WiSE-FT repository](https://github.com/mlfoundations/wise-ft).
*   **Training Guide:** Detailed information on installing, training, and evaluating models can be found in the original README.
*   **CoCa Training:** You can train CoCa models by specifying a CoCa config using the ```--model``` parameter of the training script.

*   **Acknowledgments:**
    We gratefully acknowledge the Gauss Centre for Supercomputing e.V. (www.gauss-centre.eu) for funding this part of work by providing computing time through the John von Neumann Institute for Computing (NIC) on the GCS Supercomputer JUWELS Booster at Jülich Supercomputing Centre (JSC).

*   **Team:** Current development of this repository is led by [Ross Wightman](https://rwightman.com/), [Romain Beaumont](https://github.com/rom1504), [Cade Gordon](http://cadegordon.io/), and [Vaishaal Shankar](http://vaishaal.com/).

*   **Citing:** If you found this repository useful, please consider citing the relevant publications.  BibTeX entries for the papers and software are provided in the original README.