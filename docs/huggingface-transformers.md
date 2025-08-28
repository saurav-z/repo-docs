<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/transformers-logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/transformers-logo-light.svg">
    <img alt="Hugging Face Transformers Library" src="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/transformers-logo-light.svg" width="352" height="59" style="max-width: 100%;">
  </picture>
</p>

## Hugging Face Transformers: State-of-the-Art Models for AI

**Unlock the power of cutting-edge AI with the Hugging Face Transformers library, your gateway to pre-trained models for various tasks.**  This library offers a unified and easy-to-use interface for accessing and utilizing a vast range of pre-trained models for tasks including text, computer vision, audio, and multimodal applications. [Explore the original repository](https://github.com/huggingface/transformers) for more details.

**Key Features:**

*   **Wide Range of Models:** Access over 1 million pre-trained models on the [Hugging Face Hub](https://huggingface.co/models) for diverse tasks.
*   **Unified API:**  Streamlined API for all models, regardless of the task.
*   **Cross-Framework Compatibility:** Seamless integration with major training frameworks, inference engines, and related libraries.
*   **Easy to Use:**  Get started with the [Pipeline API](https://huggingface.co/docs/transformers/pipeline_tutorial) in just a few lines of code.
*   **Customizable & Efficient:**  Customize models and examples to your needs while reducing compute costs and carbon footprint.

### Installation

Choose your preferred installation method. Ensure you have Python 3.9+ and a virtual environment activated, or use a fast Rust-based Python package and project manager like [uv](https://docs.astral.sh/uv/).

**Install using pip:**

```bash
pip install "transformers[torch]"  # Install with PyTorch support (recommended)
# or
pip install "transformers[tensorflow]" # Install with TensorFlow support
```

**Install using uv:**

```bash
uv pip install "transformers[torch]" # Install with PyTorch support (recommended)
# or
uv pip install "transformers[tensorflow]" # Install with TensorFlow support
```

**Install from source (for the latest updates):**

```bash
git clone https://github.com/huggingface/transformers.git
cd transformers
pip install .[torch] # Install with PyTorch support (recommended)
# or
pip install .[tensorflow] # Install with TensorFlow support
```

### Quickstart: Get Started with Pipelines

The `Pipeline` API simplifies inference and training by handling input preprocessing and output post-processing.

```python
from transformers import pipeline

# Text generation example
pipeline = pipeline(task="text-generation", model="Qwen/Qwen2.5-1.5B")
print(pipeline("the secret to baking a really good cake is ") )
```

**Example: Chat with a Model**

```python
import torch
from transformers import pipeline

chat = [
    {"role": "system", "content": "You are a sassy, wise-cracking robot as imagined by Hollywood circa 1986."},
    {"role": "user", "content": "Hey, can you tell me any fun things to do in New York?"}
]

pipeline = pipeline(task="text-generation", model="meta-llama/Meta-Llama-3-8B-Instruct", dtype=torch.bfloat16, device_map="auto")
response = pipeline(chat, max_new_tokens=512)
print(response[0]["generated_text"][-1]["content"])
```

**Explore Pipeline Examples for Various Tasks:**

*   **Automatic Speech Recognition:** (Whisper)
*   **Image Classification:** (DINOv2)
*   **Visual Question Answering:** (BLIP)

### Why Use Transformers?

*   **Ease of Use:** Simple API, great for developers, and a large variety of models at your fingertips.
*   **Cost-Effective:** Leverage pre-trained models to reduce training time, compute costs, and environmental impact.
*   **Flexibility:** Train models in PyTorch/JAX/TF2.0 and easily switch between frameworks.
*   **Customization:** Customize models and examples to match your requirements.

###  Why Not Use Transformers?

*   Transformers is not meant to be a modular toolbox for building neural networks.
*   The training API is optimized for PyTorch models from Transformers.
*   The [example scripts](https://github.com/huggingface/transformers/tree/main/examples) are meant to be examples that you adapt to your own needs.

### Projects Using Transformers

The community around Transformers is vast and active. Check out the [awesome-transformers](./awesome-transformers.md) page for an impressive list of projects.

### Example Models

Explore a selection of models across different modalities:

*   **Audio:** (Whisper, Moonshine, Wav2Vec2, Moshi, MusicGen, Bark)
*   **Computer Vision:** (SAM, DepthPro, DINOv2, SuperPoint, SuperGlue, RT-DETRv2, VitPose, OneFormer, VideoMAE)
*   **Multimodal:** (Qwen2-Audio, LayoutLMv3, Qwen-VL, BLIP-2, GOT-OCR2, TAPAS, Emu3, Llava-OneVision, Llava, Kosmos-2)
*   **NLP:** (ModernBERT, Gemma, Mixtral, BART, T5, Llama, Qwen)

### Citation

```bibtex
@inproceedings{wolf-etal-2020-transformers,
    title = "Transformers: State-of-the-Art Natural Language Processing",
    author = "Thomas Wolf and Lysandre Debut and Victor Sanh and Julien Chaumond and Clement Delangue and Anthony Moi and Pierric Cistac and Tim Rault and Rémi Louf and Morgan Funtowicz and Joe Davison and Sam Shleifer and Patrick von Platen and Clara Ma and Yacine Jernite and Julien Plu and Canwen Xu and Teven Le Scao and Sylvain Gugger and Mariama Drame and Quentin Lhoest and Alexander M. Rush",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
    month = oct,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-demos.6",
    pages = "38--45"
}
```