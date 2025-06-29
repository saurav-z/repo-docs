<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/transformers-logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/transformers-logo-light.svg">
    <img alt="Hugging Face Transformers Library" src="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/transformers-logo-light.svg" width="352" height="59" style="max-width: 100%;">
  </picture>
  <br/>
  <br/>
</p>

<p align="center">
    <a href="https://huggingface.com/models"><img alt="Checkpoints on Hub" src="https://img.shields.io/endpoint?url=https://huggingface.co/api/shields/models&color=brightgreen"></a>
    <a href="https://circleci.com/gh/huggingface/transformers"><img alt="Build" src="https://img.shields.io/circleci/build/github/huggingface/transformers/main"></a>
    <a href="https://github.com/huggingface/transformers/blob/main/LICENSE"><img alt="GitHub" src="https://img.shields.io/github/license/huggingface/transformers.svg?color=blue"></a>
    <a href="https://huggingface.co/docs/transformers/index"><img alt="Documentation" src="https://img.shields.io/website/http/huggingface.co/docs/transformers/index.svg?down_color=red&down_message=offline&up_message=online"></a>
    <a href="https://github.com/huggingface/transformers/releases"><img alt="GitHub release" src="https://img.shields.io/github/release/huggingface/transformers.svg"></a>
    <a href="https://github.com/huggingface/transformers/blob/main/CODE_OF_CONDUCT.md"><img alt="Contributor Covenant" src="https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg"></a>
    <a href="https://zenodo.org/badge/latestdoi/155220641"><img src="https://zenodo.org/badge/155220641.svg" alt="DOI"></a>
</p>

<h4 align="center">
    <p>
        <b>English</b> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_zh-hans.md">简体中文</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_zh-hant.md">繁體中文</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ko.md">한국어</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_es.md">Español</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ja.md">日本語</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_hd.md">हिन्दी</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ru.md">Русский</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_pt-br.md">Рortuguês</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_te.md">తెలుగు</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_fr.md">Français</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_de.md">Deutsch</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_vi.md">Tiếng Việt</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ar.md">العربية</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ur.md">اردو</a> |
    </p>
</h4>

## Transformers: Your Gateway to State-of-the-Art AI Models

**Harness the power of cutting-edge AI with the Hugging Face Transformers library, providing pre-trained models for various tasks, from text generation to image classification.**

[![Hugging Face Transformers Library](https://huggingface.co/datasets/huggingface/documentation-images/raw/main/transformers-logo-light.svg)](https://github.com/huggingface/transformers)

<br/>

**Key Features:**

*   **Extensive Pre-trained Models:** Access over 1 million pre-trained models on the [Hugging Face Hub](https://huggingface.com/models), covering a wide range of modalities and tasks.
*   **Unified API:**  Easily utilize pre-trained models across different frameworks with a consistent and intuitive API.
*   **Simplified Usage:** Reduce complexity with user-friendly abstractions for both inference and training, enabling rapid prototyping and development.
*   **Cost-Effective:** Leverage pre-trained models to lower compute costs and reduce your carbon footprint.
*   **Flexibility:** Fine-tune or customize models to meet your specific requirements, and move models seamlessly between PyTorch, JAX, and TensorFlow.

### Installation

Install the library using pip or uv, and specify the framework (PyTorch, TensorFlow, or Flax) you will be using:

```bash
# Install with pip
pip install "transformers[torch]"
# Install with uv
uv pip install "transformers[torch]"
```

For the latest changes and contributions, consider installing from source:

```bash
git clone https://github.com/huggingface/transformers.git
cd transformers
pip install .[torch]
```

### Quickstart: Text Generation

Get started with Transformers immediately using the [`pipeline`](https://huggingface.co/docs/transformers/pipeline_tutorial) API:

```python
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="Qwen/Qwen2.5-1.5B")
pipeline("the secret to baking a really good cake is ")
```
```
[{'generated_text': 'the secret to baking a really good cake is 1) to use the right ingredients and 2) to follow the recipe exactly. the recipe for the cake is as follows: 1 cup of sugar, 1 cup of flour, 1 cup of milk, 1 cup of butter, 1 cup of eggs, 1 cup of chocolate chips. if you want to make 2 cakes, how much sugar do you need? To make 2 cakes, you will need 2 cups of sugar.'}]
```

### Explore Different Tasks

  *   [Automatic Speech Recognition](#automatic-speech-recognition)
  *   [Image Classification](#image-classification)
  *   [Visual Question Answering](#visual-question-answering)

```python
# Example for visual question answering
from transformers import pipeline

pipeline = pipeline(task="visual-question-answering", model="Salesforce/blip-vqa-base")
pipeline(
    image="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/idefics-few-shot.jpg",
    question="What is in the image?",
)
```

### Why Use Transformers?

*   **Easy-to-Use Models:**  Achieve high performance on various tasks with an intuitive API and three classes.
*   **Cost-Efficient:** Save time and resources by utilizing pre-trained models.
*   **Framework Flexibility:** Seamlessly train and deploy models across PyTorch, JAX, and TensorFlow.
*   **Customization:** Adapt models to your needs with the provided examples.

### Why Not Use Transformers?

*   **Not a Modular Toolbox:** The library is focused on model definitions, not modular building blocks.
*   **Training API:**  Primarily optimized for PyTorch models provided by Transformers. Consider other libraries like [Accelerate](https://huggingface.co/docs/accelerate) for generic ML loops.
*   **Example Scripts:**  Adapt example scripts to fit your unique use case.

### Projects Using Transformers

Explore the [awesome-transformers](./awesome-transformers.md) page showcasing 100 amazing community projects built with Transformers.

### Example Models

#### Audio

*   [Whisper](https://huggingface.co/openai/whisper-large-v3-turbo)
*   [Moonshine](https://huggingface.co/UsefulSensors/moonshine)
*   [Wav2Vec2](https://huggingface.co/superb/wav2vec2-base-superb-ks)
*   [Moshi](https://huggingface.co/kyutai/moshiko-pytorch-bf16)
*   [MusicGen](https://huggingface.co/facebook/musicgen-large)
*   [Bark](https://huggingface.co/suno/bark)

#### Computer Vision

*   [SAM](https://huggingface.co/facebook/sam-vit-base)
*   [DepthPro](https://huggingface.co/apple/DepthPro-hf)
*   [DINO v2](https://huggingface.co/facebook/dinov2-base)
*   [SuperGlue](https://huggingface.co/magic-leap-community/superglue_outdoor)
*   [SuperGlue](https://huggingface.co/magic-leap-community/superglue)
*   [RT-DETRv2](https://huggingface.co/PekingU/rtdetr_v2_r50vd)
*   [VitPose](https://huggingface.co/usyd-community/vitpose-base-simple)
*   [OneFormer](https://huggingface.co/shi-labs/oneformer_ade20k_swin_large)
*   [VideoMAE](https://huggingface.co/MCG-NJU/videomae-large)

#### Multimodal

*   [Qwen2-Audio](https://huggingface.co/Qwen/Qwen2-Audio-7B)
*   [LayoutLMv3](https://huggingface.co/microsoft/layoutlmv3-base)
*   [Qwen-VL](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
*   [BLIP-2](https://huggingface.co/Salesforce/blip2-opt-2.7b)
*   [GOT-OCR2](https://huggingface.co/stepfun-ai/GOT-OCR-2.0-hf)
*   [TAPAS](https://huggingface.co/google/tapas-base)
*   [Emu3](https://huggingface.co/BAAI/Emu3-Gen)
*   [Llava-OneVision](https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-ov-hf)
*   [Llava](https://huggingface.co/llava-hf/llava-1.5-7b-hf)
*   [Kosmos-2](https://huggingface.co/microsoft/kosmos-2-patch14-224)

#### NLP

*   [ModernBERT](https://huggingface.co/answerdotai/ModernBERT-base)
*   [Gemma](https://huggingface.co/google/gemma-2-2b)
*   [Mixtral](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1)
*   [BART](https://huggingface.co/facebook/bart-large-cnn)
*   [T5](https://huggingface.co/google-t5/t5-base)
*   [Llama](https://huggingface.co/meta-llama/Llama-3.2-1B)
*   [Qwen](https://huggingface.co/Qwen/Qwen2.5-0.5B)

### Citation

For academic use, please cite our paper:

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