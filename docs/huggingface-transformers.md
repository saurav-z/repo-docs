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

<h1 align="center">Hugging Face Transformers: State-of-the-Art Models for NLP, Computer Vision, and More</h1>

<p align="center">
    Unlock the power of advanced AI with the <a href="https://github.com/huggingface/transformers">Hugging Face Transformers library</a>, providing pre-trained models for various modalities.
</p>

<h3 align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/transformers_as_a_model_definition.png"/>
</h3>

Hugging Face Transformers is a powerful, open-source library offering cutting-edge, pre-trained models for natural language processing (NLP), computer vision, audio, video, and multimodal tasks, enabling both inference and training.  This library centralizes model definitions, ensuring compatibility across diverse frameworks and making it a central hub for model development and deployment.

Key features include:

*   **Vast Model Library:** Access over 1 million pre-trained model checkpoints on the [Hugging Face Hub](https://huggingface.co/models?library=transformers&sort=trending).
*   **Unified API:**  A simple and consistent API for using all pre-trained models.
*   **Cross-Framework Compatibility:** Seamless integration with major training frameworks (PyTorch, TensorFlow, JAX) and inference engines (vLLM, SGLang).
*   **Ease of Use:** Simplified model usage through the `Pipeline` API, allowing quick implementation across tasks.
*   **Customization and Flexibility:** Easily adapt models and examples to meet specific project requirements.
*   **Cost-Effective:**  Reduce compute costs and carbon footprint by leveraging pre-trained models instead of training from scratch.

## Installation

Transformers supports Python 3.9+ with PyTorch, TensorFlow, and Flax.

Choose your installation method:

**1. Using venv (recommended)**

```bash
# Create and activate a virtual environment
python -m venv .my-env
source .my-env/bin/activate
```

**2.  Using uv (fast package manager)**
```bash
uv venv .my-env
source .my-env/bin/activate
```

Install the Transformers library:

```bash
pip install "transformers[torch]" # or "transformers[tf]" for Tensorflow
```

or
```bash
uv pip install "transformers[torch]" # or "transformers[tf]" for Tensorflow
```

To install from source:

```bash
git clone https://github.com/huggingface/transformers.git
cd transformers
pip install .[torch] # or pip install .[tf]
```
or
```bash
uv pip install .[torch] # or uv pip install .[tf]
```

## Quickstart

Get started quickly with the `Pipeline` API for various tasks:

```python
from transformers import pipeline

# Example: Text Generation
generator = pipeline(task="text-generation", model="Qwen/Qwen2.5-1.5B")
result = generator("the secret to baking a really good cake is ")
print(result[0]['generated_text'])
```
```python
# Example: Chat with a Model
import torch
from transformers import pipeline

chat = [
    {"role": "system", "content": "You are a sassy, wise-cracking robot as imagined by Hollywood circa 1986."},
    {"role": "user", "content": "Hey, can you tell me any fun things to do in New York?"}
]

pipeline = pipeline(task="text-generation", model="meta-llama/Meta-Llama-3-8B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")
response = pipeline(chat, max_new_tokens=512)
print(response[0]["generated_text"][-1]["content"])
```

Explore other modalities and tasks with the `Pipeline` API:

<details>
<summary>Automatic Speech Recognition</summary>

```python
from transformers import pipeline

asr = pipeline(task="automatic-speech-recognition", model="openai/whisper-large-v3")
result = asr("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
print(result['text'])
```

</details>

<details>
<summary>Image Classification</summary>

<h3 align="center">
    <a><img src="https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png"></a>
</h3>

```python
from transformers import pipeline

classifier = pipeline(task="image-classification", model="facebook/dinov2-small-imagenet1k-1-layer")
result = classifier("https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png")
print(result)
```

</details>

<details>
<summary>Visual Question Answering</summary>

<h3 align="center">
    <a><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/idefics-few-shot.jpg"></a>
</h3>

```python
from transformers import pipeline

vqa = pipeline(task="visual-question-answering", model="Salesforce/blip-vqa-base")
result = vqa(
    image="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/idefics-few-shot.jpg",
    question="What is in the image?",
)
print(result[0]['answer'])
```

</details>

## Why Use Transformers?

*   **Simplified AI:** Easy-to-use state-of-the-art models with a unified API.
*   **Efficiency:** Reduced compute costs and carbon footprint through model sharing.
*   **Flexibility:** Train, evaluate, and deploy models across different frameworks.
*   **Customization:** Easily adapt models to your specific needs.

## Why *Not* Use Transformers?

*   **Not a Modular Toolbox:** Designed for direct model iteration, not extensive abstraction.
*   **Training Limitations:** Training API primarily supports PyTorch models provided by Transformers.
*   **Example Scripts:**  Adapt scripts to your specific use cases.

## Community Projects

Explore the [awesome-transformers](./awesome-transformers.md) page to discover 100+ projects built with Transformers.

## Example Models

Test models directly on their [Hub model pages](https://huggingface.co/models).

<details>
<summary>Audio</summary>
<ul>
<li>Audio classification with [Whisper](https://huggingface.co/openai/whisper-large-v3-turbo)</li>
<li>Automatic speech recognition with [Moonshine](https://huggingface.co/UsefulSensors/moonshine)</li>
<li>Keyword spotting with [Wav2Vec2](https://huggingface.co/superb/wav2vec2-base-superb-ks)</li>
<li>Speech to speech generation with [Moshi](https://huggingface.co/kyutai/moshiko-pytorch-bf16)</li>
<li>Text to audio with [MusicGen](https://huggingface.co/facebook/musicgen-large)</li>
<li>Text to speech with [Bark](https://huggingface.co/suno/bark)</li>
</ul>
</details>

<details>
<summary>Computer vision</summary>
<ul>
<li>Automatic mask generation with [SAM](https://huggingface.co/facebook/sam-vit-base)</li>
<li>Depth estimation with [DepthPro](https://huggingface.co/apple/DepthPro-hf)</li>
<li>Image classification with [DINO v2](https://huggingface.co/facebook/dinov2-base)</li>
<li>Keypoint detection with [SuperPoint](https://huggingface.co/magic-leap-community/superpoint)</li>
<li>Keypoint matching with [SuperGlue](https://huggingface.co/magic-leap-community/superglue_outdoor)</li>
<li>Object detection with [RT-DETRv2](https://huggingface.co/PekingU/rtdetr_v2_r50vd)</li>
<li>Pose Estimation with [VitPose](https://huggingface.co/usyd-community/vitpose-base-simple)</li>
<li>Universal segmentation with [OneFormer](https://huggingface.co/shi-labs/oneformer_ade20k_swin_large)</li>
<li>Video classification with [VideoMAE](https://huggingface.co/MCG-NJU/videomae-large)</li>
</ul>
</details>

<details>
<summary>Multimodal</summary>
<ul>
<li>Audio or text to text with [Qwen2-Audio](https://huggingface.co/Qwen/Qwen2-Audio-7B)</li>
<li>Document question answering with [LayoutLMv3](https://huggingface.co/microsoft/layoutlmv3-base)</li>
<li>Image or text to text with [Qwen-VL](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)</li>
<li>Image captioning [BLIP-2](https://huggingface.co/Salesforce/blip2-opt-2.7b)</li>
<li>OCR-based document understanding with [GOT-OCR2](https://huggingface.co/stepfun-ai/GOT-OCR-2.0-hf)</li>
<li>Table question answering with [TAPAS](https://huggingface.co/google/tapas-base)</li>
<li>Unified multimodal understanding and generation with [Emu3](https://huggingface.co/BAAI/Emu3-Gen)</li>
<li>Vision to text with [Llava-OneVision](https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-ov-hf)</li>
<li>Visual question answering with [Llava](https://huggingface.co/llava-hf/llava-1.5-7b-hf)</li>
<li>Visual referring expression segmentation with [Kosmos-2](https://huggingface.co/microsoft/kosmos-2-patch14-224)</li>
</ul>
</details>

<details>
<summary>NLP</summary>
<ul>
<li>Masked word completion with [ModernBERT](https://huggingface.co/answerdotai/ModernBERT-base)</li>
<li>Named entity recognition with [Gemma](https://huggingface.co/google/gemma-2-2b)</li>
<li>Question answering with [Mixtral](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1)</li>
<li>Summarization with [BART](https://huggingface.co/facebook/bart-large-cnn)</li>
<li>Translation with [T5](https://huggingface.co/google-t5/t5-base)</li>
<li>Text generation with [Llama](https://huggingface.co/meta-llama/Llama-3.2-1B)</li>
<li>Text classification with [Qwen](https://huggingface.co/Qwen/Qwen2.5-0.5B)</li>
</ul>
</details>

## Citation

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