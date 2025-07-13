# Hugging Face Transformers: State-of-the-Art Models for NLP, Computer Vision, Audio, and More

**Empower your AI projects with Hugging Face Transformers, the leading library offering pre-trained models for a wide range of tasks, from text generation to image classification, and much more!**  [Explore the original repo](https://github.com/huggingface/transformers).

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


## Key Features:

*   **Extensive Model Support:** Access to a vast collection of pre-trained models for text, vision, audio, and multimodal tasks.
*   **Ease of Use:** Simple, high-level APIs for quick model implementation and evaluation.
*   **Unified API:** Consistent interface for utilizing diverse pre-trained models, streamlining the development process.
*   **Framework Flexibility:** Seamlessly integrate models with PyTorch, TensorFlow, and Flax.
*   **Model Customization:** Modify models to suit your specific needs and tasks.
*   **Community-Driven:** Benefit from a large and active community and over 1M+ Transformers model checkpoints on the Hugging Face Hub.
*   **Cost-Effective:** Leverage pre-trained models to reduce compute costs and accelerate development.

## Installation

Transformers supports Python 3.9+ with PyTorch 2.1+, TensorFlow 2.6+, and Flax 0.4.1+.

**Using `venv` or `uv`:**

```bash
# venv
python -m venv .my-env
source .my-env/bin/activate
# uv
uv venv .my-env
source .my-env/bin/activate
```

**Install with pip or uv:**

```bash
# pip
pip install "transformers[torch]"
# uv
uv pip install "transformers[torch]"
```

**Install from source (for latest changes, potentially unstable):**

```bash
git clone https://github.com/huggingface/transformers.git
cd transformers
# pip
pip install .[torch]
# uv
uv pip install .[torch]
```

## Quickstart

Get started with the [Pipeline](https://huggingface.co/docs/transformers/pipeline_tutorial) API for various tasks.

```python
from transformers import pipeline

# Text Generation
pipeline = pipeline(task="text-generation", model="Qwen/Qwen2.5-1.5B")
print(pipeline("the secret to baking a really good cake is ") [0]['generated_text'])

# Chat with a model
import torch
chat = [{"role": "system", "content": "You are a sassy, wise-cracking robot."}, {"role": "user", "content": "Tell me fun things to do in New York?"}]
pipeline = pipeline(task="text-generation", model="meta-llama/Meta-Llama-3-8B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")
response = pipeline(chat, max_new_tokens=512)
print(response[0]["generated_text"][-1]["content"])
```

### Modality-Specific Examples

**[Expand to see examples for Automatic Speech Recognition, Image Classification, and Visual Question Answering - with code and images]**

*   Example 1: Automatic Speech Recognition
    ```python
    from transformers import pipeline
    pipeline = pipeline(task="automatic-speech-recognition", model="openai/whisper-large-v3")
    print(pipeline("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")['text'])
    ```
*   Example 2: Image Classification
    ```python
    from transformers import pipeline
    pipeline = pipeline(task="image-classification", model="facebook/dinov2-small-imagenet1k-1-layer")
    print(pipeline("https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png"))
    ```

*   Example 3: Visual Question Answering
    ```python
    from transformers import pipeline
    pipeline = pipeline(task="visual-question-answering", model="Salesforce/blip-vqa-base")
    print(pipeline(image="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/idefics-few-shot.jpg", question="What is in the image?"))
    ```

## Why Choose Transformers?

*   **State-of-the-Art Models:** Leverage cutting-edge models for diverse AI tasks.
*   **Reduced Costs:** Minimize training time and expenses with pre-trained models.
*   **Framework Agnostic:** Switch between PyTorch, JAX, and TensorFlow with ease.
*   **Highly Customizable:** Adapt models to your specific use cases.

## Limitations

*   Not a general-purpose neural network building block library (focus on model implementation).
*   Training API optimized for Transformers models; for broader ML use cases, explore [Accelerate](https://huggingface.co/docs/accelerate).
*   Example scripts serve as starting points and may require adaptation.

## Explore the Community

Discover 100+ community projects built with Transformers on the [awesome-transformers](./awesome-transformers.md) page.

## Example Models

**[Expand each category below to see examples of Audio, Computer Vision, Multimodal, and NLP models]**

*   **Audio**
    *   Audio classification with [Whisper](https://huggingface.co/openai/whisper-large-v3-turbo)
    *   Automatic speech recognition with [Moonshine](https://huggingface.co/UsefulSensors/moonshine)
    *   Keyword spotting with [Wav2Vec2](https://huggingface.co/superb/wav2vec2-base-superb-ks)
    *   Speech to speech generation with [Moshi](https://huggingface.co/kyutai/moshiko-pytorch-bf16)
    *   Text to audio with [MusicGen](https://huggingface.co/facebook/musicgen-large)
    *   Text to speech with [Bark](https://huggingface.co/suno/bark)
*   **Computer Vision**
    *   Automatic mask generation with [SAM](https://huggingface.co/facebook/sam-vit-base)
    *   Depth estimation with [DepthPro](https://huggingface.co/apple/DepthPro-hf)
    *   Image classification with [DINO v2](https://huggingface.co/facebook/dinov2-base)
    *   Keypoint detection with [SuperGlue](https://huggingface.co/magic-leap-community/superglue_outdoor)
    *   Keypoint matching with [SuperGlue](https://huggingface.co/magic-leap-community/superglue)
    *   Object detection with [RT-DETRv2](https://huggingface.co/PekingU/rtdetr_v2_r50vd)
    *   Pose Estimation with [VitPose](https://huggingface.co/usyd-community/vitpose-base-simple)
    *   Universal segmentation with [OneFormer](https://huggingface.co/shi-labs/oneformer_ade20k_swin_large)
    *   Video classification with [VideoMAE](https://huggingface.co/MCG-NJU/videomae-large)
*   **Multimodal**
    *   Audio or text to text with [Qwen2-Audio](https://huggingface.co/Qwen/Qwen2-Audio-7B)
    *   Document question answering with [LayoutLMv3](https://huggingface.co/microsoft/layoutlmv3-base)
    *   Image or text to text with [Qwen-VL](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
    *   Image captioning [BLIP-2](https://huggingface.co/Salesforce/blip2-opt-2.7b)
    *   OCR-based document understanding with [GOT-OCR2](https://huggingface.co/stepfun-ai/GOT-OCR-2.0-hf)
    *   Table question answering with [TAPAS](https://huggingface.co/google/tapas-base)
    *   Unified multimodal understanding and generation with [Emu3](https://huggingface.co/BAAI/Emu3-Gen)
    *   Vision to text with [Llava-OneVision](https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-ov-hf)
    *   Visual question answering with [Llava](https://huggingface.co/llava-hf/llava-1.5-7b-hf)
    *   Visual referring expression segmentation with [Kosmos-2](https://huggingface.co/microsoft/kosmos-2-patch14-224)
*   **NLP**
    *   Masked word completion with [ModernBERT](https://huggingface.co/answerdotai/ModernBERT-base)
    *   Named entity recognition with [Gemma](https://huggingface.co/google/gemma-2-2b)
    *   Question answering with [Mixtral](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1)
    *   Summarization with [BART](https://huggingface.co/facebook/bart-large-cnn)
    *   Translation with [T5](https://huggingface.co/google-t5/t5-base)
    *   Text generation with [Llama](https://huggingface.co/meta-llama/Llama-3.2-1B)
    *   Text classification with [Qwen](https://huggingface.co/Qwen/Qwen2.5-0.5B)

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