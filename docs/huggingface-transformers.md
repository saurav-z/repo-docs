<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/transformers-logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/transformers-logo-light.svg">
    <img alt="Hugging Face Transformers Library" src="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/transformers-logo-light.svg" width="352" height="59" style="max-width: 100%;">
  </picture>
  <br/>
  <br/>
</p>

## Supercharge Your AI Projects with Hugging Face Transformers

**Harness the power of cutting-edge AI models with the Hugging Face Transformers library, your one-stop shop for state-of-the-art natural language processing, computer vision, audio, video, and multimodal AI.** This library provides easy-to-use tools for both inference and training, empowering you to build and deploy AI solutions with speed and efficiency.  Explore the [original repository](https://github.com/huggingface/transformers) for more details.

**Key Features:**

*   **Wide Variety of Models:** Access a vast library of pre-trained models for diverse tasks, including text generation, image classification, speech recognition, and more.
*   **Simplified Usage:** Leverage a unified API and intuitive pipelines for easy model integration and deployment, regardless of the framework.
*   **Cross-Framework Compatibility:** Seamlessly switch between PyTorch, TensorFlow, and Flax for training and inference.
*   **Efficiency and Cost Savings:** Benefit from pre-trained models to reduce compute costs and accelerate project timelines.
*   **Community-Driven:** Benefit from the vibrant community and a wealth of resources to customize models and adapt them to your specific needs.

### Installation

Transformers supports Python 3.9+ with PyTorch, TensorFlow, and Flax. Here's how to install it:

**Install with pip:**
```bash
pip install "transformers[torch]"
```

**Or with uv:**
```bash
uv pip install "transformers[torch]"
```

For the latest changes, or if you want to contribute, install from source:

```bash
git clone https://github.com/huggingface/transformers.git
cd transformers
pip install .[torch] # or uv pip install .[torch]
```

### Quickstart with Pipeline API

The `Pipeline` API is a high-level, user-friendly tool for interacting with pre-trained models. Here's a simple example:

```python
from transformers import pipeline

# Text Generation
generator = pipeline(task="text-generation", model="Qwen/Qwen2.5-1.5B")
result = generator("The secret to baking a really good cake is")
print(result)
```

### Why Choose Transformers?

*   **Easy to Use:** Get started quickly with state-of-the-art models.
*   **Cost-Effective:** Reduce training time and costs with pre-trained models.
*   **Flexible:** Train, evaluate, and deploy models with the framework that suits you best.
*   **Customizable:** Tailor models and examples to your specific project needs.

### Example Models

*   **Audio:** Speech recognition, audio classification, and more.
*   **Computer Vision:** Image classification, object detection, and segmentation.
*   **Multimodal:** Image captioning, visual question answering.
*   **NLP:** Text generation, summarization, translation, and question answering.

### Additional Resources

*   [Checkpoints on Hub](https://huggingface.co/models)
*   [Documentation](https://huggingface.co/docs/transformers/index)
*   [100 Projects](https://github.com/huggingface/transformers/blob/main/awesome-transformers.md)
*   [Citation](https://www.aclweb.org/anthology/2020.emnlp-demos.6/)