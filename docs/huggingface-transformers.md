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
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_pt-br.md">Português</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_te.md">తెలుగు</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_fr.md">Français</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_de.md">Deutsch</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_vi.md">Tiếng Việt</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ar.md">العربية</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ur.md">اردو</a> |
    </p>
</h4>

# Hugging Face Transformers: State-of-the-Art Models for NLP, Computer Vision, and More

**Harness the power of cutting-edge AI with Hugging Face Transformers, a comprehensive library providing pre-trained models and tools for various machine learning tasks.**

<p align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/transformers_as_a_model_definition.png"/>
</p>

Transformers empowers you to build, train, and deploy state-of-the-art models for:

*   **Natural Language Processing (NLP):** Text generation, translation, summarization, question answering, and more.
*   **Computer Vision:** Image classification, object detection, segmentation, and video analysis.
*   **Audio Processing:** Speech recognition, speech synthesis, and audio classification.
*   **Multimodal Tasks:** Combining text, images, and audio for complex applications.

**Key Features:**

*   **Wide Range of Pre-trained Models:** Access over 1 million pre-trained models on the [Hugging Face Hub](https://huggingface.co/models?library=transformers&sort=trending) for various tasks and modalities.
*   **Unified API:** Utilize a consistent and easy-to-use API for all supported models.
*   **Flexibility and Customization:** Adapt models to your specific needs with ease.
*   **Cross-Framework Compatibility:** Seamlessly work with PyTorch, TensorFlow, and Flax.
*   **Community-Driven:** Benefit from a vibrant community and extensive resources.
*   **Efficient & Accessible:** Reduce compute costs and lower the barrier to entry.

**Get Started:**

1.  **Installation:**

    ```bash
    # Install with PyTorch
    pip install "transformers[torch]"

    # Install with TensorFlow
    pip install "transformers[tensorflow]"

    # Install with Flax
    pip install "transformers[flax]"
    ```
    Or from source:
    ```shell
    git clone https://github.com/huggingface/transformers.git
    cd transformers
    pip install .[torch]
    ```

2.  **Quickstart:**

    Use the `pipeline` API for immediate results, supporting text, audio, vision, and multimodal tasks.
    ```py
    from transformers import pipeline

    # Text Generation
    generator = pipeline(task="text-generation", model="Qwen/Qwen2.5-1.5B")
    print(generator("the secret to baking a really good cake is ")

    # Chatbot
    from transformers import pipeline
    import torch

    chat = [
        {"role": "system", "content": "You are a sassy, wise-cracking robot as imagined by Hollywood circa 1986."},
        {"role": "user", "content": "Hey, can you tell me any fun things to do in New York?"}
    ]

    pipe = pipeline(task="text-generation", model="meta-llama/Meta-Llama-3-8B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")
    response = pipe(chat, max_new_tokens=512)
    print(response[0]["generated_text"][-1]["content"])
    ```

    Explore different modalities and tasks with the example pipelines in the original README.

**Why Use Transformers?**

*   **Easy to Use:** State-of-the-art models are easily accessible with high performance.
*   **Cost-Effective:** Share trained models to save resources.
*   **Flexible:** Choose your framework for training, evaluation, and deployment.
*   **Customizable:** Adapt models and examples to your project.

**Resources:**

*   [Hugging Face Hub](https://huggingface.co/models)
*   [Documentation](https://huggingface.co/docs/transformers/index)
*   [Examples](https://github.com/huggingface/transformers/tree/main/examples)
*   [Original Repository](https://github.com/huggingface/transformers)

**Citation:**

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