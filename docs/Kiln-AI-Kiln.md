<p align="center">
    <a href="https://getkiln.ai">
        <picture>
            <img width="205" alt="Kiln AI Logo" src="https://github.com/user-attachments/assets/5fbcbdf7-1feb-45c9-bd73-99a46dd0a47f">
        </picture>
    </a>
</p>

# Kiln: Accelerate AI Development with Rapid Prototyping and Collaborative Datasets

Kiln is a cutting-edge tool that empowers you to rapidly prototype AI solutions and foster seamless collaboration on datasets, making AI development faster and more effective.  [Check out the original repo](https://github.com/Kiln-AI/Kiln) for more details.

## Key Features

*   **Intuitive Desktop Apps**: Get started instantly with user-friendly apps for MacOS, Windows, and Linux.
*   **Zero-Code Fine-Tuning**: Easily fine-tune models like Llama, GPT-4o, and more, with automatic serverless deployment.
*   **Advanced Evaluation**: Utilize state-of-the-art evaluators to assess the quality of your AI models and tasks.
*   **Synthetic Data Generation**: Create tailored training data using our interactive visual tools.
*   **Reasoning Model Training**: Develop and refine custom reasoning models to enhance AI performance.
*   **Git-Based Team Collaboration**: Collaborate seamlessly with your team through Git for dataset version control and management.
*   **Automated Prompt Generation**: Generate effective prompts, including chain-of-thought, few-shot, and multi-shot, directly from your data.
*   **Extensive Model and Provider Support**: Integrate with a wide array of models via Ollama, OpenAI, OpenRouter, Fireworks, Groq, AWS, and any OpenAI compatible API.
*   **Open-Source Library and API**: Leverage our MIT-licensed Python library and OpenAPI REST API for extended functionality.
*   **Privacy-Focused Design**: Maintain data privacy by using your own API keys or running models locally with Ollama.
*   **Structured Data Support**: Build AI tasks that excel with JSON-formatted data.
*   **Free and Accessible**: Enjoy free desktop apps and an open-source Python library, making AI development accessible to everyone.

## Get Started with Kiln

### Download Kiln Desktop Apps

Download the free Kiln desktop app for a seamless AI development experience. Available for MacOS, Windows, and Linux.

[<img width="220" alt="Download button" src="https://github.com/user-attachments/assets/a5d51b8b-b30a-4a16-a902-ab6ef1d58dc0">](https://getkiln.ai/download)

### Demo

Experience Kiln's capabilities with our interactive demo.

[<img width="220" alt="Download button" src="https://github.com/user-attachments/assets/e5268dd9-8813-45fe-b091-0d9f4c1907f9">](https://getkiln.ai#demo)

<kbd>
<a href="https://getkiln.ai#demo">
<img alt="Kiln Preview" src="guides/kiln_preview.gif">
</a>
</kbd>

## Documentation and Guides

Explore comprehensive documentation and guides to master Kiln's features.

### Video Guides

*   [Fine Tuning LLM Models](https://docs.getkiln.ai/docs/fine-tuning-guide)
*   [Guide: Train a Reasoning Model](https://docs.getkiln.ai/docs/guide-train-a-reasoning-model)
*   [LLM Evaluators](https://docs.getkiln.ai/docs/evaluators)

### All Docs

*   [Quick Start](https://docs.getkiln.ai/getting-started/quickstart)
*   [How to use any AI model or provider in Kiln](https://docs.getkiln.ai/docs/models-and-ai-providers)
*   [Reasoning & Chain of Thought](https://docs.getkiln.ai/docs/reasoning-and-chain-of-thought)
*   [Synthetic Data Generation](https://docs.getkiln.ai/docs/synthetic-data-generation)
*   [Collaborating with Kiln](https://docs.getkiln.ai/docs/collaboration)
*   [Rating and Labeling Data](https://docs.getkiln.ai/docs/reviewing-and-rating)
*   [Prompt Styles](https://docs.getkiln.ai/docs/prompts)
*   [Structure Data / JSON](https://docs.getkiln.ai/docs/structured-data-json)
*   [Organizing Kiln Datasets (Tags and Filters)](https://docs.getkiln.ai/docs/organizing-datasets)
*   [Our Data Model](https://docs.getkiln.ai/docs/kiln-datamodel)
*   [Repairing Responses](https://docs.getkiln.ai/docs/repairing-responses)
*   [Keyboard Shortcuts](https://docs.getkiln.ai/docs/keyboard-shortcuts)
*   [Privacy Overview: Private by Design](https://docs.getkiln.ai/docs/privacy)

For developers, check out our [Kiln Python Library Docs](https://kiln-ai.github.io/Kiln/kiln_core_docs/kiln_ai.html).

## Install Python Library

[![PyPI - Version](https://img.shields.io/pypi/v/kiln-ai.svg?logo=pypi&label=PyPI&logoColor=gold)](https://pypi.org/project/kiln-ai/) [![Docs](https://img.shields.io/badge/docs-pdoc-blue)](https://kiln-ai.github.io/Kiln/kiln_core_docs/index.html)

Integrate Kiln datasets seamlessly into your workflows with our open-source [Python library](https://pypi.org/project/kiln-ai/). [Explore the documentation](https://kiln-ai.github.io/Kiln/kiln_core_docs/index.html) for example use cases.

```bash
pip install kiln-ai
```

## Learn More

### Rapid Prototyping

Kiln enables you to quickly experiment with different AI techniques and compare results effortlessly, empowering you to enhance the quality and performance of your models.

Currently supports:

*   Variety of prompting techniques
*   Chain of thought / thinking
*   Many models: GPT, Llama, Claude, Gemini, Mistral, Gemma, Phi
*   Fine Tuning

### Collaborate Across Technical and Non-Technical Teams

Kiln fosters collaboration between subject matter experts and technical teams, streamlining the AI development process.

Subject matter experts can easily generate datasets using the UI.

Data-scientists can consume the dataset created by subject matter experts.

QA and PM can easily identify issues and help generate the dataset content needed to fix the issue at the model layer.

The dataset file format is designed to be be used with Git for powerful collaboration and attribution. Many people can contribute in parallel; collisions are avoided using UUIDs, and attribution is captured inside the dataset files. You can even share a dataset on a shared drive, letting completely non-technical team members contribute data and evals without knowing Git.

### Build High Quality AI Products with Datasets

Kiln helps you create high-quality AI products by enabling you to build datasets, which improve model quality as they grow. As your product evolves, you can easily iterate on the dataset to address any issues.

## Contributing & Development

Refer to [CONTRIBUTING.md](CONTRIBUTING.md) for detailed instructions on setting up your development environment and contributing to Kiln.

## Citation

```bibtex
@software{kiln_ai,
  title = {Kiln: Rapid AI Prototyping and Dataset Collaboration Tool},
  author = {{Chesterfield Laboratories Inc.}},
  year = {2025},
  url = {https://github.com/Kiln-AI/Kiln},
  version = {latest}
}
```

## Licenses & Trademarks

*   Python Library: [MIT License](libs/core/LICENSE.txt)
*   Python REST Server/API: [MIT License](libs/server/LICENSE.txt)
*   Desktop App: free to download and use under our [EULA](app/EULA.md), and [source-available](/app). [License](app/LICENSE.txt)
*   The Kiln names and logos are trademarks of Chesterfield Laboratories Inc.

Copyright 2024 - Chesterfield Laboratories Inc.