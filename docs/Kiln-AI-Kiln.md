<p align="center">
    <a href="https://getkiln.ai">
        <picture>
            <img width="205" alt="Kiln AI Logo" src="https://github.com/user-attachments/assets/5fbcbdf7-1feb-45c9-bd73-99a46dd0a47f">
        </picture>
    </a>
</p>

# Kiln: Rapid AI Prototyping and Dataset Collaboration

**Kiln is the all-in-one platform for rapidly prototyping AI solutions and streamlining dataset collaboration for teams.**  ([See the original repo](https://github.com/Kiln-AI/Kiln))

## Key Features

*   **Intuitive Desktop Apps**: One-click apps for Windows, MacOS, and Linux with a truly intuitive design.
*   **Effortless Fine-Tuning**: Zero-code fine-tuning for popular models like Llama, GPT-4o, and more, with automatic serverless deployment.
*   **Advanced Evaluations**: Evaluate your models using state-of-the-art evaluators.
*   **Interactive Synthetic Data Generation**: Generate training data visually with our interactive tools.
*   **Custom Reasoning Models**: Train or distill custom reasoning models tailored to your needs.
*   **Seamless Team Collaboration**: Git-based version control for your AI datasets with an intuitive UI for collaboration.
*   **Intelligent Prompt Generation**: Generate prompts automatically from your data, including chain-of-thought, few-shot, multi-shot, and more.
*   **Broad Model & Provider Support**: Utilize any model via Ollama, OpenAI, OpenRouter, Fireworks, Groq, AWS, or any OpenAI compatible API.
*   **Open-Source Library & API**: Leverage our MIT-licensed Python library and OpenAPI REST API for powerful customization.
*   **Privacy-Focused Design**:  Bring your own API keys or run locally with Ollama to keep your data secure.
*   **Structured Data**: Build AI tasks that speak JSON.
*   **Completely Free**: The Kiln apps are free to use, and the Python library is open-source.

## Get Started with Kiln

### Download the Desktop App

The Kiln desktop app is completely free. Available on MacOS, Windows and Linux.

[<img width="220" alt="Download button" src="https://github.com/user-attachments/assets/a5d51b8b-b30a-4a16-a902-ab6ef1d58dc0">](https://getkiln.ai/download)

### See a Quick Demo

[<img width="220" alt="Download button" src="https://github.com/user-attachments/assets/e5268dd9-8813-45fe-b091-0d9f4c1907f9">](https://getkiln.ai#demo)

<kbd>
<a href="https://getkiln.ai#demo">
<img alt="Kiln Preview" src="guides/kiln_preview.gif">
</a>
</kbd>

## Docs & Guides

Kiln is intuitive, but for deeper dives, explore our comprehensive [documentation](https://docs.getkiln.ai).

### Featured Guides
*   [Fine Tuning LLM Models](https://docs.getkiln.ai/docs/fine-tuning-guide)
*   [Train a Reasoning Model](https://docs.getkiln.ai/docs/guide-train-a-reasoning-model)
*   [LLM Evaluators](https://docs.getkiln.ai/docs/evaluators)

### All Docs

-   [Quick Start](https://docs.getkiln.ai/getting-started/quickstart)
-   [How to use any AI model or provider in Kiln](https://docs.getkiln.ai/docs/models-and-ai-providers)
-   [Reasoning & Chain of Thought](https://docs.getkiln.ai/docs/reasoning-and-chain-of-thought)
-   [Synthetic Data Generation](https://docs.getkiln.ai/docs/synthetic-data-generation)
-   [Collaborating with Kiln](https://docs.getkiln.ai/docs/collaboration)
-   [Rating and Labeling Data](https://docs.getkiln.ai/docs/reviewing-and-rating)
-   [Prompt Styles](https://docs.getkiln.ai/docs/prompts)
-   [Structure Data / JSON](https://docs.getkiln.ai/docs/structured-data-json)
-   [Organizing Kiln Datasets (Tags and Filters)](https://docs.getkiln.ai/docs/organizing-datasets)
-   [Our Data Model](https://docs.getkiln.ai/docs/kiln-datamodel)
-   [Repairing Responses](https://docs.getkiln.ai/docs/repairing-responses)
-   [Keyboard Shortcuts](https://docs.getkiln.ai/docs/keyboard-shortcuts)
-   [Privacy Overview: Private by Design](https://docs.getkiln.ai/docs/privacy)

For developers, explore our [Kiln Python Library Docs](https://kiln-ai.github.io/Kiln/kiln_core_docs/kiln_ai.html). These include how to load datasets into Kiln, or using Kiln datasets in your own code-base/notebooks.

## Install the Python Library

[![PyPI - Version](https://img.shields.io/pypi/v/kiln-ai.svg?logo=pypi&label=PyPI&logoColor=gold)](https://pypi.org/project/kiln-ai/) [![Docs](https://img.shields.io/badge/docs-pdoc-blue)](https://kiln-ai.github.io/Kiln/kiln_core_docs/index.html)

Integrate Kiln datasets into your workflows and customize your AI projects with our open-source [Python library](https://pypi.org/project/kiln-ai/). Learn more through our [documentation](https://kiln-ai.github.io/Kiln/kiln_core_docs/index.html).

```bash
pip install kiln-ai
```

## Why Use Kiln?

### Accelerate Prototyping

Kiln simplifies trying new AI techniques, enabling rapid experimentation and comparison without coding. This leads to improved results and higher quality.

Kiln currently supports:

*   Prompting techniques: basic, few-shot, multi-shot, repair & feedback
*   Chain of thought / thinking, with optional custom “thinking” instructions
*   Many models: GPT, Llama, Claude, Gemini, Mistral, Gemma, Phi
*   Fine Tuning: create custom models using your Kiln dataset

### Foster Team Collaboration

Kiln connects technical and non-technical team members to bridge gaps in building AI products.

Subject matter experts use our desktop apps to create structured datasets without code.

Data scientists can consume those datasets, via the UI or Python library.

QA and PM can quickly identify issues and contribute dataset content.

### Build High-Quality AI Products with Datasets

Kiln helps you create a dynamic dataset that improves model quality over time.

*   Capture all inputs, outputs, ratings, and feedback to build high-quality models.
*   Model quality automatically increases as the dataset grows, by providing more examples of quality content (and mistakes).
*   Quickly iterate the dataset to address product goal shifts and new bugs.

## Contributing & Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for setting up your development environment.

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