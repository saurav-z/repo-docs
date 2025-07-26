<p align="center">
    <a href="https://getkiln.ai">
        <picture>
            <img width="205" alt="Kiln AI Logo" src="https://github.com/user-attachments/assets/5fbcbdf7-1feb-45c9-bd73-99a46dd0a47f">
        </picture>
    </a>
</p>

# Kiln: Accelerate AI Development with Rapid Prototyping and Seamless Dataset Collaboration

Kiln empowers developers and teams to quickly prototype, fine-tune, and collaborate on AI projects with ease. [Explore Kiln on GitHub](https://github.com/Kiln-AI/Kiln).

## Key Features:

*   **Intuitive Desktop Apps:** One-click apps for Windows, MacOS, and Linux with a user-friendly interface.
*   **Fine-tuning:** Zero-code fine-tuning for models like Llama, GPT-4o, and more, with automatic serverless deployment.
*   **Evaluation:** Assess model performance using state-of-the-art evaluators.
*   **Synthetic Data Generation:** Create training data visually with our interactive tools.
*   **Reasoning Models:** Develop custom reasoning models through training or distillation.
*   **Team Collaboration:** Git-based version control for datasets, streamlining collaboration among QA, PM, and subject matter experts.
*   **Prompt Generation:** Automatically generate prompts from your data, including chain-of-thought, few-shot, and multi-shot.
*   **Broad Model & Provider Support:** Use any model via Ollama, OpenAI, OpenRouter, Fireworks, Groq, AWS, or any OpenAI compatible API.
*   **Open-Source Library and API:** Access powerful tools with our MIT-licensed Python library and REST API.
*   **Privacy-Focused:** Bring your own API keys or run locally with Ollama; Kiln doesn't access your data.
*   **Structured Data:** Build AI tasks that use JSON.
*   **Free to Use:** Our desktop apps are free, and our library is open source.

## Get Started

### Download Kiln Desktop Apps

The Kiln desktop app is completely free. Available on MacOS, Windows and Linux.

[<img width="220" alt="Download button" src="https://github.com/user-attachments/assets/a5d51b8b-b30a-4a16-a902-ab6ef1d58dc0">](https://getkiln.ai/download)

### Demo

[<img width="220" alt="Download button" src="https://github.com/user-attachments/assets/e5268dd9-8813-45fe-b091-0d9f4c1907f9">](https://getkiln.ai#demo)

<kbd>
<a href="https://getkiln.ai#demo">
<img alt="Kiln Preview" src="guides/kiln_preview.gif">
</a>
</kbd>

## Docs & Guides

Dive deeper into Kiln with our comprehensive documentation and helpful guides.

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

For developers, see our [Kiln Python Library Docs](https://kiln-ai.github.io/Kiln/kiln_core_docs/kiln_ai.html). These include how to load datasets into Kiln, or using Kiln datasets in your own code-base/notebooks.

## Install Python Library

[![PyPI - Version](https://img.shields.io/pypi/v/kiln-ai.svg?logo=pypi&label=PyPI&logoColor=gold)](https://pypi.org/project/kiln-ai/) [![Docs](https://img.shields.io/badge/docs-pdoc-blue)](https://kiln-ai.github.io/Kiln/kiln_core_docs/index.html)

Integrate Kiln datasets into your workflows with our open-source Python library. Read the [docs](https://kiln-ai.github.io/Kiln/kiln_core_docs/index.html) for examples.

```bash
pip install kiln-ai
```

## Learn More

### Rapid Prototyping

Kiln simplifies experimenting with new AI models and techniques, enabling you to compare approaches quickly and improve performance. Kiln supports:

*   Various prompting techniques: basic, few-shot, multi-shot, repair & feedback
*   Chain of thought / thinking, with optional custom “thinking” instructions
*   Many models: GPT, Llama, Claude, Gemini, Mistral, Gemma, Phi
*   Fine Tuning: create custom models using your Kiln dataset

In the future, we plan to add more powerful no-code options like evals, and RAG. For experienced data-scientists, you can create these techniques today using Kiln datasets and our python library.

### Collaborate Across Technical and Non-Technical Teams

Kiln facilitates collaboration between subject matter experts and technical teams in AI development.

Subject matter experts can use our intuitive desktop apps to generate structured datasets and ratings, without coding or using technical tools.

Data-scientists can consume the dataset created by subject matter experts, using the UI, or deep dive with our python library.

QA and PM can easily identify issues sooner and help generate the dataset content needed to fix the issue at the model layer.

The dataset file format is designed to be be used with Git for powerful collaboration and attribution. Many people can contribute in parallel; collisions are avoided using UUIDs, and attribution is captured inside the dataset files. You can even share a dataset on a shared drive, letting completely non-technical team members contribute data and evals without knowing Git.

### Build High Quality AI Products with Datasets

Kiln helps you create a dataset for your product. Every time you use Kiln, we capture the inputs, outputs, human ratings, feedback, and repairs needed to build high quality models for use in your product. The more you use it, the more data you have.

Your model quality improves automatically as the dataset grows, by giving the models more examples of quality content (and mistakes).

If your product goals shift or new bugs are found (as is almost always the case), you can easily iterate the dataset to address issues.

## Contributing & Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for information on setting up a development environment and contributing.

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