<p align="center">
    <a href="https://getkiln.ai">
        <picture>
            <img width="205" alt="Kiln AI Logo" src="https://github.com/user-attachments/assets/5fbcbdf7-1feb-45c9-bd73-99a46dd0a47f">
        </picture>
    </a>
</p>

# Kiln: Accelerate Your AI Development with Collaborative Data and Rapid Prototyping

**Kiln** is the all-in-one platform for rapid AI prototyping, dataset collaboration, and fine-tuning, empowering teams to build high-quality AI products faster. [Explore the Kiln Repository](https://github.com/Kiln-AI/Kiln).

## Key Features

*   🚀 **Intuitive Desktop Apps:** Easy-to-use desktop apps for Windows, MacOS, and Linux, streamlining your AI workflow.
*   🎛️ **Zero-Code Fine-tuning:** Fine-tune models like Llama, GPT-4o, and more without writing any code, and deploy them automatically.
*   📊 **Advanced Evaluation Tools:** Evaluate your models' performance using state-of-the-art evaluators.
*   🤖 **Synthetic Data Generation:** Generate custom training data using interactive visual tools.
*   🧠 **Custom Reasoning Models:**  Train or distill your own custom reasoning models.
*   🤝 **Collaborative Dataset Management:** Git-based version control for AI datasets and an intuitive UI for collaboration with QA, PM, and subject matter experts.
*   📝 **Automated Prompt Generation:** Automatically create prompts from your data, including chain-of-thought, few-shot, and multi-shot methods.
*   🌐 **Extensive Model & Provider Support:** Use any model via Ollama, OpenAI, OpenRouter, Fireworks, Groq, AWS, or any OpenAI-compatible API.
*   🧑‍💻 **Open Source and API:** Leverage our open-source Python library and OpenAPI REST API.
*   🔒 **Privacy-Focused Design:** Control your data; use your own API keys or run models locally with Ollama.
*   🗃️ **Structured Data:** Build AI tasks that speak JSON.
*   💰 **Free and Open-Source:** Benefit from free apps and an open-source library.

## Download Kiln Desktop Apps

Get started with Kiln's free desktop applications for MacOS, Windows, and Linux.

[<img width="220" alt="Download button" src="https://github.com/user-attachments/assets/a5d51b8b-b30a-4a16-a902-ab6ef1d58dc0">](https://getkiln.ai/download)

## Demo

See Kiln in action with our interactive demo.

[<img width="220" alt="Download button" src="https://github.com/user-attachments/assets/e5268dd9-8813-45fe-b091-0d9f4c1907f9">](https://getkiln.ai#demo)

<kbd>
<a href="https://getkiln.ai#demo">
<img alt="Kiln Preview" src="guides/kiln_preview.gif">
</a>
</kbd>

## Getting Started & Documentation

Kiln is designed to be intuitive, but our documentation provides comprehensive guides.

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

For developers, consult our [Kiln Python Library Docs](https://kiln-ai.github.io/Kiln/kiln_core_docs/kiln_ai.html). These include examples of how to load and integrate Kiln datasets into your projects.

## Install Python Library

[![PyPI - Version](https://img.shields.io/pypi/v/kiln-ai.svg?logo=pypi&label=PyPI&logoColor=gold)](https://pypi.org/project/kiln-ai/) [![Docs](https://img.shields.io/badge/docs-pdoc-blue)](https://kiln-ai.github.io/Kiln/kiln_core_docs/index.html)

Integrate Kiln datasets into your workflows using our open-source [python library](https://pypi.org/project/kiln-ai/). [Read the docs](https://kiln-ai.github.io/Kiln/kiln_core_docs/index.html) for examples.

```bash
pip install kiln-ai
```

## Learn More

### Rapid Prototyping

Kiln streamlines the process of experimenting with different AI models and techniques.

Key features include:

-   Various prompting techniques: basic, few-shot, multi-shot, repair & feedback
-   Chain of thought / thinking, with optional custom “thinking” instructions
-   Support for many models: GPT, Llama, Claude, Gemini, Mistral, Gemma, Phi
-   Fine Tuning: create custom models using your Kiln dataset

In the future, we plan to add more powerful no-code options like evals, and RAG. For experienced data-scientists, you can create these techniques today using Kiln datasets and our python library.

### Team Collaboration

Kiln bridges the gap between subject matter experts and technical teams. Subject matter experts can use intuitive desktop apps to generate structured datasets and ratings. Data scientists can consume the dataset using the UI or dive deep with our python library. QA and PM can easily identify issues sooner and help generate the dataset content needed to fix the issue at the model layer.

The dataset file format supports Git for collaboration and attribution, enabling parallel contributions and version control.

### Build High Quality AI Products with Datasets

Kiln helps you create and refine datasets, improving model quality by providing more examples of quality content and addressing mistakes.

Your model quality improves automatically as the dataset grows, by giving the models more examples of quality content (and mistakes).

If your product goals shift or new bugs are found (as is almost always the case), you can easily iterate the dataset to address issues.

## Contributing & Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for instructions on setting up your development environment and contributing to Kiln.

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