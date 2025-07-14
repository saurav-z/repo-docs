<p align="center">
    <a href="https://getkiln.ai">
        <picture>
            <img width="205" alt="Kiln AI Logo" src="https://github.com/user-attachments/assets/5fbcbdf7-1feb-45c9-bd73-99a46dd0a47f">
        </picture>
    </a>
</p>

# Kiln: The All-in-One Platform for Rapid AI Prototyping and Dataset Collaboration

**Unlock the power of AI with Kiln, the user-friendly platform that accelerates your AI development from prototyping to deployment.** [Visit the original repo](https://github.com/Kiln-AI/Kiln).

## Key Features

*   🚀 **Intuitive Desktop Apps:** Get started in seconds with one-click apps for MacOS, Windows, and Linux.
*   🎛️ **Fine-tuning:** Fine-tune LLMs like Llama and GPT-4o with no code and automatic serverless deployment.
*   📊 **Evals:** Quickly evaluate your AI models/tasks using state-of-the-art evaluators.
*   🤖 **Synthetic Data Generation:** Generate high-quality training data with our interactive visual tooling.
*   🧠 **Reasoning Models:** Train and customize reasoning models to power your AI applications.
*   🤝 **Team Collaboration:** Utilize Git-based version control for effortless collaboration and dataset management.
*   📝 **Prompt Generation:** Automatically generate various prompts including chain-of-thought, few-shot, and multi-shot.
*   🌐 **Wide Model and Provider Support:** Seamlessly integrate with a wide range of models via Ollama, OpenAI, OpenRouter, Fireworks, Groq, AWS, and any OpenAI-compatible API.
*   🧑‍💻 **Open-Source Library and API:** Leverage our open-source Python library and OpenAPI REST API for flexible integration.
*   🔒 **Privacy-First:** Maintain complete control of your data by bringing your own API keys or running locally with Ollama.
*   🗃️ **Structured Data:** Build AI tasks that speak JSON, streamlining your data handling.
*   💰 **Free to Use:** Explore the potential of AI without financial barriers – our apps are free to use, and our library is open-source.

## Download Kiln Desktop Apps

Get started with Kiln's free desktop apps for MacOS, Windows, and Linux.

[<img width="220" alt="Download button" src="https://github.com/user-attachments/assets/a5d51b8b-b30a-4a16-a902-ab6ef1d58dc0">](https://getkiln.ai/download)

## Demo

See Kiln in action!

[<img width="220" alt="Download button" src="https://github.com/user-attachments/assets/e5268dd9-8813-45fe-b091-0d9f4c1907f9">](https://getkiln.ai#demo)

<kbd>
<a href="https://getkiln.ai#demo">
<img alt="Kiln Preview" src="guides/kiln_preview.gif">
</a>
</kbd>

## Docs & Guides

Discover our comprehensive documentation and guides to get the most out of Kiln.

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

For developers, explore our [Kiln Python Library Docs](https://kiln-ai.github.io/Kiln/kiln_core_docs/kiln_ai.html). Learn how to load datasets, or use Kiln datasets in your own code-base/notebooks.

## Install Python Library

[![PyPI - Version](https://img.shields.io/pypi/v/kiln-ai.svg?logo=pypi&label=PyPI&logoColor=gold)](https://pypi.org/project/kiln-ai/) [![Docs](https://img.shields.io/badge/docs-pdoc-blue)](https://kiln-ai.github.io/Kiln/kiln_core_docs/index.html)

Integrate Kiln datasets seamlessly into your workflows with our open-source [python library](https://pypi.org/project/kiln-ai/), for fine-tuning, use in Notebooks, and building custom tools! [Read the docs](https://kiln-ai.github.io/Kiln/kiln_core_docs/index.html) for examples.

```bash
pip install kiln-ai
```

## Learn More

### Rapid Prototyping

Kiln empowers you to experiment and compare various AI approaches effortlessly, speeding up your development cycle.

We currently support:

*   Various prompting techniques: basic, few-shot, multi-shot, repair & feedback
*   Chain of thought / thinking, with optional custom “thinking” instructions
*   Many models: GPT, Llama, Claude, Gemini, Mistral, Gemma, Phi
*   Fine Tuning: create custom models using your Kiln dataset

In the future, we plan to add more powerful no-code options like evals, and RAG. For experienced data-scientists, you can create these techniques today using Kiln datasets and our python library.

### Collaborate Across Technical and Non-Technical Teams

Kiln bridges the gap between subject matter experts and technical teams, facilitating seamless collaboration.

Subject matter experts can use our intuitive desktop apps to generate structured datasets and ratings, without coding or using technical tools.

Data-scientists can leverage the UI, or deep dive with our python library.

QA and PM can easily identify issues sooner and help generate the dataset content needed to fix the issue at the model layer.

The dataset file format is designed to be be used with Git for powerful collaboration and attribution. Many people can contribute in parallel; collisions are avoided using UUIDs, and attribution is captured inside the dataset files. You can even share a dataset on a shared drive, letting completely non-technical team members contribute data and evals without knowing Git.

### Build High Quality AI Products with Datasets

Create high-quality AI models by building datasets within Kiln. Every interaction captures inputs, outputs, human ratings, feedback, and repairs to improve model quality.

Your model quality improves automatically as the dataset grows, by giving the models more examples of quality content (and mistakes).

If your product goals shift or new bugs are found (as is almost always the case), you can easily iterate the dataset to address issues.

## Contributing & Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for information on how to setup a development environment and contribute to Kiln.

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