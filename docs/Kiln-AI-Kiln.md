<p align="center">
    <a href="https://getkiln.ai">
        <picture>
            <img width="205" alt="Kiln AI Logo" src="https://github.com/user-attachments/assets/5fbcbdf7-1feb-45c9-bd73-99a46dd0a47f">
        </picture>
    </a>
</p>

# Kiln: The Ultimate AI Prototyping and Dataset Collaboration Tool

Kiln empowers you to rapidly prototype, fine-tune, and collaborate on AI projects with its intuitive tools and collaborative features. **[Explore the Kiln Repository](https://github.com/Kiln-AI/Kiln)**

## Key Features

*   **🚀 Intuitive Desktop Apps:** Easy-to-use desktop apps for Windows, macOS, and Linux for a seamless AI experience.
*   **🎛️ Fine-tuning:** Fine-tune Llama, GPT-4o, and more with zero code. Automated serverless model deployment.
*   **📊 Evals:** Evaluate your models using state-of-the-art evaluators.
*   **🤖 Synthetic Data Generation:** Create training data with our visual tools.
*   **🧠 Reasoning Models:** Train custom reasoning models.
*   **🤝 Team Collaboration:** Git-based version control for AI datasets. Intuitive UI for QA, PM, and SMEs.
*   **📝 Prompt Generation:** Generate prompts from your data (chain-of-thought, few-shot, multi-shot, etc.).
*   **🌐 Wide Model and Provider Support:** Use any model via Ollama, OpenAI, OpenRouter, Fireworks, Groq, AWS, or any OpenAI-compatible API.
*   **🧑‍💻 Open-Source Library and API:** MIT-licensed Python library and OpenAPI REST API for flexibility.
*   **🔒 Privacy-First:** Bring your API keys or run locally with Ollama. We don't see your data!
*   **🗃️ Structured Data:** Build AI tasks that speak JSON.
*   **💰 Free:** Free apps and an open-source library.

## Download Kiln Desktop Apps

Get started with Kiln today!

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

Get detailed information on using Kiln.

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

For developers, check out the [Kiln Python Library Docs](https://kiln-ai.github.io/Kiln/kiln_core_docs/kiln_ai.html).

## Install Python Library

[![PyPI - Version](https://img.shields.io/pypi/v/kiln-ai.svg?logo=pypi&label=PyPI&logoColor=gold)](https://pypi.org/project/kiln-ai/) [![Docs](https://img.shields.io/badge/docs-pdoc-blue)](https://kiln-ai.github.io/Kiln/kiln_core_docs/index.html)

Integrate Kiln datasets into your workflows with our open-source Python library. [Read the docs](https://kiln-ai.github.io/Kiln/kiln_core_docs/index.html) for examples.

```bash
pip install kiln-ai
```

## Learn More

### Rapid Prototyping

Kiln simplifies trying various AI approaches without writing code for faster, better results.

Current features:

*   Various prompting techniques (basic, few-shot, multi-shot, repair & feedback)
*   Chain of thought with custom "thinking" instructions
*   Model Support: GPT, Llama, Claude, Gemini, Mistral, Gemma, Phi
*   Fine Tuning: create custom models

Future plans include evals and RAG. Advanced users can already create these using Kiln datasets and the Python library.

### Collaborate Across Technical and Non-Technical Teams

Kiln bridges the gap between subject matter experts and technical teams.

Subject matter experts can use the intuitive desktop app for generating structured datasets and ratings.

Data scientists can consume the dataset via the UI or dive deep with the Python library.

QA and PM can quickly identify and resolve issues at the model layer.

The dataset file format works with Git, and collaboration is easy, even for non-technical team members.

### Build High-Quality AI Products with Datasets

Kiln helps you create a dataset for your product. Your model quality improves automatically as the dataset grows, giving models more examples of quality content and mistakes.

Easy iteration enables prompt adaptation to evolving product goals and new bug fixes.

## Contributing & Development

See [CONTRIBUTING.md](CONTRIBUTING.md) to set up your environment and contribute to Kiln.

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
*   Desktop App: Free to use under our [EULA](app/EULA.md), and [source-available](/app). [License](app/LICENSE.txt)
*   The Kiln names and logos are trademarks of Chesterfield Laboratories Inc.

Copyright 2024 - Chesterfield Laboratories Inc.