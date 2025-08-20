<p align="center">
    <a href="https://kiln.tech">
        <picture>
            <img width="205" alt="Kiln AI Logo" src="https://github.com/user-attachments/assets/5fbcbdf7-1feb-45c9-bd73-99a46dd0a47f">
        </picture>
    </a>
</p>

# Kiln: Supercharge Your AI Development with Rapid Prototyping and Collaborative Datasets

**Kiln empowers you to build and refine AI models faster and collaboratively, offering intuitive tools for fine-tuning, dataset creation, and model evaluation.**

[Link to Original Repo](https://github.com/Kiln-AI/Kiln)

## Key Features

*   **🚀 Intuitive Desktop Apps:** One-click apps for Windows, MacOS, and Linux, designed for ease of use.
*   **🎛️ Zero-Code Fine-Tuning:** Fine-tune Llama, GPT-4o, and more without writing any code, with automatic serverless deployment.
*   **📊 Powerful Evaluations:** Assess and compare model performance using state-of-the-art evaluators.
*   **🤖 Synthetic Data Generation:** Generate diverse and representative training data with interactive visual tools.
*   **🧠 Custom Reasoning Models:** Train or distill custom reasoning models tailored to your needs.
*   **🤝 Collaborative Workflows:** Leverage Git-based version control for your AI datasets, enabling seamless collaboration with QA, PMs, and subject matter experts.
*   **📝 Automated Prompt Engineering:** Generate prompts from your data, including chain-of-thought, few-shot, and multi-shot strategies.
*   **🌐 Extensive Model & Provider Support:** Integrate with any model via Ollama, OpenAI, OpenRouter, Fireworks, Groq, AWS, or any OpenAI-compatible API.
*   **🧑‍💻 Open-Source Library & API:** Access our MIT-licensed Python library and OpenAPI REST API for flexible integration.
*   **🔒 Privacy-Focused:** Maintain complete control over your data with bring-your-own-key options and local execution via Ollama.
*   **🗃️ Structured Data Support:** Build AI tasks that work seamlessly with structured JSON data.
*   **💰 Free to Use:** Kiln's apps are free, and our library is open-source, making AI development accessible.

## Download Kiln Desktop Apps

Get started with Kiln's user-friendly desktop app, available for MacOS, Windows, and Linux.

[<img width="220" alt="Download button" src="https://github.com/user-attachments/assets/a5d51b8b-b30a-4a16-a902-ab6ef1d58dc0">](https://kiln.tech/download)

## Demo

See Kiln in action!

[<img width="220" alt="Download button" src="https://github.com/user-attachments/assets/e5268dd9-8813-45fe-b091-0d9f4c1907f9">](https://kiln.tech#demo)

<kbd>
<a href="https://kiln.tech#demo">
<img alt="Kiln Preview" src="guides/kiln_preview.gif">
</a>
</kbd>

## Documentation & Guides

Explore our comprehensive documentation to learn more about Kiln's features and how to use them effectively. Our resources are designed to get you up and running quickly.

[Docs](https://docs.getkiln.ai)

### Video Guides

*   [Fine Tuning LLM Models](https://docs.getkiln.ai/docs/fine-tuning-guide)
*   [Guide: Train a Reasoning Model](https://docs.getkiln.ai/docs/guide-train-a-reasoning-model)
*   [LLM Evaluators](https://docs.getkiln.ai/docs/evaluators)

### Detailed Documentation

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

### For Developers: Kiln Python Library Docs

Learn how to load datasets into Kiln, and use Kiln datasets in your own code-base/notebooks using our [Kiln Python Library Docs](https://kiln-ai.github.io/Kiln/kiln_core_docs/kiln_ai.html).

## Install Python Library

[![PyPI - Version](https://img.shields.io/pypi/v/kiln-ai.svg?logo=pypi&label=PyPI&logoColor=gold)](https://pypi.org/project/kiln-ai/) [![Docs](https://img.shields.io/badge/docs-pdoc-blue)](https://kiln-ai.github.io/Kiln/kiln_core_docs/index.html)

Integrate Kiln datasets seamlessly into your workflows with our open-source [python library](https://pypi.org/project/kiln-ai/). Fine-tune models, create custom tools, and more!

```bash
pip install kiln-ai
```

## Rapid Prototyping & AI Development

Kiln streamlines the AI development process, allowing you to experiment with various models and techniques efficiently.
*   **Prompting Techniques:** Basic, few-shot, multi-shot, repair & feedback.
*   **Chain of Thought:** With optional custom "thinking" instructions.
*   **Model Support:** GPT, Llama, Claude, Gemini, Mistral, Gemma, Phi.
*   **Fine-Tuning:** Create custom models using your Kiln dataset.

We will add more powerful no-code options like evals and RAG in the future.

## Collaboration for AI Teams

Kiln bridges the gap between subject matter experts and technical teams, making AI product development collaborative.
*   Subject matter experts can create structured datasets and ratings without coding.
*   Data scientists can utilize these datasets.
*   QA and PM can identify and address issues quickly.

The dataset format is designed for Git-based version control, supporting parallel contributions and attribution.

## Building High-Quality AI Products with Datasets

Kiln facilitates dataset creation, capturing inputs, outputs, ratings, and feedback. This improves model quality, allowing you to iterate as your product evolves and refine your AI models.

## Contribute & Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for setting up a development environment and contributing.

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
```
Key improvements:

*   **SEO Optimization:**  The title is descriptive and includes relevant keywords ("AI Development," "Rapid Prototyping," "Collaborative Datasets").  Headings are used for structure.
*   **Concise Hook:** The one-sentence hook effectively summarizes Kiln's core value.
*   **Bulleted Key Features:**  Uses a bulleted list for readability and easy scanning.  Added emojis for visual appeal.
*   **Clear Calls to Action:**  Download and Demo buttons are prominent.
*   **Well-Organized Structure:**  Sections are logically arranged and easy to follow.
*   **Comprehensive Content:**  Includes all the important information from the original README.
*   **Link to Original Repo:** The link to the original repo is maintained.
*   **Developer Focused:**  The install and contribution sections are retained.
*   **Concise Language:** Removed redundant phrases and streamlined the writing.
*   **Emphasis on Benefits:** Focused on what users gain by using Kiln.
*   **Better Visuals:** Improved the formatting of the demonstration preview with a direct link.
*   **Better Information Architecture:** grouped similar content.
*   **License and Trademarks:** Maintained as requested.