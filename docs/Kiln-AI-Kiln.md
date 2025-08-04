<p align="center">
    <a href="https://getkiln.ai">
        <picture>
            <img width="205" alt="Kiln AI Logo" src="https://github.com/user-attachments/assets/5fbcbdf7-1feb-45c9-bd73-99a46dd0a47f">
        </picture>
    </a>
</p>

# Kiln: The All-in-One AI Prototyping and Dataset Collaboration Tool

Kiln empowers you to rapidly prototype AI applications and collaborate on datasets with ease.  Find the original repository [here](https://github.com/Kiln-AI/Kiln).

## Key Features

*   **🚀 Intuitive Desktop Apps:**  One-click apps for Windows, macOS, and Linux provide a user-friendly experience.
*   **🎛️ No-Code Fine-Tuning:** Easily fine-tune models like Llama and GPT-4o without writing any code, and deploy them automatically.
*   **📊 Advanced Evals:** Evaluate your models using state-of-the-art evaluators.
*   **🤖 Synthetic Data Generation:** Create high-quality training data effortlessly with our interactive visual tools.
*   **🧠 Custom Reasoning Models:** Train and refine your own custom reasoning models.
*   **🤝 Git-Based Team Collaboration:**  Collaborate seamlessly using Git-based version control for your AI datasets, including structured data (examples, prompts, ratings, feedback, issues).
*   **📝 Prompt Engineering Tools:** Generate prompts automatically from your data, including chain-of-thought, few-shot, and multi-shot techniques.
*   **🌐 Extensive Model and Provider Support:** Integrate with any model via Ollama, OpenAI, OpenRouter, Fireworks, Groq, AWS, or any OpenAI compatible API.
*   **🧑‍💻 Open-Source Library and API:** Access a powerful Python library and OpenAPI REST API under the MIT license.
*   **🔒 Privacy-First Design:** Bring your own API keys or run locally with Ollama, ensuring your data privacy.
*   **🗃️ Structured Data:** Build AI tasks that speak JSON.
*   **💰 Free & Open-Source:** Our apps are free, and our library is open-source.

## Download Kiln Desktop Apps

The Kiln desktop app is completely free and available for MacOS, Windows, and Linux.

[<img width="220" alt="Download button" src="https://github.com/user-attachments/assets/a5d51b8b-b30a-4a16-a902-ab6ef1d58dc0">](https://getkiln.ai/download)

## Demo

[<img width="220" alt="Demo button" src="https://github.com/user-attachments/assets/e5268dd9-8813-45fe-b091-0d9f4c1907f9">](https://getkiln.ai#demo)

<kbd>
<a href="https://getkiln.ai#demo">
<img alt="Kiln Preview" src="guides/kiln_preview.gif">
</a>
</kbd>

## Docs & Guides

Kiln's intuitive design makes getting started a breeze.  However, if you have any questions or would like to learn more, our [docs are here to help](https://docs.getkiln.ai).

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

Our open-source [python library](https://pypi.org/project/kiln-ai/) allows you to integrate Kiln datasets into your own workflows, build fine tunes, use Kiln in Notebooks, build custom tools, and much more! [Read the docs](https://kiln-ai.github.io/Kiln/kiln_core_docs/index.html) for examples.

```bash
pip install kiln-ai
```

## Learn More

### Rapid AI Prototyping

Kiln simplifies the process of experimenting with new AI models and techniques, allowing you to compare different approaches quickly and efficiently, and build better AI applications with better outcomes.

Key functionalities:

*   Basic, few-shot, multi-shot, repair & feedback prompting techniques
*   Chain of thought / thinking, with optional custom “thinking” instructions
*   Support for models: GPT, Llama, Claude, Gemini, Mistral, Gemma, Phi
*   Fine Tuning: create custom models using your Kiln dataset

Future additions: evals, and RAG. For data-scientists, you can create these techniques today using Kiln datasets and our python library.

### Collaborate Across Technical and Non-Technical Teams

Kiln acts as a bridge between subject matter experts, who understand the problem you are trying to solve, and the technical teams responsible for building the model.

Subject matter experts can generate structured datasets and ratings using our intuitive desktop apps without any coding.

Data scientists can use the UI, or dive deep into the data with the python library.

QA and PM teams can leverage Kiln to identify and fix model issues.

The dataset file format is designed to be Git-friendly for efficient collaboration and attribution.  Multiple people can work on the same dataset in parallel; collisions are avoided using UUIDs, and attribution is tracked.

### Build High Quality AI Products with Datasets

Kiln allows you to create and refine a "dataset" for your AI projects, improving your model quality and performance over time by providing it with examples of quality content and identifying and resolving mistakes.

Model quality improves automatically as the dataset grows, by giving the models more examples of quality content (and mistakes).

Iterate the dataset to address shifts in product goals and/or new bugs as needed.

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