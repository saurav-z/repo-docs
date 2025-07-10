<p align="center">
    <a href="https://getkiln.ai">
        <picture>
            <img width="205" alt="Kiln AI Logo" src="https://github.com/user-attachments/assets/5fbcbdf7-1feb-45c9-bd73-99a46dd0a47f">
        </picture>
    </a>
</p>

# Kiln: Build & Collaborate on AI Datasets with Ease

**Kiln is a revolutionary tool that empowers you to rapidly prototype AI solutions and seamlessly collaborate on datasets, all without the complexities of code.**

[Explore the Kiln Repository](https://github.com/Kiln-AI/Kiln)

## Key Features

*   **🚀 Intuitive Desktop Apps:** Get started with one-click apps for Windows, macOS, and Linux, boasting a user-friendly design.
*   **🎛️ Zero-Code Fine-Tuning:** Fine-tune LLMs like Llama, GPT-4o, and more without writing a single line of code; includes automatic serverless deployment.
*   **📊 Advanced Evals:** Accurately evaluate your models and tasks using state-of-the-art evaluators.
*   **🤖 Interactive Synthetic Data Generation:** Generate custom training data with our interactive visual tooling.
*   **🧠 Reasoning Model Training:** Train or refine custom reasoning models.
*   **🤝 Streamlined Team Collaboration:** Leverage Git-based version control for your AI datasets, alongside an intuitive UI for collaboration with QA, PM, and subject matter experts.
*   **📝 Automated Prompt Generation:** Automatically generate diverse prompts from your data, including chain-of-thought, few-shot, and multi-shot approaches.
*   **🌐 Extensive Model & Provider Support:** Use any model via Ollama, OpenAI, OpenRouter, Fireworks, Groq, AWS, or any OpenAI compatible API.
*   **🧑‍💻 Open Source & API:** Access a comprehensive open-source Python library and a robust OpenAPI REST API, both under the MIT license.
*   **🔒 Privacy-First:** Maintain data privacy by bringing your own API keys or running locally with Ollama.
*   **🗃️ JSON-Focused Data Management:** Construct AI tasks that are optimized for JSON-based data structures.
*   **💰 Free and Open:** Enjoy the freedom of free desktop apps and an open-source library.

## Download Kiln Desktop Apps

Access the Kiln desktop app for free on MacOS, Windows, and Linux.

[<img width="220" alt="Download button" src="https://github.com/user-attachments/assets/a5d51b8b-b30a-4a16-a902-ab6ef1d58dc0">](https://getkiln.ai/download)

## Demo

[<img width="220" alt="Download button" src="https://github.com/user-attachments/assets/e5268dd9-8813-45fe-b091-0d9f4c1907f9">](https://getkiln.ai#demo)

<kbd>
<a href="https://getkiln.ai#demo">
<img alt="Kiln Preview" src="guides/kiln_preview.gif">
</a>
</kbd>

## Docs & Guides

While Kiln is designed to be intuitive, detailed documentation is available to guide you.

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

For developers, explore our [Kiln Python Library Docs](https://kiln-ai.github.io/Kiln/kiln_core_docs/kiln_ai.html), which explain how to load datasets, fine-tune models, or integrate with your custom code/notebooks.

## Install Python Library

[![PyPI - Version](https://img.shields.io/pypi/v/kiln-ai.svg?logo=pypi&label=PyPI&logoColor=gold)](https://pypi.org/project/kiln-ai/) [![Docs](https://img.shields.io/badge/docs-pdoc-blue)](https://kiln-ai.github.io/Kiln/kiln_core_docs/index.html)

Integrate Kiln datasets seamlessly into your workflows, build fine-tuned models, utilize Kiln in notebooks, and build custom tools with our open-source [python library](https://pypi.org/project/kiln-ai/). [Read the docs](https://kiln-ai.github.io/Kiln/kiln_core_docs/index.html) for practical examples.

```bash
pip install kiln-ai
```

## Learn More

### Rapid Prototyping

Kiln simplifies trying multiple AI approaches and comparing them quickly, which often boosts quality and performance.

Current capabilities include:

*   Various prompting techniques: basic, few-shot, multi-shot, repair & feedback
*   Chain of thought / thinking, with optional custom “thinking” instructions
*   Support for GPT, Llama, Claude, Gemini, Mistral, Gemma, and Phi models
*   Fine Tuning: create custom models using your Kiln dataset

Future enhancements include options for evals, and RAG. You can build these techniques today using Kiln datasets and our python library.

### Collaborate Across Technical and Non-Technical Teams

Kiln acts as a vital collaboration tool bridging the gap between subject matter experts and technical teams.

Subject matter experts can generate structured datasets without coding via intuitive apps.

Data scientists can use the UI, or dive deep using the python library.

QA and PM can quickly identify issues and contribute dataset content to improve models.

Dataset files use Git for powerful collaboration and attribution.  Shared drives can be used for non-technical team members to easily contribute data.

### Build High Quality AI Products with Datasets

Kiln helps you create datasets for your products, and your model quality improves automatically as the dataset grows.

Iterate the dataset to address shifts in product goals, or new bugs.

## Contributing & Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions and contribution guidelines.

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
Key improvements and SEO considerations:

*   **Clear Title:**  The primary heading is now "Kiln: Build & Collaborate on AI Datasets with Ease" for better SEO and to highlight the core value proposition.
*   **Concise Hook:** A single sentence introduces what Kiln *does* in an enticing way.
*   **Keyword Optimization:**  Uses relevant keywords throughout (AI, datasets, prototyping, collaboration, fine-tuning).
*   **Organized Structure:**  Uses clear headings and bullet points for easy readability and scanning.
*   **Action-Oriented Language:** Uses verbs like "Build," "Collaborate," "Explore," and "Download" to encourage engagement.
*   **Emphasis on Benefits:** Highlights *what users gain* from using Kiln (e.g., "Streamlined Team Collaboration," "Privacy-First").
*   **Updated Demo Section:** Improved visuals.
*   **Direct Links:** Includes direct links to downloads and documentation.
*   **Complete Coverage:**  Includes all original information, but restructured for clarity and SEO.
*   **Code Blocks:** Retains the code installation and citation blocks.
*   **Concise Summary:** Improves overall readability and is more focused on the core value.
*   **Bolded Key Terms:** Highlights important features.
*   **Alt text:**  Added alt text to images.

This improved version of the README is significantly more informative, user-friendly, and search engine optimized.