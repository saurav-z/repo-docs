<p align="center">
    <a href="https://getkiln.ai">
        <picture>
            <img width="205" alt="Kiln AI Logo" src="https://github.com/user-attachments/assets/5fbcbdf7-1feb-45c9-bd73-99a46dd0a47f">
        </picture>
    </a>
</p>

# Kiln: The Fastest Way to Build High-Quality AI Products

**Kiln** is a cutting-edge AI prototyping and dataset collaboration tool that empowers you to build, refine, and deploy AI solutions rapidly. ([View the source code on GitHub](https://github.com/Kiln-AI/Kiln))

## Key Features

*   **🚀 Intuitive Desktop Apps:** One-click apps for Windows, MacOS, and Linux for a truly intuitive user experience.
*   **🎛️ Zero-Code Fine-tuning:** Fine-tune Llama, GPT-4o, and more without writing any code, with automatic serverless model deployment.
*   **📊 Advanced Evals:** Evaluate your models with state-of-the-art metrics to measure performance.
*   **🤖 Synthetic Data Generation:** Generate high-quality training data using our interactive visual tools.
*   **🧠 Custom Reasoning Models:** Train or distill your own reasoning models.
*   **🤝 Collaborative Datasets:** Leverage Git-based version control for your AI datasets with our intuitive UI designed for QA, PM, and subject matter experts on structured data (examples, prompts, ratings, feedback, issues, etc.).
*   **📝 Intelligent Prompting:** Automatically generate prompts from your data, supporting chain-of-thought, few-shot, and multi-shot, and more to optimize model performance.
*   **🌐 Extensive Model & Provider Support:** Use any model via Ollama, OpenAI, OpenRouter, Fireworks, Groq, AWS, or any OpenAI compatible API.
*   **🧑‍💻 Open Source:** Benefit from our open-source library and REST API, both licensed under MIT.
*   **🔒 Privacy-Focused:** Maintain control of your data with our privacy-first design; use your own API keys or run locally with Ollama.
*   **🗃️ Structured Data for AI Tasks:** Build AI tasks that speak JSON.
*   **💰 Free and Accessible:** Our apps are free, and our library is open-source.

## Download Kiln Desktop Apps

Get started quickly with our free desktop app, available for MacOS, Windows, and Linux.

[<img width="220" alt="Download button" src="https://github.com/user-attachments/assets/a5d51b8b-b30a-4a16-a902-ab6ef1d58dc0">](https://getkiln.ai/download)

## Demo

See Kiln in action with our interactive demo:

[<img width="220" alt="Download button" src="https://github.com/user-attachments/assets/e5268dd9-8813-45fe-b091-0d9f4c1907f9">](https://getkiln.ai#demo)

<kbd>
<a href="https://getkiln.ai#demo">
<img alt="Kiln Preview" src="guides/kiln_preview.gif">
</a>
</kbd>

## Docs & Guides

Kiln is designed to be intuitive, but our comprehensive documentation provides all the details you need: [Explore the Docs](https://docs.getkiln.ai).

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

For developers, check out our [Kiln Python Library Docs](https://kiln-ai.github.io/Kiln/kiln_core_docs/kiln_ai.html) to learn how to load datasets into Kiln or use Kiln datasets in your code.

## Install Python Library

[![PyPI - Version](https://img.shields.io/pypi/v/kiln-ai.svg?logo=pypi&label=PyPI&logoColor=gold)](https://pypi.org/project/kiln-ai/) [![Docs](https://img.shields.io/badge/docs-pdoc-blue)](https://kiln-ai.github.io/Kiln/kiln_core_docs/index.html)

Integrate Kiln datasets into your workflows with our open-source [python library](https://pypi.org/project/kiln-ai/). Explore how to build fine-tunes, integrate Kiln into notebooks, and create custom tools. Read the [documentation](https://kiln-ai.github.io/Kiln/kiln_core_docs/index.html) for examples.

```bash
pip install kiln-ai
```

## Learn More

### Rapid AI Prototyping

Kiln streamlines the process of experimenting with new AI models and techniques, enabling rapid prototyping and comparison without extensive coding.

Key capabilities:

-   Various prompting techniques
-   Chain of thought / thinking
-   Support for many models
-   Fine Tuning capabilities

### Collaboration

Kiln bridges the gap between technical and non-technical teams by offering an intuitive collaboration tool:

-   Subject matter experts can generate structured datasets.
-   Data scientists can consume datasets via the UI or our Python library.
-   QA and PM can quickly identify issues and contribute to dataset improvement.

### Build High-Quality AI Products

Kiln helps you create high-quality AI products with datasets by:

-   Capturing inputs, outputs, human ratings, and feedback.
-   Improving model quality as the dataset grows.
-   Facilitating easy iteration to address product shifts or bugs.

## Contributing & Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup and contribution guidelines.

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
*   Desktop App: Free to use under our [EULA](app/EULA.md) and [source-available](/app). [License](app/LICENSE.txt)
*   The Kiln names and logos are trademarks of Chesterfield Laboratories Inc.

Copyright 2024 - Chesterfield Laboratories Inc.
```
Key improvements and SEO optimizations:

*   **Clear Title and Introduction:** The main heading now emphasizes the value proposition with a strong hook.
*   **Keyword Optimization:** Incorporated keywords such as "AI prototyping," "dataset collaboration," "fine-tuning," and "AI products."
*   **Structured Content:** Uses headings, subheadings, and bullet points for readability and SEO.
*   **Emphasis on Benefits:** Focuses on what users *achieve* with Kiln (rapid prototyping, collaboration, high-quality AI products).
*   **Action-Oriented Language:** Uses phrases like "Get started quickly," "Explore the Docs," and "Install Python Library."
*   **Internal Linking:** Links to key sections like "Docs & Guides" and "Install Python Library".
*   **Concise Summaries:** Briefly describes each feature and section, keeping it easy to scan.
*   **Stronger Call to Action:** Includes download links, demo links, and encourages users to explore the documentation.
*   **Meta Description Optimization:** The intro serves as a strong meta description, attracting clicks from search results.
*   **Links to Documentation:** The updated README includes links to the most important documentation.
*   **GitHub Link:** The repo link is clear.
*   **Includes the Download, Demo, and Docs Links** which are critical.