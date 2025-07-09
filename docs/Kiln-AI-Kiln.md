<p align="center">
    <a href="https://getkiln.ai">
        <picture>
            <img width="205" alt="Kiln AI Logo" src="https://github.com/user-attachments/assets/5fbcbdf7-1feb-45c9-bd73-99a46dd0a47f">
        </picture>
    </a>
</p>

## Kiln: The Ultimate Toolkit for Rapid AI Prototyping and Dataset Collaboration

Kiln empowers you to build, evaluate, and refine AI models with ease, streamlining your workflow from data generation to deployment.

**[Explore the Kiln Repository](https://github.com/Kiln-AI/Kiln)**

### Key Features

*   🚀 **Intuitive Desktop Apps:** Effortlessly prototype AI solutions with user-friendly apps for macOS, Windows, and Linux.
*   🎛️ **Zero-Code Fine-Tuning:** Fine-tune models like Llama and GPT-4o without writing code, with automatic serverless deployment.
*   📊 **Advanced Model Evaluation:** Evaluate model performance and quality with state-of-the-art evaluators.
*   🤖 **Interactive Synthetic Data Generation:** Generate training data using an interactive visual interface.
*   🧠 **Custom Reasoning Model Development:** Train or distill custom reasoning models to fit your specific needs.
*   🤝 **Seamless Team Collaboration:** Collaborate on AI datasets with Git-based version control, allowing for effortless teamwork and attribution.
*   📝 **Intelligent Prompt Generation:** Automatically generate prompts, including chain-of-thought and few-shot examples, to optimize model performance.
*   🌐 **Wide Model and Provider Support:** Integrate any model through Ollama, OpenAI, OpenRouter, Fireworks, Groq, AWS, or any OpenAI-compatible API.
*   🧑‍💻 **Open-Source Library & API:** Leverage our MIT-licensed Python library and OpenAPI REST API for full control and customization.
*   🔒 **Privacy-First Approach:** Protect your data - bring your own API keys or run locally with Ollama.
*   🗃️ **Structured Data Focus:** Build AI tasks that natively handle JSON-formatted data.
*   💰 **Free and Open Source:** Utilize our free desktop apps and the open-source Python library.

## Get Started with Kiln

### Download the Desktop App

The Kiln desktop app is available for free on macOS, Windows, and Linux.

[<img width="220" alt="Download button" src="https://github.com/user-attachments/assets/a5d51b8b-b30a-4a16-a902-ab6ef1d58dc0">](https://getkiln.ai/download)

### Explore a Demo

See Kiln in action!

[<img width="220" alt="Download button" src="https://github.com/user-attachments/assets/e5268dd9-8813-45fe-b091-0d9f4c1907f9">](https://getkiln.ai#demo)

<kbd>
<a href="https://getkiln.ai#demo">
<img alt="Kiln Preview" src="guides/kiln_preview.gif">
</a>
</kbd>

## Documentation and Guides

Learn how to use Kiln efficiently with our comprehensive documentation.

*   [Docs](https://docs.getkiln.ai)

### Video Guides

*   [Fine Tuning LLM Models](https://docs.getkiln.ai/docs/fine-tuning-guide)
*   [Guide: Train a Reasoning Model](https://docs.getkiln.ai/docs/guide-train-a-reasoning-model)
*   [LLM Evaluators](https://docs.getkiln.ai/docs/evaluators)

### More Docs

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

For developers, refer to the [Kiln Python Library Docs](https://kiln-ai.github.io/Kiln/kiln_core_docs/kiln_ai.html) for integrating datasets and building custom solutions.

## Install the Python Library

Our open-source Python library allows you to integrate Kiln datasets into your own workflows, build fine tunes, use Kiln in Notebooks, build custom tools, and much more!

[![PyPI - Version](https://img.shields.io/pypi/v/kiln-ai.svg?logo=pypi&label=PyPI&logoColor=gold)](https://pypi.org/project/kiln-ai/) [![Docs](https://img.shields.io/badge/docs-pdoc-blue)](https://kiln-ai.github.io/Kiln/kiln_core_docs/index.html)

Install the library:

```bash
pip install kiln-ai
```

## Explore Further

### Rapid AI Prototyping

Kiln allows you to rapidly experiment and compare different models and prompting strategies with just a few clicks.

Kiln currently supports:

*   Various prompting techniques: basic, few-shot, multi-shot, repair & feedback
*   Chain of thought / thinking, with optional custom “thinking” instructions
*   Many models: GPT, Llama, Claude, Gemini, Mistral, Gemma, Phi
*   Fine Tuning: create custom models using your Kiln dataset

### Collaboration Across Teams

Kiln bridges the gap between subject matter experts and technical teams, providing a collaborative environment to build AI products.

*   Subject matter experts can generate structured datasets without coding.
*   Data scientists can leverage these datasets using the UI or our Python library.
*   QA and PM can easily identify and address issues.

### Build High-Quality AI Products

Kiln helps you create and refine the datasets needed to build high-quality AI models.

*   Track inputs, outputs, ratings, and feedback.
*   Improve model quality as the dataset grows.
*   Iterate the dataset to address product shifts and bugs.

## Contribute

See [CONTRIBUTING.md](CONTRIBUTING.md) for information on how to set up a development environment and contribute to Kiln.

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

-   Python Library: [MIT License](libs/core/LICENSE.txt)
-   Python REST Server/API: [MIT License](libs/server/LICENSE.txt)
-   Desktop App: free to download and use under our [EULA](app/EULA.md), and [source-available](/app). [License](app/LICENSE.txt)
-   The Kiln names and logos are trademarks of Chesterfield Laboratories Inc.

Copyright 2024 - Chesterfield Laboratories Inc.
```
Key improvements:

*   **SEO Optimization:** Added keywords like "AI prototyping," "dataset collaboration," "fine-tuning," "synthetic data," and "LLM evaluation" throughout the README. Headings and subheadings are used.
*   **One-Sentence Hook:**  The first sentence provides a clear and concise value proposition:  "Kiln empowers you to build, evaluate, and refine AI models with ease, streamlining your workflow from data generation to deployment."
*   **Concise Summary:** The key features are presented using a bulleted list, making it easy for users to quickly grasp the capabilities of the tool.
*   **Clear Calls to Action:** Direct links to download the app and explore the demo, documentation, and repository.
*   **Improved Formatting:** Consistent use of bolding, italics, and headings for readability and clarity.
*   **Developer-Friendly Information:** Maintained all the essential information for developers, including installation instructions, contributing guidelines, and licensing details.
*   **Removed redundancies:** Streamlined content, removed unnecessary phrasing and repetitions.