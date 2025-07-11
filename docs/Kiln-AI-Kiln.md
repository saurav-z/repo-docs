<p align="center">
    <a href="https://getkiln.ai">
        <picture>
            <img width="205" alt="Kiln AI Logo" src="https://github.com/user-attachments/assets/5fbcbdf7-1feb-45c9-bd73-99a46dd0a47f">
        </picture>
    </a>
</p>

# Kiln: The AI Toolkit for Rapid Prototyping and Collaborative Dataset Building

**Kiln is your all-in-one solution for rapidly prototyping AI applications, collaborating on datasets, and building high-quality AI models.**  [Explore the Kiln Repository](https://github.com/Kiln-AI/Kiln)

## Key Features

*   🚀 **Intuitive Desktop Apps:** One-click apps for Windows, MacOS, and Linux provide a user-friendly experience.
*   🎛️ **Zero-Code Fine-Tuning:** Easily fine-tune models like Llama and GPT-4o without writing code and deploy them serverlessly.
*   📊 **Advanced Evals:** Evaluate your models' performance with state-of-the-art evaluators.
*   🤖 **Synthetic Data Generation:** Create training data with our intuitive, visual tools.
*   🧠 **Custom Reasoning Models:** Train or distill your own reasoning models tailored to your needs.
*   🤝 **Team Collaboration:** Utilize Git-based version control for collaborative AI dataset development.
*   📝 **Intelligent Prompt Generation:** Automate prompt creation, including chain-of-thought, few-shot, and multi-shot techniques.
*   🌐 **Wide Model and Provider Support:** Works with various models via Ollama, OpenAI, OpenRouter, Fireworks, Groq, AWS, and OpenAI-compatible APIs.
*   🧑‍💻 **Open-Source Library & API:** Benefit from our MIT-licensed Python library and OpenAPI REST API.
*   🔒 **Privacy-Focused:** Maintain control of your data with options for self-hosting or using your own API keys.
*   🗃️ **Structured Data:** Build AI tasks using JSON for consistent data formats.
*   💰 **Completely Free:** Access all the functionality of the apps and library without any cost.

## Download Kiln Desktop Apps

Get started with Kiln today!  The desktop apps are available for free on MacOS, Windows, and Linux.

[<img width="220" alt="Download button" src="https://github.com/user-attachments/assets/a5d51b8b-b30a-4a16-a902-ab6ef1d58dc0">](https://getkiln.ai/download)

## Demo

See Kiln in action!

[<img width="220" alt="Download button" src="https://github.com/user-attachments/assets/e5268dd9-8813-45fe-b091-0d9f4c1907f9">](https://getkiln.ai#demo)

<kbd>
<a href="https://getkiln.ai#demo">
<img alt="Kiln Preview" src="guides/kiln_preview.gif">
</a>
</kbd>

## Documentation & Guides

Explore our comprehensive documentation to learn more about Kiln's features and capabilities.

### Video Guides

*   [Fine Tuning LLM Models](https://docs.getkiln.ai/docs/fine-tuning-guide)
*   [Guide: Train a Reasoning Model](https://docs.getkiln.ai/docs/guide-train-a-reasoning-model)
*   [LLM Evaluators](https://docs.getkiln.ai/docs/evaluators)

### Comprehensive Guides

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

For developers, check out the [Kiln Python Library Docs](https://kiln-ai.github.io/Kiln/kiln_core_docs/kiln_ai.html) for information on using Kiln datasets in your own codebases.

## Install Python Library

[![PyPI - Version](https://img.shields.io/pypi/v/kiln-ai.svg?logo=pypi&label=PyPI&logoColor=gold)](https://pypi.org/project/kiln-ai/) [![Docs](https://img.shields.io/badge/docs-pdoc-blue)](https://kiln-ai.github.io/Kiln/kiln_core_docs/index.html)

Integrate Kiln datasets seamlessly into your workflows with our open-source Python library.

```bash
pip install kiln-ai
```

## Learn More

### AI Prototyping Made Easy

Kiln streamlines the process of experimenting with different AI models and techniques.  Easily compare approaches with a few clicks, rapidly improving model performance and results.

Kiln supports:

*   Various prompting strategies: basic, few-shot, multi-shot, repair, and feedback
*   Chain of thought / thinking, with custom "thinking" instructions
*   A wide range of models: GPT, Llama, Claude, Gemini, Mistral, Gemma, Phi
*   Fine-tuning: Build your own custom models with your Kiln dataset

We plan to add even more powerful no-code features such as evals, and RAG.  Use Kiln datasets and the python library to create these today.

### Collaboration Across Teams

Kiln bridges the gap between subject matter experts and technical teams, facilitating seamless AI product development.

*   Subject matter experts can easily generate structured datasets and ratings through our intuitive desktop apps.
*   Data scientists can work with the datasets, utilizing the UI or our Python library.
*   QA and PM teams can identify and address issues efficiently.

Utilize Git for enhanced collaboration and attribution.  Multiple team members can contribute in parallel.

### Build High-Quality AI Products

Kiln helps you build high-quality AI models by providing a comprehensive toolkit for creating high quality datasets.

*   Model quality improves automatically as the dataset grows, with the addition of quality content and corrections.
*   Quickly iterate on your dataset to address changing product goals or bugs.

## Contributing & Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for instructions on setting up a development environment and contributing to Kiln.

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
Key improvements and explanations:

*   **SEO Optimization:**  The title is now a keyword-rich headline.  The headings use more descriptive and relevant keywords like "AI toolkit," "Rapid Prototyping," and "Collaborative Dataset." Keywords are repeated naturally throughout.
*   **Concise Hook:** The first sentence is a strong, attention-grabbing sentence that clearly states what Kiln does.
*   **Clear Structure:** The README is organized with clear headings and subheadings, making it easy to scan.
*   **Bulleted Lists:** Key features are presented in concise, easy-to-read bullet points.
*   **Action-Oriented Language:**  Uses verbs like "Explore," "Download," and "Get Started" to encourage engagement.
*   **Emphasis on Benefits:** The "Learn More" section focuses on *benefits* – what Kiln helps users achieve, not just features.
*   **Call to Action:** Clear calls to action ("Download Kiln Desktop Apps," "Explore our comprehensive documentation").
*   **Comprehensive Documentation Links:** Links to the docs are included throughout.
*   **Clean Formatting:**  Uses consistent formatting (bolding, code blocks) for improved readability.
*   **GitHub Link:**  The original repo link is included.
*   **Updated Badges:** No changes to the original badges.
*   **Updated Demo Image:** No changes to the demo gif.
*   **Includes all info from original readme:** No information was left out.