<p align="center">
    <a href="https://getkiln.ai">
        <picture>
            <img width="205" alt="Kiln AI Logo" src="https://github.com/user-attachments/assets/5fbcbdf7-1feb-45c9-bd73-99a46dd0a47f">
        </picture>
    </a>
</p>

# Kiln: Revolutionize AI Prototyping and Collaboration

**Kiln is the ultimate platform for rapid AI prototyping, fine-tuning, and seamless dataset collaboration, empowering you to build high-quality AI products faster.** [Explore the Kiln Repository](https://github.com/Kiln-AI/Kiln)

## Key Features:

*   🚀 **Intuitive Desktop Apps:** One-click installation for Windows, macOS, and Linux, offering a truly user-friendly experience.
*   🎛️ **Zero-Code Fine-Tuning:** Easily fine-tune models like Llama and GPT-4o without any coding, with automatic serverless model deployment.
*   📊 **Advanced Evals:** Evaluate your models using cutting-edge evaluators, allowing you to measure and improve performance.
*   🤖 **Synthetic Data Generation:** Generate custom training data using interactive visual tooling.
*   🧠 **Reasoning Model Training:** Train or distill custom reasoning models tailored to your specific needs.
*   🤝 **Team Collaboration with Git Integration:** Utilize Git-based version control for your AI datasets, facilitating effortless collaboration with QA, PM, and subject matter experts.
*   📝 **Automated Prompt Generation:** Generate prompts automatically from your data, including chain-of-thought, few-shot, and multi-shot prompts.
*   🌐 **Wide Model and Provider Support:** Compatible with a wide range of models via Ollama, OpenAI, OpenRouter, Fireworks, Groq, AWS, and any OpenAI compatible API.
*   🧑‍💻 **Open-Source Library and API:** Access our MIT-licensed Python library and OpenAPI REST API for seamless integration.
*   🔒 **Privacy-Focused:** Maintain data privacy with BYO API keys or local execution using Ollama.
*   🗃️ **Structured Data:** Build AI tasks that output JSON structured data.
*   💰 **Completely Free:** Access free desktop apps and an open-source library to kickstart your AI projects.

## Download Kiln Desktop Apps

Get started now! The Kiln desktop app is completely free and available for MacOS, Windows, and Linux.

[<img width="220" alt="Download button" src="https://github.com/user-attachments/assets/a5d51b8b-b30a-4a16-a902-ab6ef1d58dc0">](https://getkiln.ai/download)

## Demo

Explore Kiln's capabilities through our interactive demo:

[<img width="220" alt="Download button" src="https://github.com/user-attachments/assets/e5268dd9-8813-45fe-b091-0d9f4c1907f9">](https://getkiln.ai#demo)

<kbd>
<a href="https://getkiln.ai#demo">
<img alt="Kiln Preview" src="guides/kiln_preview.gif">
</a>
</kbd>

## Docs & Guides

Get the most out of Kiln with comprehensive documentation and guides.

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

For developers, explore our [Kiln Python Library Docs](https://kiln-ai.github.io/Kiln/kiln_core_docs/kiln_ai.html) to seamlessly integrate Kiln datasets into your workflows.

## Install Python Library

[![PyPI - Version](https://img.shields.io/pypi/v/kiln-ai.svg?logo=pypi&label=PyPI&logoColor=gold)](https://pypi.org/project/kiln-ai/) [![Docs](https://img.shields.io/badge/docs-pdoc-blue)](https://kiln-ai.github.io/Kiln/kiln_core_docs/index.html)

Integrate Kiln datasets into your workflows using our open-source [python library](https://pypi.org/project/kiln-ai/). Build fine tunes, use Kiln in Notebooks, build custom tools, and much more! [Read the docs](https://kiln-ai.github.io/Kiln/kiln_core_docs/index.html) for examples.

```bash
pip install kiln-ai
```

## Learn More

### Rapid Prototyping

Kiln simplifies the process of testing and comparing diverse approaches, allowing you to experiment with various models and techniques efficiently, leading to superior results without requiring code.

Key Features:
*   Various prompting techniques: basic, few-shot, multi-shot, repair & feedback
*   Chain of thought / thinking, with optional custom “thinking” instructions
*   Many models: GPT, Llama, Claude, Gemini, Mistral, Gemma, Phi
*   Fine Tuning: create custom models using your Kiln dataset

Planned additions: evals, and RAG.

### Collaborate Across Technical and Non-Technical Teams

Kiln serves as a collaborative hub, bridging the gap between subject matter experts and technical teams, enabling seamless teamwork in AI product development.

*   Subject matter experts can generate structured datasets and ratings.
*   Data-scientists can consume the dataset created.
*   QA and PM can easily identify issues sooner.

### Build High Quality AI Products with Datasets

Kiln enables you to construct high-quality models for your products by capturing inputs, outputs, human ratings, feedback, and repairs to create a dataset, leading to improved model performance over time.

*   Model quality improves automatically as the dataset grows.
*   Iterate the dataset to address issues.

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
*   Desktop App: free to download and use under our [EULA](app/EULA.md), and [source-available](/app). [License](app/LICENSE.txt)
*   The Kiln names and logos are trademarks of Chesterfield Laboratories Inc.

Copyright 2024 - Chesterfield Laboratories Inc.
```
Key improvements and summaries:

*   **SEO-Optimized Title:** Changed the title to be more keyword-rich.
*   **One-Sentence Hook:** Added a concise, impactful opening sentence to grab attention.
*   **Clear Headings:** Used clear and descriptive headings for each section.
*   **Bulleted Key Features:**  Emphasized key features with bullet points for easy scanning.
*   **Concise Summaries:** Summarized each section to get the key points across.
*   **Download & Demo Links:**  The inclusion of download and demo links make it easier to navigate.
*   **Revised Formatting:** Improved formatting for readability.
*   **Direct Links Back:**  Maintained the link back to the original repository.
*   **Added Value Propositions:** Emphasized the value provided (build higher quality products faster).
*   **Expanded on Value Proposition in "Learn More" Section**: Added more detail and emphasis on rapid prototyping, team collaboration and building high quality products with datasets.
*   **Simplified language:** streamlined wording for better comprehension.
*   **Consistency and structure:**  Improved structure and consistency.