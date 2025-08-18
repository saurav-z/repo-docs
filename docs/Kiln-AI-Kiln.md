<p align="center">
    <a href="https://kiln.tech">
        <picture>
            <img width="205" alt="Kiln AI Logo" src="https://github.com/user-attachments/assets/5fbcbdf7-1feb-45c9-bd73-99a46dd0a47f">
        </picture>
    </a>
</p>

# Kiln: Accelerate AI Development with Collaborative Data & Fine-Tuning

**Kiln is your all-in-one platform for rapid AI prototyping, fine-tuning, and collaborative dataset management, designed to help you build and deploy high-quality AI solutions.**

[View the original repository on GitHub](https://github.com/Kiln-AI/Kiln)

---

## Key Features

*   🚀 **Intuitive Desktop Apps**: Easy-to-use desktop applications for Windows, MacOS, and Linux.
*   🎛️ **Zero-Code Fine-Tuning**: Fine-tune LLMs (Llama, GPT-4o, and more) without writing code and deploy models automatically.
*   📊 **Comprehensive Evals**: Evaluate model performance with state-of-the-art evaluators.
*   🤖 **Synthetic Data Generation**: Create training datasets using interactive visual tools.
*   🧠 **Custom Reasoning Models**: Train and refine custom reasoning models for enhanced performance.
*   🤝 **Collaborative Dataset Management**: Git-based version control for datasets, streamlining collaboration among QA, PM, and subject matter experts.
*   📝 **Automated Prompt Generation**: Simplify prompt creation with features like chain-of-thought and few-shot prompting.
*   🌐 **Broad Model & Provider Support**: Integrate with any model via Ollama, OpenAI, OpenRouter, Fireworks, Groq, AWS, and more.
*   🧑‍💻 **Open-Source Library and API**: Leverage the power of an open-source Python library and REST API.
*   🔒 **Privacy-First Approach**: Ensure data privacy by using your own API keys or running locally with Ollama.
*   🗃️ **Structured Data**: Build AI tasks optimized for JSON-based data.
*   💰 **Completely Free**: Enjoy free access to Kiln apps and an open-source library.

---

## Download Kiln Desktop Apps

[<img width="220" alt="Download button" src="https://github.com/user-attachments/assets/a5d51b8b-b30a-4a16-a902-ab6ef1d58dc0">](https://kiln.tech/download)

Available on MacOS, Windows and Linux.

---

## Demo

[<img width="220" alt="Download button" src="https://github.com/user-attachments/assets/e5268dd9-8813-45fe-b091-0d9f4c1907f9">](https://kiln.tech#demo)

<kbd>
<a href="https://kiln.tech#demo">
<img alt="Kiln Preview" src="guides/kiln_preview.gif">
</a>
</kbd>

---

## Docs & Guides

Get started easily with our intuitive desktop apps.  Our [docs are here to help](https://docs.getkiln.ai) if you need them.

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

---

## Install Python Library

[![PyPI - Version](https://img.shields.io/pypi/v/kiln-ai.svg?logo=pypi&label=PyPI&logoColor=gold)](https://pypi.org/project/kiln-ai/) [![Docs](https://img.shields.io/badge/docs-pdoc-blue)](https://kiln-ai.github.io/Kiln/kiln_core_docs/index.html)

Our open-source [python library](https://pypi.org/project/kiln-ai/) allows you to integrate Kiln datasets into your own workflows, build fine tunes, use Kiln in Notebooks, build custom tools, and much more! [Read the docs](https://kiln-ai.github.io/Kiln/kiln_core_docs/index.html) for examples.

```bash
pip install kiln-ai
```

---

## Learn More

### Rapid Prototyping

Kiln simplifies the process of experimenting with new models and techniques. Try various approaches quickly, compare results with ease, and boost quality and performance without extensive coding.

Current support includes:

-   Diverse prompting techniques: basic, few-shot, multi-shot, repair & feedback
-   Chain of thought / thinking, with optional custom “thinking” instructions
-   Wide model compatibility: GPT, Llama, Claude, Gemini, Mistral, Gemma, Phi
-   Fine Tuning: Customize models using your Kiln dataset

Future plans include even more powerful no-code options like evals and RAG. Data scientists can leverage Kiln datasets and our Python library to utilize these techniques now.

### Collaborate Across Technical and Non-Technical Teams

Kiln fosters collaboration between subject matter experts and technical teams. It's designed to bridge the gap between domain knowledge and model building.

Subject matter experts can create structured datasets and ratings effortlessly using our intuitive desktop apps. No coding or technical tools required.

Data scientists can utilize the datasets created by experts through the UI or our Python library.

QA and PM teams can identify and address issues efficiently, contributing to the necessary dataset content to improve model performance.

The dataset file format supports Git for enhanced collaboration and attribution. Concurrent contributions are managed with UUIDs, and all contributions are captured within the dataset files. Datasets can be shared on a shared drive, enabling non-technical team members to contribute data and evaluations.

### Build High Quality AI Products with Datasets

Kiln allows you to generate datasets for use in your products. As you use Kiln, inputs, outputs, human ratings, feedback, and repairs are captured to help build models. More use leads to richer data.

Your model improves as the dataset grows, giving models more quality examples and examples of mistakes.

If your product goals shift or new bugs are found, you can iterate on the dataset to address the issues.

---

## Contributing & Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for details on setting up a development environment and contributing to Kiln.

---

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

---

## Licenses & Trademarks

*   Python Library: [MIT License](libs/core/LICENSE.txt)
*   Python REST Server/API: [MIT License](libs/server/LICENSE.txt)
*   Desktop App: free to download and use under our [EULA](app/EULA.md), and [source-available](/app). [License](app/LICENSE.txt)
*   The Kiln names and logos are trademarks of Chesterfield Laboratories Inc.

Copyright 2024 - Chesterfield Laboratories Inc.
```
Key improvements and explanations:

*   **SEO Optimization:**  The title and headings are optimized for relevant keywords ("AI prototyping," "dataset collaboration," "fine-tuning," etc.). The introduction directly tells what Kiln is for.
*   **Clear Structure:**  Uses headings, subheadings, and bullet points to make the information easy to scan and understand. This is crucial for both users and search engines.
*   **Concise Language:** Removed some of the more verbose original text and rewrote it for clarity.
*   **Strong Hook:** The opening sentence is a direct value proposition: "Kiln is your all-in-one platform for rapid AI prototyping, fine-tuning, and collaborative dataset management, designed to help you build and deploy high-quality AI solutions."
*   **Key Feature Highlighting:**  The key features are clearly bulleted, making it easy to grasp the main benefits. The descriptions are also improved to be more compelling.
*   **Call to Action:** Includes clear "Download" links and a "Learn More" section to guide the user.
*   **Complete Information:** Retains all the important information from the original README, including the download links, demo, guides, installation instructions, contribution guidelines, and licensing information.
*   **Developer-Friendly:** The Python library installation and development instructions are clearly presented.
*   **Markdown Formatting:** Ensures proper formatting for readability on GitHub.  Bolded important terms.
*   **Concise and Direct** Avoids unnecessary marketing jargon.
*   **Link Back to Repo:**  Includes a link to the GitHub repo at the beginning to increase the likelihood that people will end up on the page.