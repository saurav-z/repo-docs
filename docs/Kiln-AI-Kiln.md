<p align="center">
    <a href="https://getkiln.ai">
        <picture>
            <img width="205" alt="Kiln AI Logo" src="https://github.com/user-attachments/assets/5fbcbdf7-1feb-45c9-bd73-99a46dd0a47f">
        </picture>
    </a>
</p>

## Kiln: The Ultimate AI Prototyping and Dataset Collaboration Tool

**Kiln empowers you to rapidly prototype AI applications and seamlessly collaborate on datasets.** Explore our GitHub repository for the latest updates: [Kiln on GitHub](https://github.com/Kiln-AI/Kiln)

**Key Features:**

*   🚀 **Intuitive Desktop Apps:**  Easy-to-use apps for Windows, MacOS, and Linux, streamlining your AI workflow.
*   🎛️ **Zero-Code Fine-Tuning:** Fine-tune models like Llama and GPT-4o without writing code, plus automatic serverless deployment.
*   📊 **Comprehensive Evals:** Evaluate model performance using state-of-the-art evaluators.
*   🤖 **Synthetic Data Generation:** Generate training data visually with our interactive tooling.
*   🧠 **Custom Reasoning Models:** Train or distill reasoning models tailored to your needs.
*   🤝 **Collaborative Dataset Management:**  Utilize Git-based version control for your AI datasets with an intuitive UI for QA, PM, and subject matter experts.
*   📝 **Automated Prompt Generation:** Generate prompts from your data, including chain-of-thought, few-shot, and multi-shot styles.
*   🌐 **Wide Model and Provider Support:** Seamlessly integrate with models from Ollama, OpenAI, OpenRouter, Fireworks, Groq, AWS, and any OpenAI-compatible API.
*   🧑‍💻 **Open-Source Library & API:** Leverage our open-source Python library and OpenAPI REST API.
*   🔒 **Privacy-Focused:**  Maintain data privacy by using your own API keys or running locally with Ollama.
*   🗃️ **Structured Data:**  Build AI tasks that speak JSON.
*   💰 **Free & Open Source:** Free apps and open-source library.

## Download Kiln Desktop Apps

Get started with Kiln for free on MacOS, Windows, and Linux:

[<img width="220" alt="Download button" src="https://github.com/user-attachments/assets/a5d51b8b-b30a-4a16-a902-ab6ef1d58dc0">](https://getkiln.ai/download)

## Demo

See Kiln in action:

[<img width="220" alt="Demo button" src="https://github.com/user-attachments/assets/e5268dd9-8813-45fe-b091-0d9f4c1907f9">](https://getkiln.ai#demo)

<kbd>
<a href="https://getkiln.ai#demo">
<img alt="Kiln Preview" src="guides/kiln_preview.gif">
</a>
</kbd>

## Docs & Guides

Our comprehensive documentation is designed to get you up and running quickly.

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

For developers, refer to our [Kiln Python Library Docs](https://kiln-ai.github.io/Kiln/kiln_core_docs/kiln_ai.html) to learn how to integrate Kiln datasets into your workflows.

## Install Python Library

[![PyPI - Version](https://img.shields.io/pypi/v/kiln-ai.svg?logo=pypi&label=PyPI&logoColor=gold)](https://pypi.org/project/kiln-ai/) [![Docs](https://img.shields.io/badge/docs-pdoc-blue)](https://kiln-ai.github.io/Kiln/kiln_core_docs/index.html)

Integrate Kiln datasets into your workflows with our open-source [Python library](https://pypi.org/project/kiln-ai/).

```bash
pip install kiln-ai
```

## Learn More

### Rapid Prototyping

Kiln simplifies the process of testing various AI approaches by offering quick model comparison with a user-friendly interface, resulting in improved quality and performance.

We currently support:

* Various prompting techniques: basic, few-shot, multi-shot, repair & feedback
* Chain of thought / thinking, with optional custom “thinking” instructions
* Many models: GPT, Llama, Claude, Gemini, Mistral, Gemma, Phi
* Fine Tuning: create custom models using your Kiln dataset

In the future, we plan to add more powerful no-code options like evals, and RAG. For experienced data-scientists, you can create these techniques today using Kiln datasets and our python library.

### Collaborate Across Technical and Non-Technical Teams

Kiln bridges the gap between subject matter experts and technical teams, enabling seamless collaboration in AI product development.

Subject matter experts can use our intuitive desktop apps to generate structured datasets and ratings, without coding or using technical tools. No command line or GPU required.

Data-scientists can consume the dataset created by subject matter experts, using the UI, or deep dive with our python library.

QA and PM can easily identify issues sooner and help generate the dataset content needed to fix the issue at the model layer.

The dataset file format is designed to be be used with Git for powerful collaboration and attribution. Many people can contribute in parallel; collisions are avoided using UUIDs, and attribution is captured inside the dataset files. You can even share a dataset on a shared drive, letting completely non-technical team members contribute data and evals without knowing Git.

### Build High Quality AI Products with Datasets

Kiln helps you create a structured dataset from your product interactions, giving models more examples of quality content and mistakes.

Your model quality improves automatically as the dataset grows.

If your product goals shift or new bugs are found (as is almost always the case), you can easily iterate the dataset to address issues.

## Contributing & Development

Learn how to set up a development environment and contribute to Kiln by reviewing [CONTRIBUTING.md](CONTRIBUTING.md).

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

**Key improvements and SEO considerations:**

*   **Clear, Concise Hook:** The one-sentence summary at the beginning immediately captures the essence of Kiln and provides context.
*   **Keyword Optimization:** Incorporated relevant keywords like "AI Prototyping," "Dataset Collaboration," "Fine-tuning," "Synthetic Data Generation," "Open Source," "LLMs".
*   **Structured Headings:** Used clear, descriptive headings for better readability and SEO.
*   **Bulleted Key Features:**  Makes the most important aspects of the project easily scannable.
*   **Emphasis on Benefits:** Highlights the value proposition of Kiln, such as ease of use, collaboration, and building high-quality AI products.
*   **Call to Actions:**  Encourages users to download, view demos, and explore the documentation.
*   **Internal Links:**  Linking to the docs, video guides, and specific sections enhances user experience and helps with SEO.
*   **External Links:**  Provides direct links to the main repository, the demo, and relevant resources.
*   **Concise and Summarized Content:** Reduced redundant information, streamlining the document for quick understanding.
*   **Alt Text for Images:** Added descriptive alt text to images, improving accessibility and SEO.
*   **Code Blocks Formatting:** Properly formatted code blocks for readability.
*   **Clear Licensing Information:**  Provides clear information about licenses.
*   **Relevant Trademarks:** Includes trademark information as appropriate.