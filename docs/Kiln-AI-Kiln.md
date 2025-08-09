<p align="center">
    <a href="https://getkiln.ai">
        <picture>
            <img width="205" alt="Kiln AI Logo" src="https://github.com/user-attachments/assets/5fbcbdf7-1feb-45c9-bd73-99a46dd0a47f">
        </picture>
    </a>
</p>

# Kiln: Supercharge Your AI Development with Rapid Prototyping and Collaborative Dataset Creation

**Kiln** is your all-in-one solution for rapidly prototyping AI models and fostering seamless dataset collaboration, empowering you to build high-quality AI products. ([See the original repo](https://github.com/Kiln-AI/Kiln))

## Key Features

*   **🚀 Intuitive Desktop Apps:** Experience one-click apps for Windows, MacOS, and Linux, designed for ease of use.
*   🎛️ **Zero-Code Fine-Tuning:** Fine-tune your Llama, GPT-4o, and other models without writing any code, with automatic serverless deployment.
*   📊 **Comprehensive Evals:** Evaluate your models and tasks using state-of-the-art evaluators.
*   🤖 **Interactive Synthetic Data Generation:** Generate training data with our interactive visual tooling.
*   🧠 **Reasoning Model Training & Distillation:** Train or distill your own custom reasoning models.
*   🤝 **Effortless Team Collaboration:** Utilize Git-based version control for your AI datasets and simplify collaboration with QA, PMs, and subject matter experts.
*   📝 **Intelligent Prompt Generation:** Automatically generate prompts including chain-of-thought, few-shot, and multi-shot, directly from your data.
*   🌐 **Extensive Model & Provider Compatibility:** Works with a wide range of models via Ollama, OpenAI, OpenRouter, Fireworks, Groq, AWS, and any OpenAI-compatible API.
*   🧑‍💻 **Open-Source Flexibility:** Leverage our open-source Python library and OpenAPI REST API (MIT License).
*   🔒 **Privacy-First Design:** Protect your data with our privacy-focused approach; bring your own API keys or run locally with Ollama.
*   🗃️ **Structured Data Focus:** Build AI tasks optimized for JSON-based structured data.
*   💰 **Completely Free:** Benefit from free desktop apps and an open-source library.

## Download Kiln Desktop Apps

Get started with Kiln today. Our desktop apps are free and available on MacOS, Windows and Linux.

[<img width="220" alt="Download button" src="https://github.com/user-attachments/assets/a5d51b8b-b30a-4a16-a902-ab6ef1d58dc0">](https://getkiln.ai/download)

## Demo

Explore Kiln's capabilities through a demo.

[<img width="220" alt="Download button" src="https://github.com/user-attachments/assets/e5268dd9-8813-45fe-b091-0d9f4c1907f9">](https://getkiln.ai#demo)

<kbd>
<a href="https://getkiln.ai#demo">
<img alt="Kiln Preview" src="guides/kiln_preview.gif">
</a>
</kbd>

## Docs & Guides

Kiln is designed to be intuitive, but our comprehensive documentation is available to help you get the most out of Kiln.

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

For developers, see our [Kiln Python Library Docs](https://kiln-ai.github.io/Kiln/kiln_core_docs/kiln_ai.html) which include how to load datasets into Kiln, or using Kiln datasets in your own code-base/notebooks.

## Install Python Library

[![PyPI - Version](https://img.shields.io/pypi/v/kiln-ai.svg?logo=pypi&label=PyPI&logoColor=gold)](https://pypi.org/project/kiln-ai/) [![Docs](https://img.shields.io/badge/docs-pdoc-blue)](https://kiln-ai.github.io/Kiln/kiln_core_docs/index.html)

Integrate Kiln datasets seamlessly into your workflows, build fine-tunes, and utilize Kiln in notebooks with our open-source [python library](https://pypi.org/project/kiln-ai/). [Explore the documentation](https://kiln-ai.github.io/Kiln/kiln_core_docs/index.html) for detailed examples.

```bash
pip install kiln-ai
```

## Learn More

### Streamlined Prototyping

With Kiln, effortlessly experiment with diverse approaches and evaluate them within clicks, without getting bogged down in code. This leads to better performance and quality in your model.

We presently offer support for:

-   A variety of prompting techniques: basic, few-shot, multi-shot, repair & feedback.
-   Chain of thought / thinking with custom “thinking” instructions.
-   Many models: GPT, Llama, Claude, Gemini, Mistral, Gemma, Phi
-   Fine Tuning: creating custom models using your Kiln dataset

We intend to implement no-code options like evals and RAG in the future. If you are experienced in data-science, feel free to apply these techniques using Kiln datasets and our python library.

### Facilitate Collaboration Across Technical and Non-Technical Teams

Kiln helps bridge the gap between subject matter experts and technical teams when building AI products. It provides a platform for non-technical users to generate structured datasets without complex tools or coding knowledge.

With our user-friendly desktop apps, subject matter experts can generate structured datasets and ratings without coding or technical skills.

Data-scientists can work with datasets created by experts using the UI, or deep dive into data with our Python library.

QA and PM teams can identify issues earlier and create the necessary dataset content to fix the model layer.

The dataset file format integrates with Git for enhanced collaboration and attribution. The file format allows parallel contributions, avoiding collisions with UUIDs and capturing attribution. You can even share the dataset on a shared drive, allowing any non-technical team member to contribute data and evals without knowing Git.

### Build High Quality AI Products with Datasets

Create and refine datasets for your product using Kiln. Each time you use Kiln, we capture the inputs, outputs, human ratings, feedback, and repairs required to build top-quality models for your product. The more you use Kiln, the more data you have.

As your dataset expands, the model quality naturally improves by giving your models quality examples (and mistakes).

You can easily iterate the dataset if product goals or new issues arise, as often happens.

## Contributing & Development

For setup and contribution details, see [CONTRIBUTING.md](CONTRIBUTING.md).

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

*   **SEO Optimization:** Added keywords like "AI prototyping," "dataset collaboration," and "fine-tuning" to headings and content.
*   **One-Sentence Hook:** The first sentence provides a clear and concise value proposition.
*   **Clear Structure:** Uses headings, subheadings, and bullet points for readability and easy scanning.
*   **Comprehensive Feature List:**  Expanded and improved the key features section.
*   **Actionable Download and Demo Buttons:**  Reinforced calls to action with clear download and demo links.
*   **Improved Formatting:** Consistent and readable formatting.
*   **Contextual Links:**  Links back to the original repository.
*   **Emphasis on Benefits:** Focused on the *benefits* of using Kiln (e.g., "Supercharge your AI Development").
*   **Added "Learn More" sections:** This is a good place to provide more context and keywords.
*   **Removed redundant links:** Combined similar links.
*   **Added a brief introduction to each section:** This helps with readability.
*   **"Build High Quality AI Products with Datasets" sections:** More directly addresses the user's needs.
*   **Citation:** Keep the citation information to credit the project.