<p align="center">
    <a href="https://kiln.tech">
        <picture>
            <img width="205" alt="Kiln AI Logo" src="https://github.com/user-attachments/assets/fc20f392-d913-4a94-a828-a66d21689496">
        </picture>
    </a>
</p>

<h2 align="center">
    Kiln: Build, Evaluate, and Fine-tune AI Models Faster with Ease
</h2>

<p align="center">
  [**Explore the Kiln AI Repository on GitHub**](https://github.com/Kiln-AI/Kiln)
</p>

Kiln is a powerful, free, and intuitive platform designed to simplify AI model development, offering tools for evals, synthetic data generation, and fine-tuning all in one place.

## Key Features

*   **Intuitive Desktop Apps:** One-click installation for Windows, MacOS, and Linux, streamlining your workflow.
*   **Advanced Evaluations:** Assess the quality of your AI models and tasks using state-of-the-art evaluators.
*   **Effortless Fine-tuning:** Zero-code fine-tuning capabilities for models like Llama, GPT-4o, and more, with automatic serverless deployment.
*   **Synthetic Data Generation:** Create evaluation datasets and fine-tuning data using our interactive, visual tools.
*   **Powerful Tools & MCP:** Connect a suite of powerful tools directly to your Kiln tasks.
*   **Custom Reasoning Models:** Train or distill your own tailored reasoning models.
*   **Intelligent Prompt Generation:** Automate prompt creation, including chain-of-thought, few-shot, and multi-shot techniques.
*   **Extensive Model Compatibility:** Seamlessly integrate with over 100 different AI models via providers like Ollama, OpenAI, OpenRouter, Fireworks, Groq, AWS, and more.
*   **Collaborative Workflows:** Leverage Git-based version control for collaborative AI dataset management, including data samples, evals, prompts, and ratings.
*   **Structured Data Support:** Build AI tasks that integrate with JSON data.
*   **Open-Source Library and API:** Utilize our MIT-licensed Python library and OpenAPI REST API for custom integrations.
*   **Privacy-Focused:** Kiln operates locally, ensuring your data privacy. Use your own API keys or Ollama.
*   **Comprehensive Documentation:** Benefit from easy-to-follow video guides and documentation for both beginners and advanced users.
*   **Free to Use:** Enjoy free desktop apps and an open-source library.

## Demo

See Kiln in action! Watch a 2-minute overview or a 20-minute end-to-end project demo.

<kbd>
<a href="https://kiln.tech#demo">
<img alt="Kiln Preview" src="guides/kiln_preview.gif">
</a>
</kbd>

## Who Uses Kiln?

<img width="600" alt="Logo Grid" src="https://github.com/user-attachments/assets/fa633000-d0db-4140-b3be-485e4c4a71c0" />

<sub>People from these companies have joined our communities on Github & Discord.</sub>

## Get Started

Download Kiln Desktop Apps, available for MacOS, Windows, and Linux.

[<img width="180" alt="Download button" src="https://github.com/user-attachments/assets/a5d51b8b-b30a-4a16-a902-ab6ef1d58dc0">](https://kiln.tech/download)

## Documentation & Resources

Explore our extensive documentation and guides to get the most out of Kiln:

### Video Guides

*   [Fine Tuning LLM Models](https://docs.kiln.tech/docs/fine-tuning-guide)
*   [Guide: Train a Reasoning Model](https://docs.kiln.tech/docs/guide-train-a-reasoning-model)
*   [LLM Evaluators](https://docs.kiln.tech/docs/evaluators)
*   [End to End Project Demo](https://docs.kiln.tech/docs/end-to-end-project-demo)

### All Docs

*   [Quick Start](https://docs.kiln.tech/getting-started/quickstart)
*   [How to use any AI model or provider in Kiln](https://docs.kiln.tech/docs/models-and-ai-providers)
*   [Reasoning & Chain of Thought](https://docs.kiln.tech/docs/reasoning-and-chain-of-thought)
*   [Synthetic Data Generation](https://docs.kiln.tech/docs/synthetic-data-generation)
*   [Collaborating with Kiln](https://docs.kiln.tech/docs/collaboration)
*   [Rating and Labeling Data](https://docs.kiln.tech/docs/reviewing-and-rating)
*   [Prompt Styles](https://docs.kiln.tech/docs/prompts)
*   [Structured Data / JSON](https://docs.kiln.tech/docs/structured-data-json)
*   [Organizing Kiln Datasets (Tags and Filters)](https://docs.kiln.tech/docs/organizing-datasets)
*   [Our Data Model](https://docs.kiln.tech/docs/kiln-datamodel)
*   [Repairing Responses](https://docs.kiln.tech/docs/repairing-responses)
*   [Keyboard Shortcuts](https://docs.kiln.tech/docs/keyboard-shortcuts)
*   [Privacy Overview: Private by Design](https://docs.kiln.tech/docs/privacy)

For developers, explore the [Kiln Python Library Docs](https://kiln-ai.github.io/Kiln/kiln_core_docs/kiln_ai.html) to learn how to load datasets and integrate Kiln within your codebases.

## Build & Tools

|         |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| CI      | [![Build and Test](https://github.com/Kiln-AI/kiln/actions/workflows/build_and_test.yml/badge.svg)](https://github.com/Kiln-AI/kiln/actions/workflows/build_and_test.yml) [![Format and Lint](https://github.com/Kiln-AI/kiln/actions/workflows/format_and_lint.yml/badge.svg)](https://github.com/Kiln-AI/kiln/actions/workflows/format_and_lint.yml) [![Desktop Apps Build](https://github.com/Kiln-AI/kiln/actions/workflows/build_desktop.yml/badge.svg)](https://github.com/Kiln-AI/kiln/actions/workflows/build_desktop.yml) [![Web UI Build](https://github.com/Kiln-AI/kiln/actions/workflows/web_format_lint_build.yml/badge.svg)](https://github.com/Kiln-AI/kiln/actions/workflows/web_format_lint_build.yml)                                                                                                           |
| Tests   | [![Test Count Badge](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/scosman/57742c1b1b60d597a6aba5d5148d728e/raw/test_count_kiln.json)](https://github.com/Kiln-AI/kiln/actions/workflows/test_count.yml) [![Test Coverage Badge](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/scosman/57742c1b1b60d597a6aba5d5148d728e/raw/library_coverage_kiln.json)](https://github.com/Kiln-AI/kiln/actions/workflows/test_count.yml)                                                                                                                                                                                                                                                                                                                                                       |
| Package | [![PyPI - Version](https://img.shields.io/pypi/v/kiln-ai.svg?logo=pypi&label=PyPI&logoColor=gold)](https://pypi.org/project/kiln-ai/) [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/kiln-ai.svg?logo=python&label=Python&logoColor=gold)](https://pypi.org/project/kiln-ai/)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| Meta    | [![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv) [![linting - Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff) [![types - Pyright](https://img.shields.io/badge/types-pyright-blue.svg)](https://github.com/microsoft/pyright) [![Docs](https://img.shields.io/badge/docs-pdoc-blue)](https://kiln-ai.github.io/Kiln/kiln_core_docs/index.html)                                                                                                                                                                                                                                                                     |
| Apps    | [![MacOS](https://img.shields.io/badge/MacOS-black?logo=apple)](https://kiln.tech/download) [![Windows](https://img.shields.io/badge/Windows-0067b8.svg?logo=data:image/svg%2bxml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0idXRmLTgiPz4KPHN2ZyBmaWxsPSIjZmZmIiB2aWV3Qm94PSIwIDAgMzIgMzIiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTE2Ljc0MiAxNi43NDJ2MTQuMjUzaDE0LjI1M3YtMTQuMjUzek0xLjAwNCAxNi43NDJ2MTQuMjUzaDE0LjI1NnYtMTQuMjUzek0xNi43NDIgMS4wMDR2MTQuMjU2aDE0LjI1M3YtMTQuMjU2ek0xLjAwNCAxLjAwNHYxNC4yNTZoMTQuMjU2di0xNC4yNTZ6Ij48L3BhdGg+Cjwvc3ZnPg==)](https://kiln.tech/download) [![Linux](https://img.shields.io/badge/Linux-444444?logo=linux&logoColor=ffffff)](https://kiln.tech/download) ![Github Downsloads](https://img.shields.io/github/downloads/kiln-ai/kiln/total) |
| Connect | [![Discord](https://img.shields.io/badge/Discord-Kiln_AI-blue?logo=Discord&logoColor=white)](https://kiln.tech/discord) [![Newsletter](https://img.shields.io/badge/Newsletter-subscribe-blue?logo=mailboxdotorg&logoColor=white)](https://kiln.tech/blog)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |

## Python Library

[![PyPI - Version](https://img.shields.io/pypi/v/kiln-ai.svg?logo=pypi&label=PyPI&logoColor=gold)](https://pypi.org/project/kiln-ai/) [![Docs](https://img.shields.io/badge/docs-pdoc-blue)](https://kiln-ai.github.io/Kiln/kiln_core_docs/index.html)

Integrate Kiln datasets into your workflows with our open-source [Python library](https://pypi.org/project/kiln-ai/).  Build fine-tunes, use Kiln in Notebooks, and create custom tools.  [Read the docs](https://kiln-ai.github.io/Kiln/kiln_core_docs/index.html) for more details.

```bash
pip install kiln-ai
```

## Learn More

### Accelerate AI Prototyping

Kiln provides tools to quickly experiment with different AI approaches, allowing for rapid prototyping and comparison in a code-free environment, ultimately resulting in enhanced model quality and performance.

We currently support:

-   Various prompting techniques: basic, few-shot, multi-shot, repair & feedback
-   Chain of thought / thinking, with optional custom “thinking” instructions
-   Many models: GPT, Llama, Claude, Gemini, Mistral, Gemma, Phi
-   Fine Tuning: create custom models using your Kiln dataset
-   Evaluations using LLM-as-Judge and G-Eval
-   Distilling models

In the future, we plan to add more powerful no-code options like RAG. For experienced data-scientists, you can create these techniques today using Kiln datasets and our Python library.

### Foster Collaboration

Kiln bridges the gap between subject matter experts and technical teams. Subject matter experts can create structured datasets and ratings via the intuitive desktop apps, without needing to know coding.

Data scientists can utilize the created datasets through the UI, or dive in with the python library.

QA and PM can easily identify issues sooner and help generate the dataset content needed to fix the issue at the model layer.

The dataset file format is designed to be used with Git for powerful collaboration and attribution. Many people can contribute in parallel; collisions are avoided using UUIDs, and attribution is captured inside the dataset files. You can even share a dataset on a shared drive, letting completely non-technical team members contribute data and evals without knowing Git.

### Create High-Quality AI Products

Create datasets for your products using Kiln. Capture inputs, outputs, human ratings, feedback, and repairs to build high-quality models for use in your product. The more you use it, the more data you have.

Our synthetic data generation tool can build datasets for evals and fine-tuning in minutes.

Your model quality improves automatically as the dataset grows, by giving the models more examples of quality content (and mistakes). If your product goals shift or new bugs are found (as is almost always the case), you can easily iterate the dataset to address issues.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for details on setting up a development environment and contributing.

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