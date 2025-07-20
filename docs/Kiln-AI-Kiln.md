<p align="center">
    <a href="https://getkiln.ai">
        <picture>
            <img width="205" alt="Kiln AI Logo" src="https://github.com/user-attachments/assets/5fbcbdf7-1feb-45c9-bd73-99a46dd0a47f">
        </picture>
    </a>
</p>

# Kiln: The All-in-One AI Prototyping and Dataset Collaboration Tool

**Kiln empowers you to rapidly prototype AI models and collaborate on datasets with ease.**  ([View the original repo](https://github.com/Kiln-AI/Kiln))

<p align="center">
  <a href="https://docs.getkiln.ai/docs/fine-tuning-guide"><strong>Fine Tuning</strong></a> •
  <a href="https://docs.getkiln.ai/docs/synthetic-data-generation"><strong>Synthetic Data Generation</strong></a> • 
  <a href="https://docs.getkiln.ai/docs/evaluations"><strong>Evals</strong></a> • 
  <a href="https://docs.getkiln.ai/docs/collaboration"><strong>Collaboration</strong></a> • 
  <a href="https://docs.getkiln.ai"><strong>Docs</strong></a>
</p>


[![Build and Test](https://github.com/Kiln-AI/kiln/actions/workflows/build_and_test.yml/badge.svg)](https://github.com/Kiln-AI/kiln/actions/workflows/build_and_test.yml)
[![Format and Lint](https://github.com/Kiln-AI/kiln/actions/workflows/format_and_lint.yml/badge.svg)](https://github.com/Kiln-AI/kiln/actions/workflows/format_and_lint.yml)
[![Desktop Apps Build](https://github.com/Kiln-AI/kiln/actions/workflows/build_desktop.yml/badge.svg)](https://github.com/Kiln-AI/kiln/actions/workflows/build_desktop.yml)
[![Web UI Build](https://github.com/Kiln-AI/kiln/actions/workflows/web_format_lint_build.yml/badge.svg)](https://github.com/Kiln-AI/kiln/actions/workflows/web_format_lint_build.yml)
[![Docs](https://github.com/Kiln-AI/Kiln/actions/workflows/build_docs.yml/badge.svg)](https://github.com/Kiln-AI/kiln/actions/workflows/build_docs.yml)

[![Test Count Badge](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/scosman/57742c1b1b60d597a6aba5d5148d728e/raw/test_count_kiln.json)](https://github.com/Kiln-AI/kiln/actions/workflows/test_count.yml) 
[![Test Coverage Badge](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/scosman/57742c1b1b60d597a6aba5d5148d728e/raw/library_coverage_kiln.json)](https://github.com/Kiln-AI/kiln/actions/workflows/test_count.yml)

[![PyPI - Version](https://img.shields.io/pypi/v/kiln-ai.svg?logo=pypi&label=PyPI&logoColor=gold)](https://pypi.org/project/kiln-ai/) 
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/kiln-ai.svg?logo=python&label=Python&logoColor=gold)](https://pypi.org/project/kiln-ai/)

[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv) 
[![linting - Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff) 
[![types - Pyright](https://img.shields.io/badge/types-pyright-blue.svg)](https://github.com/microsoft/pyright) 
[![Docs](https://img.shields.io/badge/docs-pdoc-blue)](https://kiln-ai.github.io/Kiln/kiln_core_docs/index.html)

[![MacOS](https://img.shields.io/badge/MacOS-black?logo=apple)](https://getkiln.ai/download) 
[![Windows](https://img.shields.io/badge/Windows-0067b8.svg?logo=data:image/svg%2bxml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0idXRmLTgiPz4KPHN2ZyBmaWxsPSIjZmZmIiB2aWV3Qm94PSIwIDAgMzIgMzIiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTE2Ljc0MiAxNi43NDJ2MTQuMjUzaDE0LjI1M3YtMTQuMjUzek0xLjAwNCAxNi43NDJ2MTQuMjUzaDE0LjI1NnYtMTQuMjUzek0xNi43NDIgMS4wMDR2MTQuMjU2aDE0LjI1M3YtMTQuMjU2ek0xLjAwNCAxLjAwNHYxNC4yNTZoMTQuMjU2di0xNC4yNTZ6Ij48L3BhdGg+Cjwvc3ZnPg==)](https://getkiln.ai/download) 
[![Linux](https://img.shields.io/badge/Linux-444444?logo=linux&logoColor=ffffff)](https://getkiln.ai/download) 
![Github Downsloads](https://img.shields.io/github/downloads/kiln-ai/kiln/total)

[![Discord](https://img.shields.io/badge/Discord-Kiln_AI-blue?logo=Discord&logoColor=white)](https://getkiln.ai/discord) 
[![Newsletter](https://img.shields.io/badge/Newsletter-subscribe-blue?logo=mailboxdotorg&logoColor=white)](https://getkiln.ai/blog)

[<img width="220" alt="Download button" src="https://github.com/user-attachments/assets/a5d51b8b-b30a-4a16-a902-ab6ef1d58dc0">](https://getkiln.ai/download) [<img width="220" alt="Quick start button" src="https://github.com/user-attachments/assets/aff1b35f-72c0-4286-9b28-40a415558359">](https://docs.getkiln.ai/getting-started/quickstart)

## Key Features:

*   **Intuitive Desktop Apps:** One-click apps for Windows, MacOS, and Linux offering a truly intuitive design for AI prototyping.
*   **Fine-Tuning:** Achieve zero-code fine-tuning for popular models like Llama, GPT-4o, and more, with automatic serverless deployment.
*   **Evals:** Evaluate model quality using state-of-the-art evaluators.
*   **Synthetic Data Generation:** Generate high-quality training data using our interactive visual tools.
*   **Reasoning Models:** Train or distill your own custom reasoning models.
*   **Team Collaboration:** Leverage Git-based version control for your AI datasets within an intuitive UI, streamlining collaboration with QA, PMs, and subject matter experts on structured data.
*   **Prompt Generation:** Automatically generate prompts from your data, including chain-of-thought, few-shot, multi-shot and more.
*   **Wide Model & Provider Support:** Utilize a wide array of models via Ollama, OpenAI, OpenRouter, Fireworks, Groq, AWS, or any OpenAI compatible API.
*   **Open-Source Library and API:** Integrate Kiln seamlessly with our MIT open-source Python library and OpenAPI REST API.
*   **Privacy-First:** Maintain complete data privacy with support for Bring Your Own API Keys or local running with Ollama.
*   **Structured Data:** Build AI tasks that communicate via JSON.
*   **Free:** Enjoy the benefits of free desktop apps and an open-source library.

## Download Kiln Desktop Apps

Get started with Kiln's free desktop apps for MacOS, Windows, and Linux.

[<img width="220" alt="Download button" src="https://github.com/user-attachments/assets/a5d51b8b-b30a-4a16-a902-ab6ef1d58dc0">](https://getkiln.ai/download)

## Demo

See Kiln in action with our interactive demo!

[<img width="220" alt="Download button" src="https://github.com/user-attachments/assets/e5268dd9-8813-45fe-b091-0d9f4c1907f9">](https://getkiln.ai#demo)

<kbd>
<a href="https://getkiln.ai#demo">
<img alt="Kiln Preview" src="guides/kiln_preview.gif">
</a>
</kbd>

## Docs & Guides

For guidance on getting started with Kiln, our [comprehensive documentation](https://docs.getkiln.ai) is available.

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

For developers, explore the [Kiln Python Library Docs](https://kiln-ai.github.io/Kiln/kiln_core_docs/kiln_ai.html) to learn how to load datasets into Kiln or integrate Kiln datasets within your code.

## Install Python Library

[![PyPI - Version](https://img.shields.io/pypi/v/kiln-ai.svg?logo=pypi&label=PyPI&logoColor=gold)](https://pypi.org/project/kiln-ai/) [![Docs](https://img.shields.io/badge/docs-pdoc-blue)](https://kiln-ai.github.io/Kiln/kiln_core_docs/index.html)

Integrate Kiln datasets into your workflows using our open-source [Python library](https://pypi.org/project/kiln-ai/). [Read the docs](https://kiln-ai.github.io/Kiln/kiln_core_docs/index.html) for examples.

```bash
pip install kiln-ai
```

## Learn More

### Rapid Prototyping

Kiln simplifies experimentation, making it easy to quickly test and compare different AI approaches without extensive coding.

Current features include:

*   Prompting techniques: basic, few-shot, multi-shot, repair & feedback
*   Chain of thought / thinking, with optional custom "thinking" instructions
*   Model Support: GPT, Llama, Claude, Gemini, Mistral, Gemma, Phi
*   Fine Tuning: create custom models using your Kiln dataset

Future features will include more no-code options like evals and RAG. Experienced data-scientists can leverage Kiln datasets with the python library today to create these techniques.

### Collaborate Across Technical and Non-Technical Teams

Kiln fosters collaboration by bridging the gap between subject matter experts and technical teams.

Subject matter experts can use Kiln to generate structured datasets and ratings without needing technical skills.

Data-scientists can then use the UI or the python library to use the datasets.

QA and PM can identify and help fix issues.

The Git-based file format allows for powerful collaboration and attribution.

### Build High-Quality AI Products with Datasets

Kiln helps you build high-quality models by capturing inputs, outputs, ratings, feedback, and repairs.

Your model quality improves as the dataset grows.

Easily iterate the dataset to address product shifts or bugs.

## Contributing & Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and contribution guidelines.

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