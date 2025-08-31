# Inspect: Evaluate and Secure Large Language Models

**Safeguard your AI: Inspect, a framework developed by the UK AI Safety Institute, empowers you to rigorously evaluate and secure your large language models (LLMs).**

[<img width="295" src="https://inspect.aisi.org.uk/images/aisi-logo.svg" alt="UK AI Safety Institute Logo" />](https://aisi.gov.uk/)

This repository contains the source code for **Inspect**, a powerful and flexible framework for comprehensive large language model (LLM) evaluation.  Inspect provides the tools you need to understand and improve the performance, safety, and reliability of your AI systems.

## Key Features of Inspect

*   **Built-in Components:** Leverage pre-built modules for essential LLM evaluation tasks.
*   **Prompt Engineering:** Design and refine prompts to effectively elicit responses from your LLMs.
*   **Tool Usage:**  Evaluate how well LLMs utilize external tools and resources.
*   **Multi-turn Dialog:**  Assess LLM performance in complex, conversational scenarios.
*   **Model-Graded Evaluations:**  Automatically assess LLM outputs against various criteria.
*   **Extensible Architecture:** Easily integrate custom evaluation techniques and extend Inspect's capabilities.

## Getting Started

Explore the official documentation for detailed instructions and tutorials: [https://inspect.aisi.org.uk/](https://inspect.aisi.org.uk/)

## Development and Contribution

To contribute to the development of Inspect, follow these steps:

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/UKGovernmentBEIS/inspect_ai.git
    cd inspect_ai
    ```
2.  **Install with Development Dependencies:**
    ```bash
    pip install -e ".[dev]"
    ```
3.  **(Optional) Install Pre-commit Hooks:**
    ```bash
    make hooks
    ```
4.  **Run Linting, Formatting, and Tests:**
    ```bash
    make check
    make test
    ```

### Recommended Development Environment

For an optimal development experience, we recommend using Visual Studio Code (VS Code) with the following extensions:

*   Python
*   Ruff
*   MyPy

VS Code will prompt you to install these extensions when you open the project.

**[View the original repository on GitHub](https://github.com/UKGovernmentBEIS/inspect_ai)**