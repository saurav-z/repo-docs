# Inspect: Evaluate Large Language Models with Confidence (by UK AI Security Institute)

[![Inspect Logo](https://inspect.aisi.org.uk/images/aisi-logo.svg)](https://aisi.gov.uk/)

**Tired of guessing how well your large language models (LLMs) perform?** Inspect, a powerful framework from the UK AI Security Institute, empowers you to rigorously evaluate LLMs and ensure their reliability and safety.

## Key Features of Inspect:

*   **Comprehensive Evaluation:** Built-in components for prompt engineering, tool usage assessment, multi-turn dialogue evaluation, and model-graded evaluations.
*   **Extensible Architecture:** Designed for flexibility, allowing you to easily integrate new elicitation and scoring techniques through Python packages.
*   **Open Source & Community Driven:** Developed by the UK AI Security Institute and available for contributions to improve LLM safety.
*   **Focus on Security:** Inspect is designed with security in mind, helping you assess and mitigate potential risks associated with LLMs.

## Getting Started with Inspect

For detailed documentation, including installation instructions, tutorials, and API references, please visit: <https://inspect.aisi.org.uk/>

## Contributing & Development

This section outlines how to contribute and develop on Inspect.

### Installation

To contribute to or develop Inspect, follow these steps:

```bash
git clone https://github.com/UKGovernmentBEIS/inspect_ai.git
cd inspect_ai
pip install -e ".[dev]"
```

### Optional Setup

*   **Pre-commit Hooks:** Install pre-commit hooks for automated code style checks:

    ```bash
    make hooks
    ```

*   **Linting, Formatting, and Testing:** Run these commands to ensure code quality:

    ```bash
    make check
    make test
    ```

### VS Code Recommended Extensions

If you are using VS Code, ensure that the following extensions are installed:
*   Python
*   Ruff
*   MyPy

## Learn More

Visit the official repository for the latest updates and information: [https://github.com/UKGovernmentBEIS/inspect_ai](https://github.com/UKGovernmentBEIS/inspect_ai)