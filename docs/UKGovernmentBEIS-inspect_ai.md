# Inspect: Evaluate and Enhance Large Language Models

**Inspect, developed by the UK AI Safety Institute, empowers you to rigorously evaluate and refine your Large Language Models (LLMs).**

[![AI Safety Institute Logo](https://inspect.aisi.org.uk/images/aisi-logo.svg)](https://aisi.gov.uk/)

## Key Features of Inspect:

*   **Comprehensive Evaluation:** Provides a robust framework for evaluating LLMs.
*   **Prompt Engineering Support:** Offers tools to design and optimize your prompts.
*   **Tool Usage Analysis:**  Helps you understand how your models utilize tools.
*   **Multi-Turn Dialogue Capabilities:**  Facilitates evaluation of complex, multi-turn conversations.
*   **Model-Graded Evaluations:** Enables sophisticated evaluation techniques.
*   **Extensible Architecture:**  Supports integration of custom evaluation techniques and components.

## Getting Started with Inspect

For detailed information and tutorials, please refer to the official documentation:  [https://inspect.aisi.org.uk/](https://inspect.aisi.org.uk/)

## Development and Contribution

This section details how to contribute to the Inspect project.

### Installation for Development

To begin contributing, clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/UKGovernmentBEIS/inspect_ai.git
cd inspect_ai
pip install -e ".[dev]"
```

### Development Tools

Enhance your development workflow with these tools:

*   **Pre-commit Hooks:** Install pre-commit hooks for automated code quality checks:

    ```bash
    make hooks
    ```

*   **Code Quality Checks:** Run linting, formatting, and tests:

    ```bash
    make check
    make test
    ```

*   **VS Code Extensions:**  Recommended VS Code extensions for enhanced development: Python, Ruff, and MyPy. You will be prompted to install these when you open the project in VS Code.

## Learn More

Explore the full potential of Inspect and the ongoing work of the UK AI Safety Institute.  Visit the original repository on GitHub: [https://github.com/UKGovernmentBEIS/inspect_ai](https://github.com/UKGovernmentBEIS/inspect_ai)