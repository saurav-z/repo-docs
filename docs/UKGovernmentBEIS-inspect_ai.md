# Inspect: Comprehensive Framework for LLM Evaluation

**Unlock the power of robust and reliable large language model (LLM) evaluations with Inspect, a powerful framework developed by the UK AI Safety Institute.**

[<img width="295" src="https://inspect.aisi.org.uk/images/aisi-logo.svg" alt="AI Safety Institute Logo" />](https://aisi.gov.uk/)

Inspect is designed to help you thoroughly assess and benchmark your LLMs across a variety of use cases.  This open-source tool provides a flexible and extensible platform for evaluating LLMs, promoting responsible and safe AI development.

**Key Features of Inspect:**

*   **Built-in Components:** Leverage pre-built modules for prompt engineering, tool usage analysis, multi-turn dialogue evaluation, and model-graded assessments.
*   **Extensible Architecture:** Easily integrate new evaluation techniques and scoring methods by developing custom Python packages.
*   **Comprehensive Documentation:**  Find detailed guides and examples at [https://inspect.aisi.org.uk/](https://inspect.aisi.org.uk/) to help you get started.
*   **Developed by AI Safety Experts:** Benefit from the expertise of the UK AI Safety Institute, ensuring a focus on responsible AI practices.

## Getting Started with Inspect

### Installation

To set up your development environment for Inspect, follow these steps:

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/UKGovernmentBEIS/inspect_ai.git
    cd inspect_ai
    ```

2.  **Install with Development Dependencies:**

    ```bash
    pip install -e ".[dev]"
    ```

### Development and Testing

Enhance your development workflow with these helpful commands:

*   **Install Pre-Commit Hooks:**

    ```bash
    make hooks
    ```

*   **Run Linting, Formatting, and Tests:**

    ```bash
    make check
    make test
    ```

### Recommended IDE Setup (VS Code)

For an optimal development experience, we recommend using Visual Studio Code.  Ensure you have the following extensions installed:

*   Python
*   Ruff
*   MyPy

VS Code will prompt you to install these extensions when you open the project.

[**View the Inspect Repository on GitHub**](https://github.com/UKGovernmentBEIS/inspect_ai)