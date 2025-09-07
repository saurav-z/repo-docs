# Inspect: Evaluate & Enhance Your Large Language Models (LLMs)

[![UK AI Security Institute Logo](https://inspect.aisi.org.uk/images/aisi-logo.svg)](https://aisi.gov.uk/)

**Inspect is the powerful open-source framework developed by the UK AI Security Institute to streamline the evaluation and improvement of Large Language Models (LLMs).**

## Key Features of Inspect:

*   **Comprehensive Evaluation Tools:** Built-in components for thorough LLM evaluation, including prompt engineering, tool usage analysis, and multi-turn dialogue assessment.
*   **Model-Graded Evaluations:**  Provides capabilities for evaluating models based on various scoring methods.
*   **Extensible Architecture:** Easily expand Inspect's functionality with custom extensions and integrations to support new evaluation techniques.
*   **User-Friendly Setup:** Simple installation process with clear instructions for development and testing.
*   **Development Workflow:**  Includes pre-commit hooks, linting, formatting, and testing tools to ensure code quality.

## Getting Started with Inspect

Explore the official documentation for detailed guidance on how to use Inspect:  <https://inspect.aisi.org.uk/>

##  Developing with Inspect

To contribute to the development of Inspect, follow these steps:

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/UKGovernmentBEIS/inspect_ai.git
    cd inspect_ai
    ```
2.  **Install Dependencies:**

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

## Recommended IDE Setup (VS Code)

We recommend using VS Code for development. Ensure you have the following extensions installed:

*   Python
*   Ruff
*   MyPy

You will be prompted to install these when you open the project in VS Code.

**[View the original repository on GitHub](https://github.com/UKGovernmentBEIS/inspect_ai)**