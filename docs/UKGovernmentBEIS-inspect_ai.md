# Inspect: Evaluate and Enhance Large Language Models (LLMs)

[![Inspect Logo](https://inspect.aisi.org.uk/images/aisi-logo.svg)](https://aisi.gov.uk/)

**Unlock the potential of your AI models with Inspect, a powerful framework developed by the UK AI Safety Institute for comprehensive large language model (LLM) evaluation and analysis.**

Inspect provides a robust and flexible toolkit for assessing and improving LLMs, enabling researchers and developers to gain deeper insights into their performance.

## Key Features:

*   **Comprehensive Evaluation:** Evaluate LLMs using a variety of built-in components.
*   **Prompt Engineering Capabilities:** Facilitate prompt engineering for optimized LLM performance.
*   **Tool Usage Support:**  Analyze and refine LLMs leveraging external tools.
*   **Multi-Turn Dialog Analysis:** Evaluate LLMs in complex, multi-turn conversational scenarios.
*   **Model-Graded Evaluations:** Utilize sophisticated methods for assessing LLM outputs.
*   **Extensible Architecture:** Easily extend Inspect's functionality through Python packages, allowing for custom elicitation and scoring techniques.

## Getting Started

For detailed information and usage examples, please refer to the official Inspect documentation: [https://inspect.aisi.org.uk/](https://inspect.aisi.org.uk/)

## Development and Contribution

To contribute to Inspect's development, follow these steps:

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

### VS Code Setup (Recommended)

Ensure you have the recommended VS Code extensions installed (Python, Ruff, and MyPy). You will be prompted to install these when you open the project in VS Code.

**[View the original Inspect repository on GitHub](https://github.com/UKGovernmentBEIS/inspect_ai)**