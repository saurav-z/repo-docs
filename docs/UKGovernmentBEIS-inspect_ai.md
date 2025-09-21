# Inspect: Evaluate and Enhance Your Large Language Models (LLMs)

[<img width="295" src="https://inspect.aisi.org.uk/images/aisi-logo.svg" alt="AI Security Institute Logo" />](https://aisi.gov.uk/)

**Inspect, a powerful framework from the UK AI Security Institute, empowers you to rigorously evaluate and improve your Large Language Models (LLMs).**

This open-source framework provides a comprehensive suite of tools for:

*   **Prompt Engineering:** Design and refine prompts to optimize LLM performance.
*   **Tool Usage:** Evaluate how well your LLMs utilize external tools.
*   **Multi-Turn Dialogue Analysis:** Analyze and improve LLMs in complex conversational scenarios.
*   **Model-Graded Evaluations:** Implement and customize advanced evaluation techniques.
*   **Extensibility:** Easily add support for new elicitation and scoring methods through Python packages.

## Getting Started

For detailed information on using Inspect, please refer to the official documentation: [https://inspect.aisi.org.uk/](https://inspect.aisi.org.uk/)

## Development Setup

Contribute to the development of Inspect by following these steps:

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/UKGovernmentBEIS/inspect_ai.git
    cd inspect_ai
    ```

2.  **Install with development dependencies:**

    ```bash
    pip install -e ".[dev]"
    ```

3.  **(Optional) Install pre-commit hooks:**

    ```bash
    make hooks
    ```

4.  **Run checks, formatting, and tests:**

    ```bash
    make check
    make test
    ```

### VS Code Setup

For an optimal development experience with VS Code, ensure you have the following recommended extensions installed:

*   Python
*   Ruff
*   MyPy

You will be prompted to install these extensions when opening the project in VS Code.

**Learn more about Inspect on the original GitHub repository:** [https://github.com/UKGovernmentBEIS/inspect_ai](https://github.com/UKGovernmentBEIS/inspect_ai)