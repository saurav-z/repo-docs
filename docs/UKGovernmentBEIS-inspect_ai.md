# Inspect: Evaluate Large Language Models with Precision 🤖

[<img width="295" src="https://inspect.aisi.org.uk/images/aisi-logo.svg" alt="UK AI Security Institute Logo" />](https://aisi.gov.uk/)

**Inspect, developed by the UK AI Security Institute (AISI), is your go-to framework for comprehensive large language model (LLM) evaluations.**

This framework empowers you to rigorously assess and understand the capabilities of your LLMs, ensuring their reliability and safety.

## Key Features of Inspect:

*   **Built-in Components:** Offers a comprehensive suite of tools, including:
    *   Prompt engineering capabilities
    *   Tool usage evaluation
    *   Multi-turn dialog assessment
    *   Model-graded evaluations
*   **Extensible Architecture:** Easily extend Inspect with custom components and features through Python packages, supporting new elicitation and scoring techniques.
*   **Open Source:** Leverage and contribute to this powerful framework to improve your LLM evaluation processes.

## Getting Started with Inspect

For detailed instructions and comprehensive documentation, please visit the official Inspect documentation at:  <https://inspect.aisi.org.uk/>

## Development and Contribution

Contribute to the development of Inspect by following these steps:

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/UKGovernmentBEIS/inspect_ai.git
    cd inspect_ai
    ```

2.  **Install with Development Dependencies:**

    ```bash
    pip install -e ".[dev]"
    ```

3.  **Optional: Install Pre-commit Hooks:**

    ```bash
    make hooks
    ```

4.  **Run Linting, Formatting, and Tests:**

    ```bash
    make check
    make test
    ```

5.  **VS Code Configuration:**  For optimal development, we recommend using VS Code with the following extensions: Python, Ruff, and MyPy.  You will be prompted to install these when you open the project in VS Code.

[**Original Repository on GitHub**](https://github.com/UKGovernmentBEIS/inspect_ai)