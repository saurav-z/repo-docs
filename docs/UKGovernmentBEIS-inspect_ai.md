# Inspect: Evaluate Large Language Models with Ease

**Assess and refine your Large Language Models with Inspect, a powerful evaluation framework from the UK AI Safety Institute.** ([See the original repository](https://github.com/UKGovernmentBEIS/inspect_ai))

[<img width="295" src="https://inspect.aisi.org.uk/images/aisi-logo.svg" alt="Inspect AI Logo" />](https://aisi.gov.uk/)

## Key Features of the Inspect Framework:

*   **Comprehensive Evaluation:** Inspect offers a robust framework for evaluating Large Language Models (LLMs).
*   **Built-in Components:** Leverage pre-built features for prompt engineering, tool usage, and multi-turn dialog management.
*   **Model-Graded Evaluations:** Conduct effective evaluations with model-graded scoring capabilities.
*   **Extensible Design:** Easily extend Inspect's functionality with custom Python packages to support new elicitation and scoring techniques.
*   **Developed by Experts:** Built by the UK AI Safety Institute, ensuring a focus on safety and responsible AI practices.

## Getting Started with Inspect

For detailed information and usage examples, please consult the official documentation: <https://inspect.aisi.org.uk/>

## Development and Contribution

Contribute to the development of Inspect by following these steps:

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

4.  **Run Checks and Tests:**

    ```bash
    make check
    make test
    ```

## Recommended Development Environment

For an optimal development experience, we recommend using VS Code with the following extensions installed:

*   Python
*   Ruff
*   MyPy

You will be prompted to install these extensions when you open the project in VS Code.