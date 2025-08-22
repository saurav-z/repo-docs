# Inspect: Evaluate & Enhance Your AI Models with Precision

**Are you seeking to rigorously assess and refine your large language models?**  Inspect, a powerful framework from the UK AI Safety Institute, empowers you to do just that.

[<img width="295" src="https://inspect.aisi.org.uk/images/aisi-logo.svg" alt="UK AI Safety Institute Logo" />](https://aisi.gov.uk/)

## Key Features of Inspect:

*   **Comprehensive Evaluation:** Assess LLMs across various dimensions, including prompt engineering, tool usage, and multi-turn dialogues.
*   **Modular Design:**  Leverage built-in components or extend functionality with custom Python packages to support new elicitation and scoring techniques.
*   **Multi-Turn Dialog:** Inspect can be used in evaluating multi-turn LLM interactions.
*   **Model-Graded Evaluations:** Evaluate with model-graded evaluations.

## Getting Started with Inspect

For comprehensive documentation and usage examples, please visit the official documentation at: <https://inspect.aisi.org.uk/>

## Developing Inspect

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

## Recommended IDE Setup (VS Code)

For the best development experience, we recommend using VS Code with the following extensions:

*   Python
*   Ruff
*   MyPy

You will be prompted to install these extensions when you open the project in VS Code.

**Original Repository:** Find the source code and further information on the [Inspect GitHub repository](https://github.com/UKGovernmentBEIS/inspect_ai).