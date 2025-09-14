# Inspect: Evaluate Large Language Models with Ease (Official Framework)

<p align="center">
    <a href="https://aisi.gov.uk/">
        <img src="https://inspect.aisi.org.uk/images/aisi-logo.svg" alt="UK AI Security Institute Logo" width="295"/>
    </a>
</p>

**Inspect, developed by the UK AI Security Institute, is your go-to framework for comprehensive evaluation of Large Language Models (LLMs).**

This powerful Python-based framework allows you to rigorously assess LLMs, providing a flexible and extensible platform for your evaluation needs.

## Key Features:

*   **Built-in Components:** Access pre-built modules for prompt engineering, tool usage, multi-turn dialogue analysis, and model-graded evaluations.
*   **Extensible Architecture:** Easily integrate custom components and extensions to support new elicitation and scoring techniques.
*   **Official Framework:** Backed by the UK AI Security Institute, ensuring quality and relevance for cutting-edge AI evaluation.

## Getting Started

For detailed instructions on using Inspect, including documentation and tutorials, please visit the official documentation at: <https://inspect.aisi.org.uk/>

## Development & Contribution

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

3.  **(Optional) Install Pre-Commit Hooks:**

    ```bash
    make hooks
    ```

4.  **Run Quality Checks:**

    ```bash
    make check
    make test
    ```

5.  **Recommended VS Code Extensions:** Python, Ruff, and MyPy.

**Original Repository:** [https://github.com/UKGovernmentBEIS/inspect_ai](https://github.com/UKGovernmentBEIS/inspect_ai)