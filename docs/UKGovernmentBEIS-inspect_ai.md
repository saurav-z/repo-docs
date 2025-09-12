# Inspect: Evaluate and Enhance Your Large Language Models

**Safeguard your AI: Inspect, the evaluation framework developed by the UK AI Safety Institute, helps you understand and improve your large language models (LLMs).**

[![AI Safety Institute Logo](https://inspect.aisi.org.uk/images/aisi-logo.svg)](https://aisi.gov.uk/)

This repository provides the source code for Inspect, a powerful framework designed for comprehensive large language model evaluation. Inspect empowers developers and researchers to rigorously assess and enhance their LLMs.

**Key Features:**

*   **Prompt Engineering:** Easily craft and refine prompts to elicit specific behaviors from your LLMs.
*   **Tool Usage:** Evaluate how well your LLMs utilize external tools and APIs.
*   **Multi-Turn Dialog Support:** Analyze complex conversational interactions with multi-turn dialogue evaluation.
*   **Model-Graded Evaluations:** Implement sophisticated evaluation techniques to assess model performance.
*   **Extensible Architecture:**  Extend Inspect with custom components and integrate new elicitation and scoring techniques.

**Getting Started**

1.  **Documentation:**  Comprehensive documentation is available at <https://inspect.aisi.org.uk/>.

2.  **Installation (for Development):**

    ```bash
    git clone https://github.com/UKGovernmentBEIS/inspect_ai.git
    cd inspect_ai
    pip install -e ".[dev]"
    ```

3.  **Optional Setup (for Contributing):**

    *   **Install pre-commit hooks:**
        ```bash
        make hooks
        ```

    *   **Run linting, formatting, and tests:**
        ```bash
        make check
        make test
        ```

    *   **VS Code Recommendations:** We recommend using VS Code with the following extensions installed: Python, Ruff, and MyPy. You'll be prompted to install these when opening the project.

[**View the original repository on GitHub**](https://github.com/UKGovernmentBEIS/inspect_ai)