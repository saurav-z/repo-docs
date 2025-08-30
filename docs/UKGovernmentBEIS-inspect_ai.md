# Inspect: Evaluate and Improve Your Large Language Models

**Easily evaluate and refine your LLMs with Inspect, a powerful framework developed by the UK AI Safety Institute.**

[<img width="295" src="https://inspect.aisi.org.uk/images/aisi-logo.svg" alt="AI Safety Institute Logo" />](https://aisi.gov.uk/)

Inspect is a robust Python framework designed for comprehensive Large Language Model (LLM) evaluation.  It provides a streamlined approach to assess and enhance LLM performance across various applications.

**Key Features of Inspect:**

*   **Built-in Components:** Leverage ready-to-use modules for prompt engineering, tool usage analysis, multi-turn dialog evaluations, and model-graded assessments.
*   **Extensible Architecture:** Easily integrate custom components and extend functionality by creating new Python packages to support diverse elicitation and scoring techniques.
*   **Modular Design:**  Flexible for various evaluation scenarios.
*   **Developed by Experts:** Created by the UK AI Safety Institute, ensuring a focus on safety and reliability.

**Getting Started**

For detailed information on installation, usage, and advanced features, please consult the official Inspect documentation:  <https://inspect.aisi.org.uk/>.

**Development and Contribution**

Interested in contributing to Inspect?  Follow these steps to get started:

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

4.  **Run Checks and Tests:** Ensure code quality and functionality with these commands:
    ```bash
    make check
    make test
    ```

**Recommended IDE Setup (VS Code)**

To optimize your development workflow, install the following VS Code extensions:

*   Python
*   Ruff
*   MyPy

You will be prompted to install these extensions when opening the project in VS Code.
```