# Inspect: Evaluate and Enhance Large Language Models (LLMs)

**Tired of guesswork in AI?** Inspect, a powerful framework from the UK AI Safety Institute, empowers you to rigorously evaluate and refine your large language models.

[![Inspect Logo](https://inspect.aisi.org.uk/images/aisi-logo.svg)](https://aisi.gov.uk/)

Inspect offers a robust suite of tools for comprehensive LLM assessment. This framework allows you to:

*   **Streamline Evaluation:** Leverage built-in components for prompt engineering, tool usage analysis, and multi-turn dialog assessments.
*   **Graded Evaluations:** Implement model-graded evaluations to gain deeper insights.
*   **Extendable Functionality:** Easily integrate new elicitation and scoring techniques through Python package extensions.
*   **Open-Source:** Built by the UK AI Safety Institute to support responsible AI development.

**Get Started with Inspect**

Explore the comprehensive documentation and user guides at [https://inspect.aisi.org.uk/](https://inspect.aisi.org.uk/).

**Contributing and Development**

To contribute to Inspect or develop your own extensions:

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/UKGovernmentBEIS/inspect_ai.git
    cd inspect_ai
    ```

2.  **Install with Development Dependencies:**

    ```bash
    pip install -e ".[dev]"
    ```

3.  **(Optional) Install Pre-Commit Hooks:**

    ```bash
    make hooks
    ```

4.  **Run Checks and Tests:**

    ```bash
    make check
    make test
    ```

**Recommended Development Environment:**

For optimal development experience, we recommend using VS Code with the following extensions installed: Python, Ruff, and MyPy. You will be prompted to install these when you open the project in VS Code.

**[View the original repository on GitHub](https://github.com/UKGovernmentBEIS/inspect_ai)**