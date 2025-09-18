# Inspect: Evaluate and Enhance Large Language Models (LLMs)

**Assess and improve your LLMs with Inspect, a powerful framework developed by the UK AI Safety Institute (UK AISI).**

[<img width="295" src="https://inspect.aisi.org.uk/images/aisi-logo.svg" alt="UK AI Safety Institute Logo" />](https://aisi.gov.uk/)

Inspect empowers you to rigorously evaluate and enhance your large language models. This robust framework offers a comprehensive suite of tools for:

*   **Prompt Engineering:** Design and refine prompts to optimize LLM performance.
*   **Tool Usage:** Integrate and evaluate LLMs' ability to effectively use tools.
*   **Multi-Turn Dialog:** Assess LLMs' capabilities in complex, multi-turn conversations.
*   **Model-Graded Evaluations:** Implement sophisticated evaluation techniques to measure LLM effectiveness.
*   **Extensibility:** Easily integrate custom elicitation and scoring methods through Python packages.

**Get Started with Inspect**

For detailed information and usage instructions, please refer to the official Inspect documentation: <https://inspect.aisi.org.uk/>

**Contributing and Development**

Interested in contributing to or developing Inspect? Follow these steps:

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

4.  **Run Linting, Formatting, and Tests:**
    ```bash
    make check
    make test
    ```

**VS Code Integration**

For an optimal development experience, we recommend using Visual Studio Code (VS Code) with the following extensions installed:

*   Python
*   Ruff
*   MyPy

You'll be prompted to install these extensions when you open the project in VS Code.

**[Learn more and contribute on GitHub](https://github.com/UKGovernmentBEIS/inspect_ai)**