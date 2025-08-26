# Inspect: Evaluate & Improve Your Large Language Models (LLMs)

[![Inspect AI Security Institute Logo](https://inspect.aisi.org.uk/images/aisi-logo.svg)](https://aisi.gov.uk/)

**Unleash the power of thorough LLM evaluation with Inspect, the open-source framework from the UK AI Safety Institute.**

Inspect empowers researchers and developers to rigorously assess and refine their Large Language Models (LLMs).  Built by the UK AI Safety Institute, this framework provides a comprehensive suite of tools for evaluating LLMs.

**Key Features:**

*   **Built-in Components:** Offers a rich set of tools for various evaluation tasks.
*   **Prompt Engineering:** Streamline your prompt design and optimization efforts.
*   **Tool Usage Evaluation:** Assess how effectively your LLMs utilize tools.
*   **Multi-Turn Dialogue Analysis:** Evaluate LLMs in complex, multi-turn conversational scenarios.
*   **Model-Graded Evaluations:** Provides robust mechanisms for scoring and comparing LLM performance.
*   **Extensible Architecture:** Easily integrate custom elicitation and scoring techniques through Python package extensions.

**Getting Started:**

For detailed instructions on using Inspect, including installation and usage examples, please refer to the official documentation: [https://inspect.aisi.org.uk/](https://inspect.aisi.org.uk/)

**Development & Contribution:**

Interested in contributing to Inspect? Follow these steps to get started:

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

**Recommended VS Code Extensions:**

If you use VS Code, we recommend installing the following extensions for optimal development:

*   Python
*   Ruff
*   MyPy

These extensions will be prompted for installation upon opening the project in VS Code.

**Contribute to Inspect on GitHub:** [https://github.com/UKGovernmentBEIS/inspect_ai](https://github.com/UKGovernmentBEIS/inspect_ai)