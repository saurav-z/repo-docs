## Inspect: Evaluate Large Language Models with Confidence

**Unleash the power of comprehensive LLM evaluation with Inspect, a framework built by the UK AI Security Institute.** ([Original Repository](https://github.com/UKGovernmentBEIS/inspect_ai))

[<img width="295" src="https://inspect.aisi.org.uk/images/aisi-logo.svg" alt="UK AI Security Institute Logo" />](https://aisi.gov.uk/)

Inspect provides a robust and flexible platform for rigorously assessing and refining your large language models. Whether you're focused on prompt engineering, tool usage, multi-turn dialogues, or model grading, Inspect equips you with the tools you need to ensure LLM reliability and performance.

### Key Features:

*   **Built-in Components:** Leverage pre-built functionalities for various LLM evaluation aspects, streamlining your workflow.
*   **Prompt Engineering Support:** Facilitate the design, testing, and optimization of effective prompts.
*   **Tool Usage Evaluation:** Analyze and refine the performance of LLMs when interacting with external tools.
*   **Multi-Turn Dialogue Capabilities:** Evaluate LLMs in complex, conversational scenarios.
*   **Model-Graded Evaluations:** Benefit from built-in tools for assessing model output.
*   **Extensible Architecture:** Seamlessly integrate custom evaluation techniques through Python packages.
*   **Developed by Experts:** Built by the UK AI Security Institute, ensuring a focus on security and reliability.

### Getting Started with Inspect

For detailed information on using Inspect, please consult the official documentation: <https://inspect.aisi.org.uk/>

### Development Setup

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

5.  **(Recommended) VS Code Setup:** Install recommended extensions (Python, Ruff, and MyPy) when prompted upon opening the project in VS Code for an optimal development experience.