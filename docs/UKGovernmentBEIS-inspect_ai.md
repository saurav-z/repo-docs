<div align="center">
  <a href="https://aisi.gov.uk/">
    <img width="295" src="https://inspect.aisi.org.uk/images/aisi-logo.svg" alt="Inspect Logo">
  </a>
</div>

# Inspect: The AI Safety Evaluation Framework

**Safeguard your AI systems with Inspect, the robust framework developed by the UK AI Safety Institute for comprehensive large language model (LLM) evaluations.**

## Key Features:

*   **Versatile Evaluation Capabilities:** Offers a comprehensive suite of tools for prompt engineering, tool usage analysis, and multi-turn dialog evaluation.
*   **Model-Graded Evaluations:** Includes built-in facilities for evaluating LLM performance with model-based scoring.
*   **Extensible Architecture:** Designed for extensibility, allowing you to integrate custom elicitation and scoring techniques through Python packages.
*   **Developed by AI Safety Experts:** Created by the UK AI Safety Institute (AISI), ensuring a focus on safety and responsible AI development.

## Getting Started

For detailed information on how to use Inspect, please refer to the official documentation: [https://inspect.aisi.org.uk/](https://inspect.aisi.org.uk/)

## Contributing & Development

Contribute to the development of Inspect and help improve the safety and reliability of AI.

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/UKGovernmentBEIS/inspect_ai.git
    cd inspect_ai
    ```

2.  **Install Dependencies (Development Mode):**

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

### Recommended VS Code Extensions:

Ensure you have the following extensions installed in VS Code for optimal development:

*   Python
*   Ruff
*   MyPy

These extensions will be prompted for installation when you open the project in VS Code.

**[View the Original Repository on GitHub](https://github.com/UKGovernmentBEIS/inspect_ai)**