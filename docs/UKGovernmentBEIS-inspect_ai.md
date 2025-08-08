[<img width="295" src="https://inspect.aisi.org.uk/images/aisi-logo.svg" />](https://aisi.gov.uk/)

# Inspect: Evaluate & Enhance Your Large Language Models

**Inspect is a powerful framework developed by the UK AI Safety Institute for comprehensive Large Language Model (LLM) evaluation and improvement.**

## Key Features

*   **Comprehensive Evaluation:** Assess your LLMs across various critical aspects.
*   **Prompt Engineering:** Experiment and refine prompts for optimal performance.
*   **Tool Usage Support:** Evaluate LLMs' ability to effectively use tools.
*   **Multi-turn Dialog Analysis:** Analyze LLMs in complex conversational scenarios.
*   **Graded Evaluations:** Implement model-graded assessments for nuanced insights.
*   **Extensible Architecture:** Easily add custom elicitation and scoring techniques through Python packages.

## Getting Started

To begin using Inspect, consult the detailed documentation available at [https://inspect.aisi.org.uk/](https://inspect.aisi.org.uk/).

## Development & Contribution

Interested in contributing to Inspect? Here's how to set up your development environment:

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

### Recommended VS Code Extensions

For the best development experience, install the following VS Code extensions:

*   Python
*   Ruff
*   MyPy

You will be prompted to install these extensions when opening the project in VS Code.

**[View the original repository on GitHub](https://github.com/UKGovernmentBEIS/inspect_ai)**