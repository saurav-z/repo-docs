# Inspect: Evaluate and Enhance Large Language Models with AI Security in Mind

[<img width="295" src="https://inspect.aisi.org.uk/images/aisi-logo.svg" alt="AI Security Institute Logo" />](https://aisi.gov.uk/)

**Inspect, developed by the UK AI Security Institute, is your comprehensive framework for rigorously evaluating and improving Large Language Models (LLMs).**

## Key Features

*   **Comprehensive Evaluation:** Assess your LLMs using a variety of techniques.
*   **Prompt Engineering Tools:** Experiment with different prompts to optimize model performance.
*   **Tool Usage Support:**  Evaluate how well your models utilize tools.
*   **Multi-Turn Dialog Evaluation:** Analyze and improve LLMs' conversational abilities.
*   **Model-Graded Evaluations:** Gain insights into model performance through automated grading.
*   **Extensible Architecture:** Easily integrate custom evaluation techniques via Python packages.

## Getting Started

For detailed information and guidance on using Inspect, please refer to the official documentation: [https://inspect.aisi.org.uk/](https://inspect.aisi.org.uk/)

## Development and Contribution

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

3.  **(Optional) Install Pre-commit Hooks:**

    ```bash
    make hooks
    ```

4.  **Run Checks and Tests:**

    ```bash
    make check
    make test
    ```

    These commands will run linting, formatting, and tests to ensure code quality.

## VS Code Recommended Extensions

For an optimal development experience with Inspect, we recommend installing the following VS Code extensions:

*   Python
*   Ruff
*   MyPy

You will be prompted to install these extensions when opening the project in VS Code.