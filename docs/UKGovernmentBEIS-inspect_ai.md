# Inspect: Evaluate and Enhance Your Large Language Models

**Unleash the power of rigorous evaluation with Inspect, the AI Security Institute's framework for comprehensive LLM assessment.**  ([Original Repo](https://github.com/UKGovernmentBEIS/inspect_ai))

[<img width="295" src="https://inspect.aisi.org.uk/images/aisi-logo.svg" alt="AI Security Institute Logo" />](https://aisi.gov.uk/)

Inspect, developed by the UK AI Security Institute, is a powerful framework designed to help you thoroughly evaluate and refine your Large Language Models (LLMs).  It provides a flexible and extensible platform for various evaluation tasks, ensuring the safety and performance of your AI applications.

## Key Features of Inspect:

*   **Comprehensive Evaluation:** Built-in components facilitate prompt engineering, tool usage analysis, and multi-turn dialog evaluation.
*   **Model-Graded Evaluations:**  Offers robust facilities for assessing model performance.
*   **Extensible Architecture:** Designed for customizability; easily integrate new elicitation and scoring techniques via Python packages.
*   **Developed by Experts:** Built by the UK AI Security Institute, ensuring high standards of security and reliability.
*   **Detailed Documentation:** Comprehensive documentation is available at <https://inspect.aisi.org.uk/> to guide you through implementation.

## Getting Started with Inspect:

Ready to start evaluating your LLMs? Here's how to get up and running with Inspect:

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/UKGovernmentBEIS/inspect_ai.git
    cd inspect_ai
    ```

2.  **Install Inspect with Development Dependencies:**

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

### VS Code Users

If you are using Visual Studio Code, ensure you install the recommended extensions (Python, Ruff, and MyPy). You will be prompted to install these when opening the project within VS Code.