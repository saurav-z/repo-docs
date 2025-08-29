# Inspect: Evaluate and Enhance Your Large Language Models

**Unleash the power of comprehensive evaluation for your LLMs with Inspect, a robust framework from the UK AI Safety Institute.**

[![Inspect AI Logo](https://inspect.aisi.org.uk/images/aisi-logo.svg)](https://aisi.gov.uk/)

Inspect is an open-source Python framework designed to help researchers and developers rigorously evaluate and improve the performance of large language models (LLMs). Developed by the UK AI Safety Institute, Inspect provides a comprehensive suite of tools for assessing LLM capabilities and identifying areas for enhancement.

## Key Features:

*   **Versatile Evaluation Components:** Built-in support for prompt engineering, tool usage assessment, multi-turn dialog analysis, and model-graded evaluations.
*   **Extensible Architecture:** Easily extend Inspect with custom components to support new elicitation and scoring techniques, ensuring adaptability to emerging LLM evaluation methodologies.
*   **Python-Based:** Designed for easy integration into existing Python-based LLM development workflows.
*   **Open Source:** Developed and maintained by the UK AI Safety Institute, with the code available on GitHub and a dedicated documentation site.

## Getting Started

For detailed information on how to use Inspect, including installation instructions, usage examples, and API documentation, please visit the official documentation: [https://inspect.aisi.org.uk/](https://inspect.aisi.org.uk/)

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

4.  **Run Tests and Linting:**

    ```bash
    make check
    make test
    ```

### VS Code Setup (Recommended)

For the best development experience, use VS Code and install the following recommended extensions when prompted:

*   Python
*   Ruff
*   MyPy

## Resources

*   **Source Code:** [View the Inspect Repository on GitHub](https://github.com/UKGovernmentBEIS/inspect_ai)
*   **Documentation:** [https://inspect.aisi.org.uk/](https://inspect.aisi.org.uk/)