# Inspect: Evaluate and Enhance Large Language Models (LLMs)

[![UK AI Security Institute Logo](https://inspect.aisi.org.uk/images/aisi-logo.svg)](https://aisi.gov.uk/)

**Unlock the power of AI with Inspect, the comprehensive framework for rigorous Large Language Model (LLM) evaluation developed by the UK AI Safety Institute.**

Inspect empowers you to thoroughly assess and refine your LLMs, ensuring their reliability and safety.  This open-source Python framework offers a robust set of features designed to streamline your LLM evaluation process.

## Key Features:

*   **Built-in Components:**  Includes powerful tools for prompt engineering, effective tool usage, multi-turn dialog management, and comprehensive model-graded evaluations.
*   **Extensible Architecture:**  Easily integrate new evaluation techniques and scoring methods by extending Inspect with custom Python packages.
*   **Open-Source & Community-Driven:** Benefit from a framework developed by the UK AI Safety Institute and contribute to its continuous improvement.
*   **Comprehensive Documentation:**  Detailed documentation is available at [https://inspect.aisi.org.uk/](https://inspect.aisi.org.uk/) to guide you through every step of the process.

## Getting Started

To begin using Inspect, follow these steps:

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/UKGovernmentBEIS/inspect_ai.git
    cd inspect_ai
    ```

2.  **Install Dependencies (with development dependencies):**
    ```bash
    pip install -e ".[dev]"
    ```

3.  **(Optional) Install Pre-commit Hooks:**
    ```bash
    make hooks
    ```

4.  **(Optional) Run Linting, Formatting, and Tests:**
    ```bash
    make check
    make test
    ```

### Recommended IDE:

If you're using Visual Studio Code, ensure you have the recommended extensions installed (Python, Ruff, and MyPy).  VS Code will prompt you to install these extensions when you open the project.

**[View the full Inspect repository on GitHub](https://github.com/UKGovernmentBEIS/inspect_ai)**