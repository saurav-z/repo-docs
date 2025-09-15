# Inspect: Evaluate and Enhance Your Large Language Models

**Unlock the power of robust and reliable AI with Inspect, the framework designed for comprehensive large language model (LLM) evaluation.** Developed by the UK AI Safety Institute, Inspect empowers you to assess, refine, and optimize your LLMs for peak performance. ([See the original repository](https://github.com/UKGovernmentBEIS/inspect_ai))

## Key Features of Inspect:

*   **Comprehensive Evaluation:** Rigorously assess LLMs across various dimensions, including accuracy, safety, and reliability.
*   **Prompt Engineering Capabilities:** Experiment with and refine prompts to optimize model responses and performance.
*   **Tool Usage Support:** Seamlessly integrate and evaluate LLMs that utilize tools and external resources.
*   **Multi-Turn Dialog Analysis:** Analyze and evaluate complex, multi-turn conversations for enhanced understanding.
*   **Model-Graded Evaluations:** Leverage built-in components for efficient and automated model grading.
*   **Extensible Architecture:** Easily extend Inspect with custom components and integrations to support new evaluation techniques.

## Getting Started with Inspect

For detailed instructions and usage examples, refer to the official documentation: [https://inspect.aisi.org.uk/](https://inspect.aisi.org.uk/)

## Development Setup

To contribute to Inspect or develop custom extensions, follow these steps:

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

5.  **VS Code Recommended Extensions:** Ensure you have the Python, Ruff, and MyPy extensions installed in VS Code for optimal development.  VS Code will prompt you to install these when you open the project.