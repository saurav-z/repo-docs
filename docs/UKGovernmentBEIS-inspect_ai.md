# Inspect: Evaluate and Enhance Your Large Language Models

**Unleash the power of thorough evaluation with Inspect, a robust framework designed to help you understand and improve your large language models (LLMs).** (See the original repository on GitHub: [UKGovernmentBEIS/inspect_ai](https://github.com/UKGovernmentBEIS/inspect_ai)).

[<img width="295" src="https://inspect.aisi.org.uk/images/aisi-logo.svg" alt="AI Security Institute Logo" />](https://aisi.gov.uk/)

Developed by the [UK AI Security Institute](https://aisi.gov.uk/), Inspect provides a comprehensive suite of tools for rigorously testing and refining your LLMs.

## Key Features of Inspect:

*   **Prompt Engineering Capabilities:** Design and optimize prompts for effective LLM interaction.
*   **Tool Usage Support:** Integrate and evaluate LLMs that leverage external tools.
*   **Multi-Turn Dialog Evaluation:** Assess LLM performance in complex, multi-turn conversations.
*   **Model-Graded Evaluations:** Utilize robust evaluation methods to gauge model performance.
*   **Extensible Architecture:** Easily extend Inspect with custom components and support new elicitation and scoring techniques through Python packages.

## Getting Started

For detailed information on how to use Inspect, please refer to the comprehensive documentation available at <https://inspect.aisi.org.uk/>.

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
3.  **(Optional) Install Pre-Commit Hooks:**
    ```bash
    make hooks
    ```
4.  **Run Checks and Tests:**
    ```bash
    make check
    make test
    ```

### Recommended Development Environment

If you're using VS Code, we recommend installing the following extensions for optimal development:

*   Python
*   Ruff
*   MyPy

You'll be prompted to install these extensions when you open the project in VS Code.